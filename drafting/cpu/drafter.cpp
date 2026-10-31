// CPU drafter (M5.T2): the per-token algorithm of drafting/cpu_drafter.py (RefDrafter) in C++.
//
//   ext = torch.utils.cpp_extension.load("morbius_cpu_drafter", ["drafting/cpu/drafter.cpp"])
//   core = ext.Drafter(tensors, numbers, ngrams, threads)   # see CppDrafter in cpu_drafter.py
//   core.commit(tokens, codes_or_None); core.draft(gamma, hyp, topk, last_codes); core.truncate(n)
//
// fp32 compute; tables fp16 (or fp32).  Every entry point releases the GIL for the whole call
// and holds a per-object mutex (one call at a time per drafter).  With threads > 1 the R
// routes' row gather + memory norm + KV projection (the dominant cost) are split over a
// persistent worker pool; everything else is single-threaded.  Results do not depend on the
// thread count (each route is computed by exactly one thread with the same arithmetic).

#include <torch/extension.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace {

// ---------------------------------------------------------------- worker pool
class Pool {
 public:
  explicit Pool(int threads) : nthreads_(std::max(1, threads)) {
    for (int i = 1; i < nthreads_; ++i) workers_.emplace_back([this, i] { loop(i); });
  }
  ~Pool() {
    {
      std::lock_guard<std::mutex> g(mu_);
      stop_ = true;
      ++gen_;
    }
    cv_.notify_all();
    for (auto& w : workers_) w.join();
  }
  int size() const { return nthreads_; }

  // fn(i) for i in [0, n); contiguous chunks, the caller runs chunk 0.
  void run(int n, const std::function<void(int)>& fn) {
    if (nthreads_ <= 1 || n <= 1) {
      for (int i = 0; i < n; ++i) fn(i);
      return;
    }
    {
      std::lock_guard<std::mutex> g(mu_);
      job_ = &fn;
      job_n_ = n;
      pending_ = nthreads_ - 1;
      ++gen_;
    }
    cv_.notify_all();
    chunk(0, fn, n);
    std::unique_lock<std::mutex> lk(mu_);
    done_cv_.wait(lk, [this] { return pending_ == 0; });
    job_ = nullptr;
  }

 private:
  void chunk(int tid, const std::function<void(int)>& fn, int n) const {
    const int per = (n + nthreads_ - 1) / nthreads_;
    const int lo = tid * per, hi = std::min(n, lo + per);
    for (int i = lo; i < hi; ++i) fn(i);
  }
  void loop(int tid) {
    uint64_t seen = 0;
    for (;;) {
      const std::function<void(int)>* fn;
      int n;
      {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return gen_ != seen; });
        seen = gen_;
        if (stop_) return;
        fn = job_;
        n = job_n_;
      }
      chunk(tid, *fn, n);
      {
        std::lock_guard<std::mutex> g(mu_);
        if (--pending_ == 0) done_cv_.notify_one();
      }
    }
  }
  int nthreads_;
  std::vector<std::thread> workers_;
  std::mutex mu_;
  std::condition_variable cv_, done_cv_;
  const std::function<void(int)>* job_ = nullptr;
  int job_n_ = 0;
  int pending_ = 0;
  uint64_t gen_ = 0;
  bool stop_ = false;
};

// ---------------------------------------------------------------- math helpers
inline float dot(const float* a, const float* b, int64_t n) {
  float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
  int64_t i = 0;
  for (; i + 4 <= n; i += 4) {
    s0 += a[i] * b[i];
    s1 += a[i + 1] * b[i + 1];
    s2 += a[i + 2] * b[i + 2];
    s3 += a[i + 3] * b[i + 3];
  }
  for (; i < n; ++i) s0 += a[i] * b[i];
  return (s0 + s1) + (s2 + s3);
}

// y[i] = W[i, :] . x   (W row-major [rows, cols])
inline void matvec(const float* W, const float* x, float* y, int64_t rows, int64_t cols) {
  for (int64_t i = 0; i < rows; ++i) y[i] = dot(W + i * cols, x, cols);
}

// out = x * rsqrt(mean(x^2) + eps) * w
inline void rmsnorm(const float* x, const float* w, float eps, float* out, int64_t n) {
  double ss = 0.0;
  for (int64_t i = 0; i < n; ++i) ss += static_cast<double>(x[i]) * x[i];
  const float r = 1.0f / std::sqrt(static_cast<float>(ss / static_cast<double>(n)) + eps);
  for (int64_t i = 0; i < n; ++i) out[i] = x[i] * r * w[i];
}

const float* fptr(const std::map<std::string, at::Tensor>& t, const std::string& k, bool required = true) {
  auto it = t.find(k);
  if (it == t.end()) {
    if (required) throw std::runtime_error("bundle tensor missing: " + k);
    return nullptr;
  }
  TORCH_CHECK(it->second.scalar_type() == at::kFloat && it->second.is_contiguous(), k, " must be contiguous fp32");
  return it->second.data_ptr<float>();
}

struct Cursor {
  int64_t t = -1;    // position of the last processed token
  int64_t tok = -1;
  std::vector<float> s;           // SSM state at t
  std::vector<uint8_t> hist;      // maxn x R codes, oldest first (rows valid for the last min(t+1, maxn))
};

// ---------------------------------------------------------------- drafter
class Drafter {
 public:
  Drafter(const std::map<std::string, at::Tensor>& tensors, const std::map<std::string, double>& num,
          const std::vector<int64_t>& ngrams, int64_t threads)
      : keep_(tensors), ngrams_(ngrams), pool_(static_cast<int>(threads)) {
    auto get = [&](const char* k) {
      auto it = num.find(k);
      if (it == num.end()) throw std::runtime_error(std::string("missing number: ") + k);
      return it->second;
    };
    d_ = static_cast<int64_t>(get("d_model"));
    V_ = static_cast<int64_t>(get("vocab"));
    R_ = static_cast<int64_t>(get("routes"));
    bits_ = static_cast<int64_t>(get("bits"));
    mem_dim_ = static_cast<int64_t>(get("mem_dim"));
    kv_ = static_cast<int64_t>(get("kv_heads"));
    hd_ = static_cast<int64_t>(get("head_dim"));
    G_ = static_cast<int64_t>(get("q_heads")) / kv_;
    N_ = static_cast<int64_t>(get("ssm_dim"));
    K_ = static_cast<int64_t>(get("sketch_k"));
    cap_ = static_cast<int64_t>(get("shortlist_cap"));
    router_ssm_ = get("router_ssm") != 0.0;
    query_ssm_ = get("query_ssm") != 0.0;
    ssm_eps_ = static_cast<float>(get("ssm_eps"));
    router_eps_ = static_cast<float>(get("router_eps"));
    draft_eps_ = static_cast<float>(get("draft_eps"));
    mem_eps_ = static_cast<float>(get("mem_eps"));
    scale_ = static_cast<float>(get("softmax_scale"));
    sink_ = static_cast<float>(get("sink_logit"));
    maxn_ = *std::max_element(ngrams_.begin(), ngrams_.end());
    Dt_ = static_cast<int64_t>(ngrams_.size()) * mem_dim_;
    KH_ = kv_ * hd_;
    I_ = kv_ * G_ * hd_;
    F_ = 2 * d_ + (router_ssm_ ? N_ : 0);

    E_ = fptr(keep_, "embedding");
    mem_norm_ = fptr(keep_, "mem_norm");
    kv_proj_ = fptr(keep_, "kv_proj");
    router_ = fptr(keep_, "router");
    prev_norm_ = fptr(keep_, "router_prev_norm");
    cur_norm_ = fptr(keep_, "router_cur_norm");
    start_ = fptr(keep_, "router_start");
    has_ssm_ = N_ > 0;
    if (has_ssm_) {
      ssm_norm_ = fptr(keep_, "ssm_norm");
      ssm_w_in_ = fptr(keep_, "ssm_w_in");
      ssm_a_ = fptr(keep_, "ssm_decay");
    }
    q_norm_ = fptr(keep_, "q_norm");
    w_q_ = fptr(keep_, "w_q");
    w_sq_ = query_ssm_ ? fptr(keep_, "w_sq") : nullptr;
    w_o_ = fptr(keep_, "w_o");
    out_norm_ = fptr(keep_, "out_norm");
    TORCH_CHECK(keep_.at("router").numel() == R_ * bits_ * F_, "router weight shape");
    TORCH_CHECK(keep_.at("kv_proj").numel() == 2 * KH_ * Dt_, "kv_proj shape");
    for (int64_t n : ngrams_) {
      const at::Tensor& tb = keep_.at("table_" + std::to_string(n));
      TORCH_CHECK(tb.is_contiguous() && tb.size(1) == mem_dim_, "table shape");
      TORCH_CHECK(tb.scalar_type() == at::kHalf || tb.scalar_type() == at::kFloat, "table dtype fp16/fp32");
      tables_.push_back(tb.data_ptr());
      table_half_.push_back(tb.scalar_type() == at::kHalf);
      if (K_ > 0) {
        const at::Tensor& si = keep_.at("sketch_ids_" + std::to_string(n));
        const at::Tensor& sc = keep_.at("sketch_counts_" + std::to_string(n));
        TORCH_CHECK(si.scalar_type() == at::kInt && sc.scalar_type() == at::kInt, "sketch arrays must be int32");
        sk_ids_.push_back(si.data_ptr<int32_t>());
        sk_cnt_.push_back(sc.data_ptr<int32_t>());
      }
    }
    auto fb = keep_.find("fallback");
    if (fb != keep_.end() && fb->second.numel() > 0) {
      TORCH_CHECK(fb->second.scalar_type() == at::kInt, "fallback must be int32");
      const int32_t* p = fb->second.data_ptr<int32_t>();
      fallback_.assign(p, p + fb->second.numel());
    }
    // work buffers
    rt_.resize(R_ * Dt_);
    kvbuf_.resize(R_ * 2 * KH_);
    acc_.assign(V_, 0);
    mark_.assign(V_, 0);
  }

  // ------------------------------------------------------------ state API
  void reset() {
    py::gil_scoped_release nogil;
    std::lock_guard<std::mutex> lk(mu_);
    tokens_.clear();
    codes_.clear();
    ssm_.clear();
  }

  int64_t length() const { return static_cast<int64_t>(tokens_.size()); }

  void truncate(int64_t n) {
    py::gil_scoped_release nogil;
    std::lock_guard<std::mutex> lk(mu_);
    TORCH_CHECK(n >= 0 && n <= length(), "cannot truncate to ", n);
    tokens_.resize(n);
    codes_.resize(n * R_);
    if (has_ssm_) ssm_.resize(n * N_);
  }

  at::Tensor committed_codes() {
    std::lock_guard<std::mutex> lk(mu_);
    auto out = torch::empty({length(), R_}, torch::kUInt8);
    std::copy(codes_.begin(), codes_.end(), out.data_ptr<uint8_t>());
    return out;
  }

  void commit(const std::vector<int64_t>& tokens, std::optional<at::Tensor> codes) {
    const int32_t* cp = nullptr;
    at::Tensor ct;
    if (codes.has_value()) {
      ct = codes->to(torch::kInt).contiguous();
      TORCH_CHECK(ct.dim() == 2 && ct.size(0) == static_cast<int64_t>(tokens.size()) && ct.size(1) == R_,
                  "codes must be [n, R]");
      cp = ct.data_ptr<int32_t>();
    }
    py::gil_scoped_release nogil;
    std::lock_guard<std::mutex> lk(mu_);
    Cursor c = cursor();
    std::vector<uint8_t> over(R_);
    for (size_t i = 0; i < tokens.size(); ++i) {
      check_token(tokens[i]);
      if (cp) {
        for (int64_t r = 0; r < R_; ++r) over[r] = static_cast<uint8_t>(cp[i * R_ + r]);
      }
      advance(c, tokens[i], cp ? over.data() : nullptr);
      tokens_.push_back(tokens[i]);
      codes_.insert(codes_.end(), c.hist.end() - R_, c.hist.end());
      if (has_ssm_) ssm_.insert(ssm_.end(), c.s.begin(), c.s.end());
    }
  }

  std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> draft(int64_t gamma, const std::vector<int64_t>& hyp,
                                                                   int64_t topk, bool last_codes) {
    TORCH_CHECK(gamma >= 0 && topk >= 1, "need gamma >= 0 and topk >= 1");
    auto toks = torch::empty({gamma}, torch::kLong);
    auto ids = torch::full({gamma, topk}, -1, torch::kLong);
    auto lgs = torch::full({gamma, topk}, -std::numeric_limits<float>::infinity(), torch::kFloat);
    const int64_t n_codes = static_cast<int64_t>(hyp.size()) + (last_codes ? gamma : std::max<int64_t>(gamma - 1, 0));
    auto codes = torch::zeros({n_codes, R_}, torch::kUInt8);
    {
      py::gil_scoped_release nogil;
      std::lock_guard<std::mutex> lk(mu_);
      Cursor c = cursor();
      if (c.t < 0 && hyp.empty()) throw std::runtime_error("draft needs at least one committed or hypothetical token");
      uint8_t* cp = codes.data_ptr<uint8_t>();
      int64_t* tp = toks.data_ptr<int64_t>();
      int64_t* ip = ids.data_ptr<int64_t>();
      float* lp = lgs.data_ptr<float>();
      int64_t ci = 0;
      for (int64_t tok : hyp) {
        check_token(tok);
        advance(c, tok, nullptr);
        std::copy(c.hist.end() - R_, c.hist.end(), cp + (ci++) * R_);
      }
      std::vector<int64_t> sl;
      std::vector<float> lg;
      for (int64_t i = 0; i < gamma; ++i) {
        readout(c, false, sl, lg);
        std::vector<int64_t> order = top_order(lg, topk);
        for (size_t j = 0; j < order.size(); ++j) {
          ip[i * topk + j] = sl[order[j]];
          lp[i * topk + j] = lg[order[j]];
        }
        const int64_t nxt = sl[order[0]];
        tp[i] = nxt;
        if (i + 1 < gamma || last_codes) {
          advance(c, nxt, nullptr);
          std::copy(c.hist.end() - R_, c.hist.end(), cp + (ci++) * R_);
        }
      }
    }
    return {toks, ids, lgs, codes};
  }

  std::tuple<at::Tensor, at::Tensor, at::Tensor> trace(const std::vector<int64_t>& tokens,
                                                       std::optional<at::Tensor> codes, bool full) {
    const int64_t T = static_cast<int64_t>(tokens.size());
    const bool full_vocab = full || cap_ <= 0 || K_ <= 0;
    const int64_t S = full_vocab ? V_ : cap_;
    auto oc = torch::zeros({T, R_}, torch::kUInt8);
    auto oi = torch::full({T, S}, -1, torch::kLong);
    auto ol = torch::full({T, S}, -std::numeric_limits<float>::infinity(), torch::kFloat);
    at::Tensor ct;
    const int32_t* cp = nullptr;
    if (codes.has_value()) {
      ct = codes->to(torch::kInt).contiguous();
      TORCH_CHECK(ct.dim() == 2 && ct.size(0) == T && ct.size(1) == R_, "codes must be [T, R]");
      cp = ct.data_ptr<int32_t>();
    }
    {
      py::gil_scoped_release nogil;
      std::lock_guard<std::mutex> lk(mu_);
      Cursor c;
      c.hist.assign(maxn_ * R_, 0);
      if (has_ssm_) c.s.assign(N_, 0.f);
      std::vector<uint8_t> over(R_);
      std::vector<int64_t> sl;
      std::vector<float> lg;
      uint8_t* ocp = oc.data_ptr<uint8_t>();
      int64_t* oip = oi.data_ptr<int64_t>();
      float* olp = ol.data_ptr<float>();
      for (int64_t t = 0; t < T; ++t) {
        check_token(tokens[t]);
        if (cp) {
          for (int64_t r = 0; r < R_; ++r) over[r] = static_cast<uint8_t>(cp[t * R_ + r]);
        }
        advance(c, tokens[t], cp ? over.data() : nullptr);
        std::copy(c.hist.end() - R_, c.hist.end(), ocp + t * R_);
        readout(c, full, sl, lg);
        const int64_t m = std::min<int64_t>(static_cast<int64_t>(sl.size()), S);
        for (int64_t j = 0; j < m; ++j) {
          oip[t * S + j] = sl[j];
          olp[t * S + j] = lg[j];
        }
      }
    }
    return {oc, oi, ol};
  }

 private:
  void check_token(int64_t tok) const {
    if (tok < 0 || tok >= V_) throw std::runtime_error("token id out of range");
  }

  Cursor cursor() const {
    Cursor c;
    c.hist.assign(maxn_ * R_, 0);
    if (has_ssm_) c.s.assign(N_, 0.f);
    const int64_t n = length();
    if (n == 0) return c;
    c.t = n - 1;
    c.tok = tokens_.back();
    if (has_ssm_) std::copy(ssm_.end() - N_, ssm_.end(), c.s.begin());
    const int64_t h = std::min<int64_t>(n, maxn_);
    // the last h committed codes go to the last h slots of hist
    std::copy(codes_.end() - h * R_, codes_.end(), c.hist.end() - h * R_);
    return c;
  }

  // Process token `tok` after the cursor: SSM step, router codes (or `code_override`), shift history.
  void advance(Cursor& c, int64_t tok, const uint8_t* code_override) {
    const float* e = E_ + tok * d_;
    if (has_ssm_) {
      std::vector<float> xn(d_), u(N_);
      rmsnorm(e, ssm_norm_, ssm_eps_, xn.data(), d_);
      matvec(ssm_w_in_, xn.data(), u.data(), N_, d_);
      for (int64_t i = 0; i < N_; ++i) c.s[i] = ssm_a_[i] * c.s[i] + (1.f - ssm_a_[i]) * u[i];
    }
    std::vector<uint8_t> code(R_, 0);
    if (code_override) {
      std::copy(code_override, code_override + R_, code.begin());
    } else {
      std::vector<float> feat(F_), logits(R_ * bits_);
      const float* ep = c.t >= 0 ? E_ + c.tok * d_ : start_;
      rmsnorm(ep, prev_norm_, router_eps_, feat.data(), d_);
      rmsnorm(e, cur_norm_, router_eps_, feat.data() + d_, d_);
      if (router_ssm_) std::copy(c.s.begin(), c.s.end(), feat.begin() + 2 * d_);
      matvec(router_, feat.data(), logits.data(), R_ * bits_, F_);
      for (int64_t r = 0; r < R_; ++r) {
        int v = 0;
        for (int64_t b = 0; b < bits_; ++b) v |= (logits[r * bits_ + b] > 0.f ? 1 : 0) << b;
        code[r] = static_cast<uint8_t>(v);
      }
    }
    std::copy(c.hist.begin() + R_, c.hist.end(), c.hist.begin());
    std::copy(code.begin(), code.end(), c.hist.end() - R_);
    c.t += 1;
    c.tok = tok;
  }

  // Global row (v2 layout) of route r, order index oi at the cursor, or -1.
  int64_t row(const Cursor& c, size_t oi, int64_t r) const {
    const int64_t n = ngrams_[oi];
    if (c.t < n - 1) return -1;
    int64_t addr = 0;
    for (int64_t k = 0; k < n; ++k) {
      const int64_t slot = maxn_ - n + k;  // oldest of the n codes first
      addr |= static_cast<int64_t>(c.hist[slot * R_ + r]) << (k * bits_);
    }
    return r * (int64_t(1) << (bits_ * n)) + addr;
  }

  void shortlist(const Cursor& c, std::vector<int64_t>& sl) {
    sl.clear();
    if (cap_ <= 0 || K_ <= 0) return;
    std::vector<int64_t> touched;
    for (size_t oi = 0; oi < ngrams_.size(); ++oi) {
      for (int64_t r = 0; r < R_; ++r) {
        const int64_t rw = row(c, oi, r);
        if (rw < 0) continue;
        const int32_t* ids = sk_ids_[oi] + rw * K_;
        const int32_t* cnt = sk_cnt_[oi] + rw * K_;
        for (int64_t k = 0; k < K_; ++k) {
          if (cnt[k] <= 0) continue;
          const int64_t id = ids[k];
          if (id < 0 || id >= V_) continue;
          if (acc_[id] == 0) touched.push_back(id);
          acc_[id] += cnt[k];
        }
      }
    }
    std::sort(touched.begin(), touched.end(), [this](int64_t a, int64_t b) {
      return acc_[a] != acc_[b] ? acc_[a] > acc_[b] : a < b;
    });
    for (int64_t id : touched) {
      if (static_cast<int64_t>(sl.size()) < cap_) {
        sl.push_back(id);
        mark_[id] = 1;
      }
      acc_[id] = 0;
    }
    for (int32_t f : fallback_) {
      if (static_cast<int64_t>(sl.size()) >= cap_) break;
      if (f < 0 || f >= V_ || mark_[f]) continue;
      sl.push_back(f);
      mark_[f] = 1;
    }
    for (int64_t id : sl) mark_[id] = 0;
  }

  // Draft hidden state at the cursor, then logits over the shortlist (or the full vocab).
  void readout(const Cursor& c, bool full, std::vector<int64_t>& sl, std::vector<float>& lg) {
    const float* e = E_ + c.tok * d_;
    // 1. rows -> memory norm -> KV, per route (parallel over routes)
    std::vector<int64_t> rows(ngrams_.size() * R_);
    for (size_t oi = 0; oi < ngrams_.size(); ++oi)
      for (int64_t r = 0; r < R_; ++r) rows[oi * R_ + r] = row(c, oi, r);
    pool_.run(static_cast<int>(R_), [&](int r) {
      float* tok = rt_.data() + r * Dt_;
      for (size_t oi = 0; oi < ngrams_.size(); ++oi) {
        float* dst = tok + oi * mem_dim_;
        const int64_t rw = rows[oi * R_ + r];
        if (rw < 0) {
          std::fill(dst, dst + mem_dim_, 0.f);
        } else if (table_half_[oi]) {
          const at::Half* src = static_cast<const at::Half*>(tables_[oi]) + rw * mem_dim_;
          for (int64_t j = 0; j < mem_dim_; ++j) dst[j] = static_cast<float>(src[j]);
        } else {
          const float* src = static_cast<const float*>(tables_[oi]) + rw * mem_dim_;
          std::copy(src, src + mem_dim_, dst);
        }
      }
      rmsnorm(tok, mem_norm_, mem_eps_, tok, Dt_);
      matvec(kv_proj_, tok, kvbuf_.data() + r * 2 * KH_, 2 * KH_, Dt_);
    });
    // 2. query
    std::vector<float> xn(d_), q(I_), tmp(I_);
    rmsnorm(e, q_norm_, draft_eps_, xn.data(), d_);
    matvec(w_q_, xn.data(), q.data(), I_, d_);
    if (w_sq_) {
      matvec(w_sq_, c.s.data(), tmp.data(), I_, N_);
      for (int64_t i = 0; i < I_; ++i) q[i] += tmp[i];
    }
    // 3. attention over routes + Sink
    std::vector<float> out(I_, 0.f), sc(R_);
    for (int64_t h = 0; h < kv_; ++h) {
      for (int64_t g = 0; g < G_; ++g) {
        const float* qv = q.data() + (h * G_ + g) * hd_;
        float mx = sink_;
        for (int64_t r = 0; r < R_; ++r) {
          sc[r] = dot(qv, kvbuf_.data() + r * 2 * KH_ + h * hd_, hd_) * scale_;
          mx = std::max(mx, sc[r]);
        }
        float den = 0.f;
        for (int64_t r = 0; r < R_; ++r) {
          sc[r] = std::exp(sc[r] - mx);
          den += sc[r];
        }
        den += std::exp(sink_ - mx);
        float* o = out.data() + (h * G_ + g) * hd_;
        for (int64_t r = 0; r < R_; ++r) {
          const float p = sc[r] / den;
          const float* v = kvbuf_.data() + r * 2 * KH_ + KH_ + h * hd_;
          for (int64_t j = 0; j < hd_; ++j) o[j] += p * v[j];
        }
      }
    }
    // 4. output projection + norm
    std::vector<float> hid(d_), hn(d_);
    matvec(w_o_, out.data(), hid.data(), d_, I_);
    rmsnorm(hid.data(), out_norm_, draft_eps_, hn.data(), d_);
    // 5. score
    if (!full) shortlist(c, sl);
    if (full || sl.empty()) {
      sl.resize(V_);
      for (int64_t v = 0; v < V_; ++v) sl[v] = v;
    }
    lg.resize(sl.size());
    for (size_t j = 0; j < sl.size(); ++j) lg[j] = dot(E_ + sl[j] * d_, hn.data(), d_);
  }

  static std::vector<int64_t> top_order(const std::vector<float>& lg, int64_t k) {
    std::vector<int64_t> idx(lg.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = static_cast<int64_t>(i);
    const int64_t m = std::min<int64_t>(k, static_cast<int64_t>(idx.size()));
    std::partial_sort(idx.begin(), idx.begin() + m, idx.end(), [&](int64_t a, int64_t b) {
      return lg[a] != lg[b] ? lg[a] > lg[b] : a < b;
    });
    idx.resize(m);
    return idx;
  }

  std::map<std::string, at::Tensor> keep_;  // owns the memory behind every pointer below
  std::vector<int64_t> ngrams_;
  Pool pool_;
  std::mutex mu_;
  int64_t d_, V_, R_, bits_, mem_dim_, kv_, hd_, G_, N_, K_, cap_, maxn_, Dt_, KH_, I_, F_;
  bool router_ssm_, query_ssm_, has_ssm_ = false;
  float ssm_eps_, router_eps_, draft_eps_, mem_eps_, scale_, sink_;
  const float *E_, *mem_norm_, *kv_proj_, *router_, *prev_norm_, *cur_norm_, *start_;
  const float *ssm_norm_ = nullptr, *ssm_w_in_ = nullptr, *ssm_a_ = nullptr;
  const float *q_norm_, *w_q_, *w_sq_, *w_o_, *out_norm_;
  std::vector<const void*> tables_;
  std::vector<bool> table_half_;
  std::vector<const int32_t*> sk_ids_, sk_cnt_;
  std::vector<int32_t> fallback_;
  std::vector<int64_t> tokens_;
  std::vector<uint8_t> codes_;
  std::vector<float> ssm_;
  std::vector<float> rt_, kvbuf_;
  std::vector<int64_t> acc_;
  std::vector<uint8_t> mark_;
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<Drafter>(m, "Drafter")
      .def(py::init<const std::map<std::string, at::Tensor>&, const std::map<std::string, double>&,
                    const std::vector<int64_t>&, int64_t>(),
           py::arg("tensors"), py::arg("numbers"), py::arg("ngrams"), py::arg("threads") = 1)
      .def("reset", &Drafter::reset)
      .def("length", &Drafter::length)
      .def("truncate", &Drafter::truncate)
      .def("committed_codes", &Drafter::committed_codes)
      .def("commit", &Drafter::commit, py::arg("tokens"), py::arg("codes") = py::none())
      .def("draft", &Drafter::draft, py::arg("gamma"), py::arg("hyp"), py::arg("topk") = 3,
           py::arg("last_codes") = false)
      .def("trace", &Drafter::trace, py::arg("tokens"), py::arg("codes") = py::none(), py::arg("full") = false);
}
