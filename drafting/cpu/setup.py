"""Build the C++ CPU drafter (M5.T2) as an importable extension ``morbius_cpu_drafter``.

    cd drafting/cpu && python setup.py build_ext --inplace     # or: pip install -e drafting/cpu
    MORBIUS_DRAFTER_NATIVE=0 python setup.py build_ext ...      # portable build (no -march=native)

Without a build, ``drafting.cpu_drafter.load_extension()`` JIT-compiles the same source with
``torch.utils.cpp_extension.load`` (needs a C++17 compiler and ninja).
"""

from __future__ import annotations

import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

if os.name == "nt":  # the C++ standard flag is left to cpp_extension (it matches torch's headers)
    flags = ["/O2"]
else:
    flags = ["-O3"]
    if os.environ.get("MORBIUS_DRAFTER_NATIVE", "1") == "1":
        flags.append("-march=native")

setup(
    name="morbius_cpu_drafter",
    ext_modules=[CppExtension("morbius_cpu_drafter", ["drafter.cpp"], extra_compile_args=flags)],  # run from drafting/cpu
    cmdclass={"build_ext": BuildExtension},
)
