# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2022 Aalto University

import setuptools
from setuptools import Extension
from setuptools.command.build_ext import build_ext
import os
import subprocess
import re

class CMakeBuildExt(build_ext):
    # adapted from https://github.com/dfm/extending-jax
    def build_extensions(self):

        install_dir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath("dummy"))
        )
        os.makedirs(install_dir, exist_ok=True)

        osx_architectures = ";".join(re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", "")))

        cmake_args = [
            "-DCMAKE_INSTALL_PREFIX={}".format(install_dir),
            "-DCMAKE_BUILD_TYPE={}".format(
                "Debug" if self.debug else "Release"
            ),
            "-DCMAKE_OSX_ARCHITECTURES={}".format(osx_architectures)
        ]

        os.makedirs(self.build_temp, exist_ok=True)

        HERE = os.path.dirname(os.path.realpath(__file__))
        subprocess.check_call(
            ["cmake", HERE] + cmake_args, cwd=self.build_temp
        )

        # Build all the extensions
        super().build_extensions()

        # Finally run install
        subprocess.check_call(
            ["cmake", "--build", ".", "--target", "install"],
            cwd=self.build_temp,
        )

    def build_extension(self, ext):
        target_name = ext.name.split(".")[-1]
        subprocess.check_call(
            ["cmake", "--build", ".", "--target", target_name],
            cwd=self.build_temp,
        )


extensions = [
    Extension(
        "chacha.native",
        [
            "lib/arm/cpu_kernel_arch.hpp",
            "lib/generic/cpu_kernel_arch.hpp",
            "lib/intel/cpu_kernel_arch.hpp",
            "lib/chacha_kernels.hpp",
            "lib/cpu_kernel.cpp",
            "lib/cpu_kernel.hpp",
            "lib/defs.hpp",
            "lib/gpu_kernel.cpp.cu",
            "lib/gpu_kernel.hpp",
            "lib/python_bindings.cpp",
        ],
    ),
]

with open("README.md", "r") as f:
    long_description = f.read()

# read version number
import importlib
spec = importlib.util.spec_from_file_location("version_module", "chacha/version.py")
version_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(version_module)

_jax_version_lower_constraint = ' >= 0.2.12'
_jax_version_optimistic_upper_constraint = ', <= 2.0.0'
_jax_version_upper_constraint = ', <= 0.3.25'

_version = version_module.VERSION
if 'JAX_CHACHA_PRNG_BUILD' in os.environ:
    _version += f"+{os.environ['JAX_CHACHA_PRNG_BUILD']}"

setuptools.setup(
    name='jax-chacha-prng',
    version=_version,
    author="Lukas Prediger, Aalto University",
    author_email="lukas.m.prediger@aalto.fi",
    description="A pseudo-random number generator for JAX based on the 20 round ChaCha cipher.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DPBayes/jax-chacha-prng",
    packages=setuptools.find_packages(include=['chacha', 'chacha.*']),
    python_requires='>=3.6',
    install_requires=[
        "numpy >= 1.16, < 2",
        "deprecation < 3",
        f"jax{_jax_version_lower_constraint}{_jax_version_optimistic_upper_constraint}"
    ],
    extras_require={
        "tests": [
            f"jax[minimum-jaxlib]",
            "pytest"
        ],
        "compatible-jax": [f"jax{_jax_version_lower_constraint}{_jax_version_upper_constraint}"]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers"
    ],
    ext_modules=extensions,
    cmdclass={'build_ext': CMakeBuildExt}
)
