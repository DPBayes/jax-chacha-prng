# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2021 Aalto University

import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

# read version number
import importlib
spec = importlib.util.spec_from_file_location("version_module", "chacha/version.py")
version_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(version_module)

_jax_version_constraints = ' >= 0.2.10, <= 0.2.16'

setuptools.setup(
    name='jax-chacha-prng',
    version=version_module.VERSION,
    author="Lukas Prediger",
    author_email="lukas.m.prediger@aalto.fi",
    description="A pseudo-random number generator for JAX based on the 20 round ChaCha cipher.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DPBayes/jax-chacha-prng",
    packages=setuptools.find_packages(include=['chacha', 'chacha.*']),
    python_requires='>=3.6',
    install_requires=[
        "numpy >= 1.16, < 2",
        f"jax{_jax_version_constraints}"
    ],
    extras_require={
        "tests": [f"jax[minimum-jaxlib]{_jax_version_constraints}"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License"
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers"
    ],
)
