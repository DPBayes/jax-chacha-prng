import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

# read version number
import importlib
spec = importlib.util.spec_from_file_location("version_module", "chacha/version.py")
version_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(version_module)

setuptools.setup(
    name='jax-chacha-prng',
    version = version_module.VERSION,
    author="Lukas Prediger",
    author_email="lukas.m.prediger@aalto.fi",
    description="A pseudo-random number generator for JAX based on the ChaCha20 cipher.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://version.aalto.fi/gitlab/predigl2/jax-chacha20-prng",
    packages=setuptools.find_packages(include=['chacha', 'chacha.*']),
    python_requires='>=3.7, <=3.9',
    install_requires=[
        'jax == 0.2.3',
        'jaxlib == 0.1.56'
    ], # todo: find stable versions
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research"
     ],
)
