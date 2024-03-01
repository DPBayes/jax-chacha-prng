# JAX-ChaCha-PRNG

A cryptographically-secure pseudo-random number generator for JAX based on the 20 round ChaCha cipher.

The ChaCha cipher was introduced in [Daniel J. Bernstein "ChaCha, a variant of Salsa20"](https://cr.yp.to/chacha/chacha-20080128.pdf).

The implementation follows the specification in the IRTF [RFC 7539: "ChaCha20 and Poly1305 for IETF Protocols"](https://datatracker.ietf.org/doc/html/rfc7539).

Note that the implementation is not security-hardened. Our threat models assumes
that the machine on which the code is executed is a trusted environment and we
keep key values, cipher states and plaintexts in plain memory.

## API

The package is split into two modules:

- The `cipher` module is a full implementation of the ChaCha20 cipher.
- The `random` module provides a `JAX`-style API for the CSPRNG based on the cipher.

### Random
The package currently exposes basic RNG functions using the same interface as `JAX`:

- `chacha.random.PRNGKey`: Equivalent to `jax.random.PRNGKey`: Given a seed of up to 256 bits, it returns a `PRNGKey` object from which randomness can be generated.
- `chacha.random.split`: Equivalent to `jax.random.split`: Splits a given `PRNGKey` into the desired number of fresh `PRNGKey` instances.
- `chacha.random.fold_in`: !Deprecated! Equivalent to `jax.random.fold_in`: Deterministically derives a new `PRNGKey` from a given one and additional data.
- `chacha.random.random_bits`: Equivalent to `jax._src.random._random_bits`: Raw access to random bits, returned as an array of unsinged integers.
- `chacha.random.uniform`: Equivalent to `jax.random.uniform`: Uniformly sampled floating point numbers in the range `[0, 1)`.

*Note*: `PRNGKey` instances of this ChaCha20-based RNG are not interoperable with those of `jax.random`, i.e., you cannot mix them.

**Security notice** Versions prior to 2.0.0 may repeat random states via the `split` and `fold_in` functions.

#### Usage notes
Per conventions of pseudo-random number generation in the `JAX` framework, the functions `random_bits` and `uniform` are
deterministic given the randomness state (the `PRNGKey` object). The user needs to split the state using `split` before each
call to `random_bits` or `uniform` to get proper pseudo-random numbers. For more details, see what the [JAX documentation](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#random-numbers) has to say about this.

#### Implementation remarks
The `PRNGKey` object is the state of the ChaCha cipher with the randomness seed provided by the user as the cipher key.
Every invocation of `split` generates random numbers from the given state to use as fresh nonces in the derived states, with the key kept constant.
The counter in the state is used internally within calls to `random_bits` and `uniform` which can thus output up to 256 GiB of random values
for each state.

### Cipher
The following methods for direct use of the ChaCha20 cipher for encryption are available:

Quick use functions:

- `chacha.cipher.encrypt_with_key`: Encrypt a of any length message by providing a 256 bit key, 96 bit nonce/IV and an optional 32 bit initial counter value.
- `chacha.cipher.decrypt_with_key`: Decrypt a of any length message by providing a 256 bit key, 96 bit nonce/IV and an optional 32 bit initial counter value.

State construction and use:

- `chacha.cipher.setup_state`: Create a ChaCha state structure by providing a 256 bit key, 96 bit nonce/IV and a 32 bit initial counter value.
- `chacha.cipher.encrypt`: Encrypt a message of any length using a ChaCha state structure.
- `chacha.cipher.decrypt`: Decrypt a message of any length using a ChaCha state structure.

## Installing

For the latest stable version install via pip
```
pip install jax-chacha-prng
```

Binaries for glibc based 64-bit linux systems (manylinux wheels) are compiled with CPU and CUDA support (you will have to [install JAX with CUDA support](https://github.com/google/jax#pip-installation-gpu-cuda) to benefit from this).
Binaries for all other systems are compiled for CPU execution only. This is because JAX does not have CUDA libraries for these systems either.

However, you can instruct pip to instead compile the package from sources via
```
pip install --no-binary :all: jax-chacha-prng
```

or by installing it directly from the `v1-stable` branch:
```
pip install git+https://github.com/DPBayes/jax-chacha-prng@v1-stable#egg=jax-chacha-prng
```

This will compile CUDA kernels if the CUDA library is present on the system,
otherwise only CPU kernels will be built. To check whether CUDA kernels were
built and installed, you can check the return value of `chacha.native.cuda_supported()`.

### Note about JAX versions

JAX is still under ungoing development and its developers currently give no
guarantee that the API remains stable between releases. However, recent releases
were mostly stable in the interfaces required for JAX-ChaCha-PRNG. In order to allow
usage with JAX-ChaCha-PRNG with the most current JAX release, we therefore do not
currently constrain the JAX version from above in our dependency list.

However, if you should encounter issues with a new JAX release at some point,
you can use the `compatible-jax` installation target to force usage of the latest
JAX version known to be compatible with JAX-ChaCha-PRNG:
```
pip install .[compatible-jax]
```

JAX-ChaCha-PRNG is currently known to work reliably with JAX versions 0.2.12 - 0.4.25 .
We regularly check the compatible version range, but do not expect new versions of JAX to be immediately tested.

## Versioning

Version numbers adhere to [Semantic Versioning](https://semver.org/). Changes between releases are tracked in `ChangeLog.txt`.

## License

The software is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
You can find the full license text in `LICENSES/Apache-2.0.txt`.

Single files included from third parties may be under a different license, which is annotated in the file
itself and a full license text included in the `LICENSES` directory. The repository is fully [REUSE-compliant](https://reuse.software/).

## Acknowledgements

We thank the NVIDIA AI Technology Center Finland for their contribution of GPU performance benchmarking and helpful discussions on optimisation.

## Developing and Testing

We welcome any fixes, improvements or other contributions via pull request to this repository.

Before submitting your changes, please make sure to run our Python unit tests via `pytest tests/` and
ensure that they all succeed. If you add new functionality, please also add tests.

If you made changes to the native C++/CUDA code, please also compile and run the native tests:
```
mkdir build
cmake -DBUILD_TESTING=On ..
make -j
./cpu_kernel_tests
./gpu_kernel_tests # if you have CUDA installed and a GPU available
```
