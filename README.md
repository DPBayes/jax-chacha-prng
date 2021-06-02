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
- `chacha.random.random_bits`: Equivalent to `jax.random._random_bits`: Raw access to random bits, returned as an array of unsinged integers.
- `chacha.random.uniform`: Equivalent to `jax.random.uniform`: Uniformly sampled floating point numbers in the range `[0, 1)`.

*Note*: `PRNGKey` instances of this ChaCha20-based RNG are not interoperable with those of `jax.random`, i.e., you cannot mix them.

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
- `chacha.cipher.decrypt_with_key`: Encrypt a of any length message by providing a 256 bit key, 96 bit nonce/IV and an optional 32 bit initial counter value.

State construction and use:

- `chacha.cipher.setup_state`: Create a ChaCha state structure by providing a 256 bit key, 96 bit nonce/IV and a 32 bit initial counter value.
- `chacha.cipher.encrypt`: Encrypt a message of any length using a ChaCha state structure.
- `chacha.cipher.decrypt`: Decrypt a message of any length using a ChaCha state structure.

as well as a number of functions to
## Installing

For the latest stable version, clone the repository, checkout the `stable`
branch and install via `pip`:
```
git checkout stable
pip install .
```

## Versioning

Version numbers adhere to [Semantic Versioning](https://semver.org/). Changes between releases are tracked in `ChangeLog.txt`.

## License

This software is currently only available for internal use at Aalto University.
