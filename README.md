# JAX-ChaCha-PRNG

A pseudo-random number generator for JAX based on the ChaCha20 cipher.

## API

The package currently exposes only basic RNG functions, namely

- `chacha.PRNGKey`: Equivalent to `jax.random.PRNGKey`: Given a seed of up to 256 bits, it returns a `PRNGKey` object from which randomness can be generated.
- `chacha.split`: Equivalent to `jax.random.split`: Splits a given `PRNGKey` into the desired number of fresh `PRNGKey` instances.
- `chacha.random_bits`: Equivalent to `jax.random._random_bits`: Raw access to random bits, returned as an array of unsinged integers.
- `chacha.uniform`: Equivalent to `jax.random.uniform`: Uniformly sampled floating point numbers in the range `[0, 1)`.

*Note*: `PRNGKey` instances of this ChaCha20-based RNG are not interoperable with those of `jax.random`, i.e., you cannot mix them.

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
