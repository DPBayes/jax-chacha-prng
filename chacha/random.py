# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2021 Aalto University

""" A cryptographically secure pseudo-random number generator for JAX.

This CSPRNG is based on the 20-round ChaCha cipher and offers the same API
as the default JAX PRNG.

Its randomness state (the PRNGKey, in JAX lingo) is simply the ChaCha cipher state.

The following invariants hold:
- The 256 bit key provides base randomness that is expanded by PRNG; it given by the user as seed to function `PRNGKey`
- The 32 bit counter in a randomness state is always set to zero; randomness expander such as `random_bits` increment
    it internally to provide streams of randomness.
- The 96 bit IV is used for randomness state splits using the `split` function; splitting results in a new state
    that maintains the same cipher key, a counter value of zero and a randomly sampled IV expanded from the key that
    was split.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax._src.random import _check_shape
import typing
from functools import partial

import chacha.cipher as cc

# importing canonicalize_shape function
try:
    # pre jax v0.2.14 location
    _canonicalize_shape = jax.abstract_arrays.canonicalize_shape  # type: ignore
except (AttributeError, ImportError):  # pragma: no cover
    # post jax v0.2.14 location
    try:
        _canonicalize_shape = jax.core.canonicalize_shape
    except (AttributeError, ImportError):  # pragma: no cover
        raise ImportError("Cannot import canonicalize_shape routine. "
                          "You are probably using an incompatible version of jax.")

# importing _UINT_DTYPES
try:
    # pre jax v0.2.20 location
    _UINT_DTYPES = jax._src.random._UINT_DTYPES  # type: ignore
except (AttributeError, ImportError):  # pragma: no cover
    # post jax v.2.20 location
    try:
        _UINT_DTYPES = jax._src.random.UINT_DTYPES
    except (AttributeError, ImportError):  # pragma: no cover
        raise ImportError("Cannot import UINT_DTYPES enum. "
                          "You are probably using an incompatible version of jax.")


RNGState = cc.ChaChaState


@partial(jax.jit, static_argnums=(1, 2))
def random_bits(rng_key: RNGState, bit_width: int, shape: typing.Sequence[int]) -> jnp.ndarray:
    """ Generate an array containing random integers.

    Args:
      rng_key: The PRNGKey object from which to generate random bits.
      bit_width: The number of bits in each element of the output.
      shape: The shape of the output array.

    Returns:
      An array of the given shape containing uniformly random unsigned integers with the given bit width.
    """
    if bit_width not in _UINT_DTYPES:
        raise ValueError(f"requires bit field width in {_UINT_DTYPES.keys()}")
    size = np.prod(shape, dtype=int)
    num_bits = bit_width * size
    num_blocks = int(np.ceil(num_bits / cc.ChaChaStateBitSize))
    counters = jax.lax.iota(jnp.uint32, num_blocks)

    def generate_block(c: RNGState) -> jnp.ndarray:
        return jnp.ravel(cc._block(cc.increase_counter(rng_key, c)))

    blocks = jnp.ravel(jax.vmap(generate_block)(counters))
    assert blocks.shape == (num_blocks * cc.ChaChaStateElementCount,)

    dtype = _UINT_DTYPES[bit_width]

    out = cc.serialize(blocks, dtype)
    assert jnp.size(out) >= size

    return out[:size].reshape(shape)


@partial(jax.jit, static_argnums=(1,))
def _split(rng_key: RNGState, num: int) -> RNGState:
    ivs = random_bits(rng_key, cc.ChaChaStateElementBitWidth, (num, cc.ChaChaNonceSizeInWords))

    def make_rng_key(nonce: jnp.ndarray) -> RNGState:
        assert jnp.shape(nonce) == (cc.ChaChaNonceSizeInWords,)
        assert jnp.dtype(nonce) == cc.ChaChaStateElementType
        return cc.set_counter(cc.set_nonce(rng_key, nonce), 0)

    return jax.vmap(make_rng_key)(ivs)


@jax.jit
def fold_in(rng_key: RNGState, data: int) -> RNGState:
    iv = cc.get_nonce(cc._block(cc.set_counter(rng_key, data)))
    return cc.set_counter(cc.set_nonce(rng_key, iv), 0)


@partial(jax.jit, static_argnums=(1, 2))
def _uniform(
        rng_key: RNGState,
        shape: typing.Tuple[int],
        dtype: type,
        minval: jnp.float_,
        maxval: jnp.float_
    ) -> jnp.ndarray:  # noqa:E121,E125
    _check_shape("uniform", shape)
    if not jnp.issubdtype(dtype, np.floating):
        print("encountered exc in _uniform")
        raise TypeError("uniform only accepts floating point dtypes.")

    minval = jax.lax.convert_element_type(minval, dtype)
    maxval = jax.lax.convert_element_type(maxval, dtype)
    minval = jax.lax.broadcast_to_rank(minval, len(shape))
    maxval = jax.lax.broadcast_to_rank(maxval, len(shape))

    finfo = jnp.finfo(dtype)
    nbits, nmant = finfo.bits, finfo.nmant

    assert nbits in (16, 32, 64)

    bits = random_bits(rng_key, nbits, shape)

    # The strategy here is to randomize only the mantissa bits with an exponent of
    # 1 (after applying the bias), then shift and scale to the desired range. The
    # bit-level transformation we use relies on Numpy and XLA having bit-for-bit
    # equivalent float representations, which might not be true on all platforms.
    float_bits = jax.lax.bitwise_or(
        jax.lax.shift_right_logical(bits, np.array(nbits - nmant, jax.lax.dtype(bits))),
        np.array(1., dtype).view(_UINT_DTYPES[nbits])
    )
    floats = jax.lax.bitcast_convert_type(float_bits, dtype) - np.array(1., dtype)
    return jax.lax.max(
        minval,
        jax.lax.reshape(floats * (maxval - minval) + minval, shape)
    )


def PRNGKey(seed: typing.Union[jnp.ndarray, int, bytes]) -> RNGState:
    """ Creates a cryptographically secure pseudo-random number generator (PRNG) key given a seed.

    The seed is used as a cryptographic key to expand into randomness. Its length in bits
    determines the cryptographic strength of the generated random numbers. It can be up to 256 bit long.

    Args:
      seed: The seed, either as an integer or a bytes object.

    Returns:
      An PRNG key, which is equivalent to a ChaChaState, which in turn is a 4x4 array of 32 bit integers.
    """
    if isinstance(seed, int):
        seed = seed % (1 << cc.ChaChaKeySizeInBits)
        seed = seed.to_bytes(cc.ChaChaKeySizeInBytes, byteorder='big', signed=False)
    if isinstance(seed, bytes):
        if len(seed) > cc.ChaChaKeySizeInBytes:
            raise ValueError(f"A ChaCha PRNGKey cannot be larger than {cc.ChaChaKeySizeInBytes} bytes.")
        seed_buffer = np.frombuffer(seed, dtype=np.uint8)
        key_buffer = np.zeros(cc.ChaChaKeySizeInBytes, dtype=np.uint8)
        key_buffer[:len(seed_buffer)] = seed_buffer
        key_buffer = key_buffer.view(jnp.uint32)
    else:
        raise TypeError(f"seed must be either an array, an integer or a bytes objects; got {type(seed)}.")

    key = jnp.array(key_buffer).flatten()[0:8]
    key = jnp.pad(key, (8 - jnp.size(key), 0), mode='constant')
    iv = jnp.zeros(3, dtype=jnp.uint32)
    return cc.setup_state(key, iv, jnp.zeros(1, dtype=jnp.uint32))


def split(rng_key: RNGState, num: int = 2) -> typing.Sequence[RNGState]:
    """Splits a PRNG key into `num` new keys by adding a leading axis.

    Args:
      key: A PRNGKey (an array with shape (4,4) and dtype uint32).
      num: An optional positive integer indicating the number of keys to produce.

    Returns:
      An array with shape (num, 4, 4) and dtype uint32 representing `num` new PRNG keys.
    """
    return _split(rng_key, int(num))


def uniform(
        key: RNGState,
        shape: typing.Sequence[int] = (),
        dtype: np.dtype = jnp.float64,
        minval: typing.Union[float, jnp.ndarray] = 0.,
        maxval: typing.Union[float, jnp.ndarray] = 1.
    ) -> jnp.ndarray:  # noqa:E121,E125
    """Samples uniform random values in [minval, maxval) with given shape/dtype.

    Args:
      key: The PRNGKey.
      shape: An optional tuple of nonnegative integers representing the result shape.
      dtype: An optional float dtype for the returned values (default float64).
      minval: An ptional minimum (inclusive) value broadcast-compatible with shape for the range (default 0).
      maxval: An optional maximum (exclusive) value broadcast-compatible with shape for the range (default 1).

    Returns:
      A random array with the specified shape and dtype.

    """
    if not jax.dtypes.issubdtype(dtype, np.floating):
        raise TypeError(f"dtype argument to `uniform` must be a float dtype, got {dtype}")
    dtype = jax.dtypes.canonicalize_dtype(dtype)
    shape = _canonicalize_shape(shape)
    return _uniform(key, shape, dtype, minval, maxval)
