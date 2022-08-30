# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2021,2022 Aalto University

""" A cryptographically secure pseudo-random number generator for JAX.

This CSPRNG is based on the 20-round ChaCha cipher and offers the same API
as the default JAX PRNG.

Its randomness state (the PRNGKey, in JAX lingo) is simply the ChaCha cipher state.

The following invariants hold:
- The 256 bit key provides base randomness that is expanded by PRNG; it given by the user as seed to function `PRNGKey`
- The 32 bit counter in a randomness state is always incremented by randomness expanders such as `random_bits`
    to provide streams of randomness.
- The 96 bit IV is used for randomness state splits using the `split` function; splitting results in a new state
    that maintains the same cipher key, a counter value of zero and a fresh IV derived from the previous IV in such
    a way that the new IV is unique among all states derived from the same initial key state. An IV value of zero
    indicates an invalidated state, which may occur when a maximum number of nested splits is reached.
"""

import numpy as np  # type: ignore
import jax
import jax.numpy as jnp
from jax._src.random import _check_shape
import typing
from functools import partial
from enum import IntFlag

import chacha.cipher as cc
from chacha import defs

# importing canonicalize_shape function
try:
    # pre jax v0.2.14 location
    _canonicalize_shape = jax.abstract_arrays.canonicalize_shape  # type: ignore
except (AttributeError, ImportError):  # pragma: no cover
    # post jax v0.2.14 location
    try:
        _canonicalize_shape = jax.core.canonicalize_shape  # type: ignore
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
        _UINT_DTYPES = jax._src.random.UINT_DTYPES  # type: ignore
    except (AttributeError, ImportError):  # pragma: no cover
        raise ImportError("Cannot import UINT_DTYPES enum. "
                          "You are probably using an incompatible version of jax.")


RNGState = cc.ChaChaState


class ErrorFlag(IntFlag):
    """Possible error states returned from `random_bits`."""
    CounterOverflow = 1,
    InvalidState = 2,


@jax.jit
def is_state_invalidated(rng_key: RNGState) -> bool:
    return jnp.all(cc.get_nonce(rng_key) == 0)


@partial(jax.jit, static_argnums=(1, 2))
def random_bits(rng_key: RNGState, bit_width: int, shape: typing.Sequence[int])\
        -> typing.Tuple[jnp.ndarray, RNGState, jnp.uint32]:
    """ Generates an array containing random integers.

    Note that this function enters a failure state if the `rng_key` is invalidated
    or the randomness counter overflows. In those cases, a flag indicating the
    error is set in the second output, while the first output is zeroed out instead
    of containing an array of random values.

    Args:
      rng_key: The not-invalidated RNGState object from which to generate random bits.
      bit_width: The number of bits in each element of the output.
      shape: The shape of the output array.

    Returns:
      A tuple containing
        - array of the given shape containing uniformly random unsigned integers with the given bit width,
        - the next RNGState after generating the requested amount of random bits (with the counter value advanced),
        - an integer representing an array of error flags (see `ErrorFlag`)
    """
    if bit_width not in _UINT_DTYPES:
        raise ValueError(f"requires bit field width in {_UINT_DTYPES.keys()}")
    size = int(np.prod(shape, dtype=int))
    num_bits = bit_width * size
    num_blocks = int(np.ceil(num_bits / cc.ChaChaStateBitSize))
    counters = jax.lax.iota(jnp.uint32, num_blocks)

    def generate_block(c: RNGState) -> jnp.ndarray:
        return cc._block(cc.increase_counter(rng_key, c))

    blocks = jnp.ravel(jax.vmap(generate_block)(counters))
    assert blocks.shape == (num_blocks * defs.ChaChaStateElementCount,)

    dtype = _UINT_DTYPES[bit_width]

    out = cc.serialize(blocks, dtype)
    assert jnp.size(out) >= size

    next_rng_key = cc.increase_counter(rng_key, num_blocks)

    counter_exceeded = cc.get_counter(rng_key) >= cc.get_counter(next_rng_key)  # detect wrap-around of counter
    error_flags = jnp.uint32((is_state_invalidated(rng_key) << 1) ^ counter_exceeded)

    out = out[:size].reshape(shape)
    out = jnp.where(error_flags == 0, out, out ^ out)

    next_rng_key = jnp.where(error_flags == 0, next_rng_key, jnp.zeros_like(rng_key))

    return out, next_rng_key, error_flags


@partial(jax.jit, static_argnums=(1,))
def _split(rng_key: RNGState, num: int) -> RNGState:
    bitlength_num = num.bit_length()
    if bitlength_num > 32:
        raise ValueError("Splits into more than 2^32 new keys are currently not supported.")

    old_nonce = cc.get_nonce(rng_key)

    old_nonce = jnp.concatenate((old_nonce, jnp.zeros((1,), rng_key.dtype)))
    assert old_nonce.shape == (defs.ChaChaNonceSizeInWords + 1,)
    new_nonce_base = ((old_nonce[:defs.ChaChaNonceSizeInWords] << bitlength_num)
                      ^ (old_nonce[1:] >> (32 - bitlength_num)))
    assert new_nonce_base.shape == (defs.ChaChaNonceSizeInWords,)

    split_nesting_exceeded = (old_nonce[0] >= (1 << (32 - bitlength_num))) | jnp.all(old_nonce == 0)
    state_previuosly_used = cc.get_counter(rng_key) > 0

    def make_rng_key(i: int) -> RNGState:
        nonce = jnp.concatenate((new_nonce_base[:defs.ChaChaNonceSizeInWords - 1], new_nonce_base[-1:] ^ i))
        nonce *= (1 - split_nesting_exceeded)  # set the nonce to 0 (invalid state) if split nesting limit is exceeded
        nonce *= (1 - state_previuosly_used) # set the nonced to 0 (invalid state) if the state was already used to generate randomness
        return cc.set_counter(cc.set_nonce(rng_key, nonce), 0)

    return jax.vmap(make_rng_key)(jnp.arange(num, dtype=rng_key.dtype))


@partial(jax.jit, static_argnums=(1, 2, 5))
def _uniform(
        rng_key: RNGState,
        shape: typing.Tuple[int],
        dtype: type,
        minval: jnp.float_,
        maxval: jnp.float_,
        return_next_key: bool = False
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

    bits, next_rng_key, errors = random_bits(rng_key, nbits, shape)

    # The strategy here is to randomize only the mantissa bits with an exponent of
    # 1 (after applying the bias), then shift and scale to the desired range. The
    # bit-level transformation we use relies on Numpy and XLA having bit-for-bit
    # equivalent float representations, which might not be true on all platforms.
    float_bits = jax.lax.bitwise_or(
        jax.lax.shift_right_logical(bits, np.array(nbits - nmant, jax.lax.dtype(bits))),
        np.array(1., dtype).view(_UINT_DTYPES[nbits])
    )
    floats = jax.lax.bitcast_convert_type(float_bits, dtype) - np.array(1., dtype)
    result = jax.lax.max(
        minval,
        jax.lax.reshape(floats * (maxval - minval) + minval, shape)
    )

    result = jnp.where(errors == 0, result, result * jnp.nan)
    if return_next_key:
        return result, next_rng_key
    return result


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
        seed = seed % (1 << defs.ChaChaKeySizeInBits)
        seed = seed.to_bytes(defs.ChaChaKeySizeInBytes, byteorder='big', signed=False)
    if isinstance(seed, bytes):
        if len(seed) > defs.ChaChaKeySizeInBytes:
            raise ValueError(f"A ChaCha PRNGKey cannot be larger than {defs.ChaChaKeySizeInBytes} bytes.")
        seed_buffer = np.frombuffer(seed, dtype=np.uint8)
        key_buffer = np.zeros(defs.ChaChaKeySizeInBytes, dtype=np.uint8)
        key_buffer[:len(seed_buffer)] = seed_buffer
        key_buffer = key_buffer.view(defs.ChaChaStateElementType)
    else:
        raise TypeError(f"seed must be either an array, an integer or a bytes objects; got {type(seed)}.")

    key = jnp.array(key_buffer).flatten()[0:defs.ChaChaKeySizeInWords]
    key = jnp.pad(key, (defs.ChaChaKeySizeInWords - jnp.size(key), 0), mode='constant')
    iv = jnp.zeros(defs.ChaChaNonceSizeInWords, dtype=defs.ChaChaStateElementType)
    iv = iv.at[-1].set(1)
    counter = jnp.zeros(defs.ChaChaCounterSizeInWords, dtype=defs.ChaChaStateElementType)
    return cc.setup_state(key, iv, counter)


def split(rng_key: RNGState, num: int = 2) -> typing.Sequence[RNGState]:
    """Splits a PRNG key into `num` new keys by adding a leading axis.

    Splits can be nested up to 96 times, after which it can no longer be guaranteed
    that PRNG states are not repeated. If this limit is exceeded, the returned
    split states are invalidated by setting their nonces to 0. The same is true
    if the input `rng_key` is already an invalidated state.

    Calling code should verify that this is not the case where neccessary.

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
        maxval: typing.Union[float, jnp.ndarray] = 1.,
        return_next_key: bool = False
    ) -> jnp.ndarray:  # noqa:E121,E125
    """Samples uniform random values in [minval, maxval) with given shape/dtype.

    If `key` was invalidated or producing random data results in a randomness counter overflow,
    the output will be an array of the requested size, containing only NaN values.

    Args:
      key: The RNGState.
      shape: An optional tuple of nonnegative integers representing the result shape.
      dtype: An optional float dtype for the returned values (default float64).
      minval: An optional minimum (inclusive) value broadcast-compatible with shape for the range (default 0).
      maxval: An optional maximum (exclusive) value broadcast-compatible with shape for the range (default 1).
      return_next_key: An optional boolean flag. If `True`, the function returns a new RNGState (with advanced counter).

    Returns:
      A random array with the specified shape and dtype if `return_next_key` is `False`.
      A tuple consisting of the random array and a new RNGState if `return_next_key` is `True`.

    """
    if not jax.dtypes.issubdtype(dtype, np.floating):
        raise TypeError(f"dtype argument to `uniform` must be a float dtype, got {dtype}")
    dtype = jax.dtypes.canonicalize_dtype(dtype)
    shape = _canonicalize_shape(shape)
    return _uniform(key, shape, dtype, minval, maxval, return_next_key)
