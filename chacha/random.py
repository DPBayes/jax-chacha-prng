import numpy as np
import jax
import jax.numpy as jnp
from jax._src.random import _UINT_DTYPES, _check_shape
import typing
from functools import partial

import chacha.cipher as cc

################# chacha cipher based rng #########

## invariant for PRNG
## - 32 bit counter always starts from zero; used to increment for single calls to obtain arbitrary many random bits
##   from current rng_key
## - 96 bit IV used for PRNG splits -> splitting always resets counter
## - 256 bit random key unchanged; base randomness that is expanded by PRNG


def random_bits(rng_key, bit_width, shape):
    if bit_width not in _UINT_DTYPES:
        raise ValueError(f"requires bit field width in {_UINT_DTYPES.keys()}")
    size = np.prod(shape)
    num_bits = bit_width * size
    num_blocks = int(np.ceil(num_bits / cc.ChaChaStateBitSize))
    counters = jax.lax.iota(jnp.uint32, num_blocks)

    def generate_block(c):
        return cc._block(cc.increase_counter(rng_key, c)).flatten()

    blocks = jax.vmap(generate_block)(counters).flatten()
    assert blocks.shape == (num_blocks * cc.ChaChaStateElementCount,)

    dtype = _UINT_DTYPES[bit_width]

    out = cc.serialize(blocks, dtype)
    assert jnp.size(out) >= size

    return out[:size].reshape(shape)


@partial(jax.jit, static_argnums=(1,))
def _split(rng_key, num) -> jnp.ndarray:
    ivs = random_bits(rng_key, cc.ChaChaStateElementBitWidth, (num, 3))

    def make_rng_key(nonce):
        assert jnp.shape(nonce) == (3,)
        assert jnp.dtype(nonce) == cc.ChaChaStateElementType
        return cc.set_counter(cc.set_nonce(rng_key, nonce), 0)

    return jax.vmap(make_rng_key)(ivs)


@partial(jax.jit, static_argnums=(1, 2))
def _uniform(rng_key, shape, dtype, minval, maxval) -> jnp.ndarray:
    _check_shape("uniform", shape)
    if not jnp.issubdtype(dtype, np.floating):
        raise TypeError("uniform only accepts floating point dtypes.")

    minval = jax.lax.convert_element_type(minval, dtype)
    maxval = jax.lax.convert_element_type(maxval, dtype)
    minval = jax.lax.broadcast_to_rank(minval, len(shape))
    maxval = jax.lax.broadcast_to_rank(maxval, len(shape))

    finfo = jnp.finfo(dtype)
    nbits, nmant = finfo.bits, finfo.nmant

    if nbits not in (16, 32, 64):
        raise TypeError("uniform only accepts 32- or 64-bit dtypes.")

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


def PRNGKey(key: typing.Union[jnp.array, int, bytes]) -> jnp.array:
    if isinstance(key, int):
        key = key % (1 << cc.ChaChaKeySizeInBits)
        key = key.to_bytes(cc.ChaChaKeySizeInBytes, byteorder='big', signed=False)
    if isinstance(key, bytes):
        if len(key) > cc.ChaChaKeySizeInBytes:
            raise ValueError(f"A ChaCha PRNGKey cannot be larger than {cc.ChaChaKeySizeInBytes} bytes.")
        key = np.frombuffer(key, dtype=np.uint8)
        buf = np.zeros(cc.ChaChaKeySizeInBytes, dtype=np.uint8)
        buf[:len(key)] = key
        key = buf.tobytes()

        key = cc.from_buffer(key)

    key = jnp.array(key).flatten()[0:8]
    key = jnp.pad(key, (8 - jnp.size(key), 0), mode='constant')
    iv = jnp.zeros(3, dtype=jnp.uint32)
    return cc.setup_state(key, iv, jnp.zeros(1, dtype=jnp.uint32))


def split(rng_key: jnp.ndarray, num: int = 2) -> jnp.ndarray:
    """Splits a PRNG key into `num` new keys by adding a leading axis.

    Args:
        key: a PRNGKey (an array with shape (4,4) and dtype uint32).
        num: optional, a positive integer indicating the number of keys to produce
        (default 2).

    Returns:
        An array with shape (num, 4, 4) and dtype uint32 representing `num` new keys.
    """
    return _split(rng_key, int(num))


def uniform(key: jnp.ndarray,
            shape: typing.Sequence[int] = (),
            dtype: np.dtype = jnp.float64,
            minval: typing.Union[float, jnp.ndarray] = 0.,
            maxval: typing.Union[float, jnp.ndarray] = 1.) -> jnp.ndarray:
    """Sample uniform random values in [minval, maxval) with given shape/dtype.

    Args:
        key: a PRNGKey used as the random key.
        shape: optional, a tuple of nonnegative integers representing the result
        shape. Default ().
        dtype: optional, a float dtype for the returned values (default float64 if
        jax_enable_x64 is true, otherwise float32).
        minval: optional, a minimum (inclusive) value broadcast-compatible with shape for the range (default 0).
        maxval: optional, a maximum (exclusive) value broadcast-compatible with shape for the range (default 1).

    Returns:
        A random array with the specified shape and dtype.
    """
    if not jax.dtypes.issubdtype(dtype, np.floating):
        raise ValueError(f"dtype argument to `uniform` must be a float dtype, "
                         f"got {dtype}")
    dtype = jax.dtypes.canonicalize_dtype(dtype)
    shape = jax.abstract_arrays.canonicalize_shape(shape)
    return _uniform(key, shape, dtype, minval, maxval)
