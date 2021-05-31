import jax.numpy as jnp
import numpy as np
import jax
import typing
from functools import partial

# jax implementation of https://cr.yp.to/chacha/chacha-20080128.pdf
# following: https://tools.ietf.org/html/rfc7539


def rotate_left(x: jnp.array, num_bits: int) -> jnp.array:
    dtype = jax.lax.dtype(x)
    issubclass(dtype.type, jnp.unsignedinteger)
    type_info = jnp.iinfo(dtype)
    return (x << num_bits) ^ (x >> (type_info.bits - num_bits))

@jax.jit
def quarterround(x):
    assert x.dtype == jnp.uint32
    assert x.size == 4
    a, b, c, d = x[0], x[1], x[2], x[3]
    a += b
    d ^= a
    d = rotate_left(d, 16)
    c += d
    b ^= c
    b = rotate_left(b, 12)
    a += b
    d ^= a
    d = rotate_left(d, 8)
    c += d
    b ^= c
    b = rotate_left(b, 7)
    return jnp.array([a,b,c,d], dtype=jnp.uint32)

@jax.jit
def chacha_double_round(ctx):
    assert ctx.shape == (4, 4)
    assert ctx.dtype == jnp.uint32

    qr_vmap = jax.vmap(quarterround, in_axes=1, out_axes=1)

    # quarterround for each column
    ctx = qr_vmap(ctx)

    # moving diagonals into columns
    diag_map = jnp.array([
         0,  1,  2,  3,
         5,  6,  7,  4,
        10, 11,  8,  9,
        15, 12, 13, 14
    ])
    diag_ctx = ctx.ravel()[diag_map].reshape(4, 4)

    # quarterround for each column (aka, diagonal)
    diag_ctx = qr_vmap(diag_ctx)

    # undo previous mapping
    back_map = jnp.array([
         0,  1,  2,  3,
         7,  4,  5,  6,
        10, 11,  8,  9,
        13, 14, 15, 12
    ])
    ctx = diag_ctx.ravel()[back_map].reshape(4, 4)
    return ctx

@jax.jit
def chacha_block(ctx):
    assert ctx.shape == (4, 4)
    assert ctx.dtype == jnp.uint32

    ori_ctx = ctx
    ctx = jax.lax.fori_loop(0, 10, lambda i, ctx: chacha_double_round(ctx), ctx)
    ctx += ori_ctx
    return ctx


KEY_GEN_CONSTANTS = {
    32: np.frombuffer("expand 32-byte k".encode("ascii"), dtype=np.uint32),
    16: np.frombuffer("expand 16-byte k".encode("ascii"), dtype=np.uint32)
}

def chacha_setup(key, iv, counter):
    if isinstance(key, bytes):
        if len(key) not in [16, 32]:
            raise ValueError("key must consist of 16 or 32 bytes")
        key = from_buffer(key)
    elif jax.lax.dtype(key) != jnp.uint32 or jnp.size(key) not in [4, 8]:
        raise ValueError("key must be a buffer or an array of 32-bit unsinged integers, totalling to 16 or 32 bytes!")


    if isinstance(iv, bytes):
        iv = from_buffer(iv)
    if jax.lax.dtype(iv) != jnp.uint32 or jnp.size(iv) != 3:
        raise ValueError("iv must be three 32-bit unsigned integers or a buffer of 12 bytes!")

    if jax.lax.dtype(counter) != jnp.uint32 or jnp.size(counter) != 1:
        raise ValueError("counter must be a single 32-bit unsigned integer!")

    key_bits = key.size * 4
    if key_bits == 16:
        key = jnp.tile(key, 2)
    key = key.reshape(2, 4)
    inputs = jnp.hstack((counter, iv))
    ctx = jnp.vstack((KEY_GEN_CONSTANTS[key_bits], key, inputs))
    return ctx

def from_buffer(buffer):
    return jnp.array(np.frombuffer(buffer, dtype=np.uint32))

def chacha_increase_counter(ctx, amount):
    return jax.ops.index_add(ctx, (3,0), amount, True, True)

def chacha_increment_counter(ctx):
    return chacha_increase_counter(ctx, 1)

def chacha_set_counter(ctx, counter):
    return jax.ops.index_update(ctx, (3,0), counter, True, True)

def chacha_get_counter(ctx):
    return ctx[3,0]

def chacha_set_nonce(ctx, nonce):
    assert jnp.shape(nonce) == (3,)
    assert jnp.dtype(ctx) == jnp.dtype(nonce)
    return jax.ops.index_update(ctx, jax.ops.index[3, 1:4], nonce, True, True)

def chacha_get_nonce(ctx):
    return ctx[3, 1:3]

def chacha_serialize(a, out_dtype=jnp.uint8):
    assert jax.lax.dtype(a) == jnp.uint32
    type_info = jnp.iinfo(out_dtype)

    a = a.flatten()
    bit_width = type_info.bits
    if bit_width == 64:
        a = [jax.lax.convert_element_type(x, out_dtype) for x in jnp.split(a, 2)]
        a = jax.lax.shift_left(a[0], out_dtype(32)) | a[1]
    elif bit_width in [8, 16]:
        a = jax.lax.shift_right_logical(
            jax.lax.broadcast(a, (1,)),
            jax.lax.mul(
                jnp.uint32(bit_width),
                jax.lax.broadcasted_iota(jnp.uint32, (32 // bit_width, 1), 0)
            )
        )
        a = jax.lax.reshape(a, (jnp.size(a),), (1, 0))
        a = jax.lax.convert_element_type(a, out_dtype)

    return a

def chacha_encrypt(key, iv, counter, message):
    i = 0
    ctx = chacha_setup(key, iv, counter)
    ciphertext = b''
    while i*64 < len(message):
        message_block = np.frombuffer(message[i*64 : (i+1)*64], dtype=np.uint8)

        key_stream = chacha_serialize(chacha_block(ctx))[:len(message_block)]
        cipher_block = message_block ^ key_stream
        ciphertext += cipher_block.tobytes()

        ctx = chacha_increment_counter(ctx)
        i += 1

    return ciphertext

################# chacha cipher based rng #########

## invariant for PRNG
## - 32 bit counter always starts from zero; used to increment for single calls to obtain arbitrary many random bits from current ctx
## - 96 bit IV used for PRNG splits -> splitting always resets counter
## - 256 bit random key unchanged; base randomness that is expanded by PRNG


def random_bits(ctx, bit_width, shape):
    if bit_width not in jax.random._UINT_DTYPES:
        raise ValueError(f"requires bit field width in {jax.random._UINT_DTYPES.keys()}")
    size = np.prod(shape)
    num_bits = bit_width * size
    block_size_in_bits = 4*4*32
    num_blocks = int(np.ceil(num_bits / block_size_in_bits))
    counters = jax.lax.iota(jnp.uint32, num_blocks)

    def generate_block(c):
        return chacha_block(chacha_increase_counter(ctx, c)).flatten()

    blocks = jax.vmap(generate_block)(counters).flatten()
    assert blocks.shape == (num_blocks * 16,)

    dtype = jax.random._UINT_DTYPES[bit_width]

    out = chacha_serialize(blocks, dtype)
    assert jnp.size(out) >= size

    return out[:size].reshape(shape)

@partial(jax.jit, static_argnums=(1,))
def _split(ctx, num) -> jnp.ndarray:
    ivs = random_bits(ctx, 32, (num, 3))

    def make_ctx(nonce):
        assert jnp.shape(nonce) == (3,)
        assert jnp.dtype(nonce) == jnp.uint32
        return chacha_set_counter(chacha_set_nonce(ctx, nonce), 0)

    return jax.vmap(make_ctx)(ivs)


@partial(jax.jit, static_argnums=(1, 2))
def _uniform(ctx, shape, dtype, minval, maxval) -> jnp.ndarray:
  jax.random._check_shape("uniform", shape)
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

  bits = random_bits(ctx, nbits, shape)

  # The strategy here is to randomize only the mantissa bits with an exponent of
  # 1 (after applying the bias), then shift and scale to the desired range. The
  # bit-level transformation we use relies on Numpy and XLA having bit-for-bit
  # equivalent float representations, which might not be true on all platforms.
  float_bits = jax.lax.bitwise_or(
      jax.lax.shift_right_logical(bits, np.array(nbits - nmant, jax.lax.dtype(bits))),
      np.array(1., dtype).view(jax.random._UINT_DTYPES[nbits]))
  floats = jax.lax.bitcast_convert_type(float_bits, dtype) - np.array(1., dtype)
  return jax.lax.max(
      minval,
      jax.lax.reshape(floats * (maxval - minval) + minval, shape))


def PRNGKey(key: typing.Union[jnp.array, int, bytes]) -> jnp.array:
    if isinstance(key, int):
        key = key % (1 << 256)
        key = key.to_bytes(32, byteorder='big', signed=False)
    if isinstance(key, bytes):
        key = from_buffer(key)

    key = jnp.array(key).flatten()[0:8]
    key = jnp.pad(key, (8-jnp.size(key),0), mode='constant')
    iv = jnp.zeros(3, dtype=jnp.uint32)
    return chacha_setup(key, iv, jnp.zeros(1, dtype=jnp.uint32))

def split(ctx: jnp.ndarray, num: int = 2) -> jnp.ndarray:
    """Splits a PRNG key into `num` new keys by adding a leading axis.

    Args:
        key: a PRNGKey (an array with shape (4,4) and dtype uint32).
        num: optional, a positive integer indicating the number of keys to produce
        (default 2).

    Returns:
        An array with shape (num, 4, 4) and dtype uint32 representing `num` new keys.
    """
    return _split(ctx, int(num))

def uniform(key: jnp.ndarray,
            shape: typing.Sequence[int] = (),
            dtype: np.dtype = jax.dtypes.float_,
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
  return _uniform(key, shape, dtype, minval, maxval)  # type: ignore
