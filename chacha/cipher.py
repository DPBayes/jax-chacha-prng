import jax.numpy as jnp
import numpy as np
import jax
from typing import Union, Type, Optional, Tuple

# jax implementation of https://cr.yp.to/chacha/chacha-20080128.pdf
# following: https://tools.ietf.org/html/rfc7539

ChaChaState = jnp.array
ChaChaStateShape = (4, 4)
ChaChaStateElementCount = np.prod(ChaChaStateShape)
ChaChaStateElementType = jnp.uint32
ChaChaStateElementBitWidth = jnp.iinfo(ChaChaStateElementType).bits
ChaChaStateBitSize = ChaChaStateElementCount * ChaChaStateElementBitWidth


#### CORE CHACHA ROUND FUNCTIONS ####

def rotate_left(x: jnp.array, num_bits: int) -> jnp.array:
    dtype = jax.lax.dtype(x)
    issubclass(dtype.type, jnp.unsignedinteger)
    type_info = jnp.iinfo(dtype)
    return (x << num_bits) ^ (x >> (type_info.bits - num_bits))

@jax.jit
def _quarterround(x: jnp.array) -> jnp.array:
    assert x.dtype == ChaChaStateElementType
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
def _double_round(ctx: ChaChaState) -> ChaChaState:
    assert ctx.shape == ChaChaStateShape
    assert ctx.dtype == ChaChaStateElementType

    qr_vmap = jax.vmap(_quarterround, in_axes=1, out_axes=1)

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
def _block(ctx: ChaChaState) -> ChaChaState:
    assert ctx.shape == ChaChaStateShape
    assert ctx.dtype == ChaChaStateElementType

    ori_ctx = ctx
    ctx = jax.lax.fori_loop(0, 10, lambda i, ctx: _double_round(ctx), ctx)
    ctx += ori_ctx
    return ctx

#### STATE SETUP AND MANIPULATION FUNCTIONS ####
KEY_GEN_CONSTANTS = {
    32: np.frombuffer("expand 32-byte k".encode("ascii"), dtype=np.uint32),
    16: np.frombuffer("expand 16-byte k".encode("ascii"), dtype=np.uint32)
}

def setup_state(
        key: Union[bytes, jnp.ndarray],
        iv: Union[bytes, jnp.ndarray],
        counter: Union[int, jnp.ndarray]
    ) -> ChaChaState:

    if isinstance(key, bytes):
        if len(key) not in [16, 32]:
            raise ValueError("key must consist of 16 or 32 bytes")
        key = from_buffer(key)
    elif jax.lax.dtype(key) != ChaChaStateElementType or jnp.size(key) not in [4, 8]:
        raise ValueError("key must be a buffer or an array of 32-bit unsinged integers, totalling to 16 or 32 bytes!")


    if isinstance(iv, bytes):
        iv = from_buffer(iv)
    if jax.lax.dtype(iv) != ChaChaStateElementType or jnp.size(iv) != 3:
        raise ValueError("iv must be three 32-bit unsigned integers or a buffer of 12 bytes!")

    if jax.lax.dtype(counter) != ChaChaStateElementType or jnp.size(counter) != 1:
        raise ValueError("counter must be a single 32-bit unsigned integer!")

    key_bits = key.size * 4
    if key_bits == 16:
        key = jnp.tile(key, 2)
    key = key.reshape(2, 4)
    inputs = jnp.hstack((counter, iv))
    ctx = jnp.vstack((KEY_GEN_CONSTANTS[key_bits], key, inputs))
    return ctx

def from_buffer(buffer: bytes) -> jnp.ndarray:
    return jnp.array(np.frombuffer(buffer, dtype=np.uint32))

def increase_counter(ctx: ChaChaState, amount: Union[int, jnp.ndarray]) -> ChaChaState:
    return jax.ops.index_add(ctx, (3,0), amount, True, True)

def increment_counter(ctx: ChaChaState) -> ChaChaState:
    return increase_counter(ctx, 1)

def set_counter(ctx: ChaChaState, counter: Union[int, jnp.ndarray]) -> ChaChaState:
    return jax.ops.index_update(ctx, (3,0), counter, True, True)

def get_counter(ctx: ChaChaState) -> int:
    return int(ctx[3,0])

def set_nonce(ctx: ChaChaState, nonce: jnp.ndarray) -> ChaChaState:
    assert jnp.shape(nonce) == (3,)
    assert jnp.dtype(ctx) == jnp.dtype(nonce)
    return jax.ops.index_update(ctx, jax.ops.index[3, 1:4], nonce, True, True)

def get_nonce(ctx: ChaChaState) -> jnp.ndarray:
    return ctx[3, 1:3]

def serialize(ctx: ChaChaState, out_dtype: Type=jnp.uint8) -> jnp.ndarray:
    """ Converts a ChaCha state into a linear array of the specified type. """
    assert jax.lax.dtype(ctx) == ChaChaStateElementType
    type_info = jnp.iinfo(out_dtype)

    ctx = ctx.flatten()
    bit_width = type_info.bits
    if bit_width == 64:
        ctx = [jax.lax.convert_element_type(x, out_dtype) for x in jnp.split(ctx, 2)]
        ctx = jax.lax.shift_left(ctx[0], out_dtype(32)) | ctx[1]
    elif bit_width in [8, 16]:
        ctx = jax.lax.shift_right_logical(
            jax.lax.broadcast(ctx, (1,)),
            jax.lax.mul(
                jnp.uint32(bit_width),
                jax.lax.broadcasted_iota(jnp.uint32, (32 // bit_width, 1), 0)
            )
        )
        ctx = jax.lax.reshape(ctx, (jnp.size(ctx),), (1, 0))
        ctx = jax.lax.convert_element_type(ctx, out_dtype)

    return ctx

def encrypt(message: bytes, ctx: ChaChaState) -> Tuple[bytes, ChaChaState]:
    """ Encrypts a message of arbitrary length with the given ChaCha state. """
    block_nr = 0
    ciphertext = b''
    ChaChaStateByteSize = ChaChaStateBitSize >> 3
    while block_nr*ChaChaStateByteSize < len(message):
        message_block = np.frombuffer(
            message[block_nr*ChaChaStateByteSize : (block_nr+1)*ChaChaStateByteSize], dtype=np.uint8
        )

        key_stream = serialize(_block(ctx))[:len(message_block)]
        cipher_block = message_block ^ key_stream
        ciphertext += cipher_block.tobytes()

        ctx = increment_counter(ctx)
        block_nr += 1

    return ciphertext, ctx


def encrypt_with_key(
        message: bytes,
        key: Union[bytes, jnp.ndarray],
        iv: Union[bytes, jnp.ndarray],
        counter: Optional[Union[int, jnp.ndarray]]=0,
    ) -> bytes:
    """ Encrypts a message of arbitrary length with given key, IV and counter. """

    ctx = setup_state(key, iv, counter)
    ciphertext, _ = encrypt(message, ctx)

    return ciphertext
