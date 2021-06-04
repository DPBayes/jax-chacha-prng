# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2021 Aalto University

import jax.numpy as jnp
import numpy as np
import jax
from typing import Union, Type, Optional, Tuple
np.set_printoptions(formatter={'int': hex})

# jax implementation of https://cr.yp.to/chacha/chacha-20080128.pdf
# following: https://tools.ietf.org/html/rfc7539

ChaChaState = jnp.array
ChaChaStateShape = (4, 4)
ChaChaStateElementCount = np.prod(ChaChaStateShape)
ChaChaStateElementType = jnp.uint32
ChaChaStateElementBitWidth = jnp.iinfo(ChaChaStateElementType).bits
ChaChaStateBitSize = ChaChaStateElementCount * ChaChaStateElementBitWidth

ChaChaKeySizeInBits = 256
ChaChaKeySizeInBytes = ChaChaKeySizeInBits >> 3
ChaChaKeySizeInWords = ChaChaKeySizeInBytes >> 2


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
    return jnp.array([a, b, c, d], dtype=jnp.uint32)


@jax.jit
def _double_round(state: ChaChaState) -> ChaChaState:
    assert state.shape == ChaChaStateShape
    assert state.dtype == ChaChaStateElementType

    qr_vmap = jax.vmap(_quarterround, in_axes=1, out_axes=1)

    # quarterround for each column
    state = qr_vmap(state)

    # moving diagonals into columns
    diag_map = jnp.array([
         0,  1,  2,  3,  # noqa: E126,E241,E131
         5,  6,  7,  4,  # noqa: E126,E241,E131
        10, 11,  8,  9,  # noqa: E126,E241,E131
        15, 12, 13, 14   # noqa: E126,E241,E131
    ])
    diag_state = state.ravel()[diag_map].reshape(4, 4)

    # quarterround for each column (aka, diagonal)
    diag_state = qr_vmap(diag_state)

    # undo previous mapping
    back_map = jnp.array([
         0,  1,  2,  3,  # noqa: E126,E241,E131
         7,  4,  5,  6,  # noqa: E126,E241,E131
        10, 11,  8,  9,  # noqa: E126,E241,E131
        13, 14, 15, 12   # noqa: E126,E241,E131
    ])
    state = diag_state.ravel()[back_map].reshape(4, 4)
    return state


@jax.jit
def _block(state: ChaChaState) -> ChaChaState:
    assert state.shape == ChaChaStateShape
    assert state.dtype == ChaChaStateElementType

    ori_state = state
    state = jax.lax.fori_loop(0, 10, lambda i, state: _double_round(state), state)
    state += ori_state
    return state


#### STATE SETUP AND MANIPULATION FUNCTIONS ####
KEY_GEN_CONSTANTS = {
    32: np.frombuffer("expand 32-byte k".encode("ascii"), dtype=np.uint32),
    16: np.frombuffer("expand 16-byte k".encode("ascii"), dtype=np.uint32)
}


def setup_state(
        key: Union[bytes, jnp.ndarray],
        iv: Union[bytes, jnp.ndarray],
        counter: Union[int, jnp.ndarray]
    ) -> ChaChaState:  # noqa:E121,E125

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

    if isinstance(counter, int):
        counter = jnp.uint32(counter)
    if jax.lax.dtype(counter) != ChaChaStateElementType or jnp.size(counter) != 1:
        raise ValueError("counter must be a single 32-bit unsigned integer!")

    key_bits = key.size * 4
    if key_bits == 16:
        key = jnp.tile(key, 2)
    key = key.reshape(2, 4)
    inputs = jnp.hstack((counter, iv))
    state = jnp.vstack((KEY_GEN_CONSTANTS[key_bits], key, inputs))
    return state


def from_buffer(buffer: bytes) -> jnp.ndarray:
    return jnp.array(np.frombuffer(buffer, dtype=np.uint32))


def increase_counter(state: ChaChaState, amount: Union[int, jnp.ndarray]) -> ChaChaState:
    return jax.ops.index_add(state, (3, 0), amount, True, True)


def increment_counter(state: ChaChaState) -> ChaChaState:
    return increase_counter(state, 1)


def set_counter(state: ChaChaState, counter: Union[int, jnp.ndarray]) -> ChaChaState:
    return jax.ops.index_update(state, (3, 0), counter, True, True)


def get_counter(state: ChaChaState) -> int:
    return int(state[3, 0])


def set_nonce(state: ChaChaState, nonce: jnp.ndarray) -> ChaChaState:
    assert jnp.shape(nonce) == (3,)
    assert jnp.dtype(state) == jnp.dtype(nonce)
    return jax.ops.index_update(state, jax.ops.index[3, 1:4], nonce, True, True)


def get_nonce(state: ChaChaState) -> jnp.ndarray:
    return state[3, 1:3]


def serialize(state: ChaChaState, out_dtype: Type = jnp.uint8) -> jnp.ndarray:
    """ Converts a ChaCha state into a linear array of the specified type. """
    assert jax.lax.dtype(state) == ChaChaStateElementType
    type_info = jnp.iinfo(out_dtype)

    state = state.flatten()
    bit_width = type_info.bits
    if bit_width == 64:
        state = [jax.lax.convert_element_type(x, out_dtype) for x in jnp.split(state, 2)]
        state = jax.lax.shift_left(state[0], out_dtype(32)) | state[1]
    elif bit_width in [8, 16]:
        state = jax.lax.shift_right_logical(
            jax.lax.broadcast(state, (1,)),
            jax.lax.mul(
                jnp.uint32(bit_width),
                jax.lax.broadcasted_iota(jnp.uint32, (32 // bit_width, 1), 0)
            )
        )
        state = jax.lax.reshape(state, (jnp.size(state),), (1, 0))
        state = jax.lax.convert_element_type(state, out_dtype)

    return state


def encrypt(message: bytes, state: ChaChaState) -> Tuple[bytes, ChaChaState]:
    """ Encrypts a message of arbitrary length with the given ChaCha state.

    Also returns the new state."""
    block_nr = 0
    ciphertext = b''
    ChaChaStateByteSize = ChaChaStateBitSize >> 3
    while block_nr * ChaChaStateByteSize < len(message):
        message_block = np.frombuffer(
            message[block_nr * ChaChaStateByteSize : (block_nr + 1) * ChaChaStateByteSize], dtype=np.uint8  # noqa: E203
        )

        key_stream = serialize(_block(state))[:len(message_block)]
        cipher_block = message_block ^ key_stream
        ciphertext += cipher_block.tobytes()

        state = increment_counter(state)
        block_nr += 1

    return ciphertext, state


def encrypt_with_key(
        message: bytes,
        key: Union[bytes, jnp.ndarray],
        iv: Union[bytes, jnp.ndarray],
        counter: Optional[Union[int, jnp.ndarray]] = 0,
    ) -> Tuple[bytes, int]:  # noqa:E121,E125
    """ Encrypts a message of arbitrary length with given key, IV and counter.

    Also returns the new counter value.
    """

    state = setup_state(key, iv, counter)
    ciphertext, state = encrypt(message, state)

    return ciphertext, get_counter(state)


def decrypt(ciphertext: bytes, state: ChaChaState) -> Tuple[bytes, ChaChaState]:
    """ Decrypts a ciphertext of arbitrary length with the given ChaCha state.

    Also returns the new state."""

    return encrypt(ciphertext, state)


def decrypt_with_key(
        ciphertext: bytes,
        key: Union[bytes, jnp.ndarray],
        iv: Union[bytes, jnp.ndarray],
        counter: Optional[Union[int, jnp.ndarray]] = 0,
    ) -> bytes:  # noqa:E121,E125
    """ Decrypts a ciphertext of arbitrary length with given key, IV and counter.

    Also returns the new counter value."""
    return encrypt_with_key(ciphertext, key, iv, counter)
