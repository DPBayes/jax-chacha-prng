# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2021 Aalto University

""" A JAX-accelerated implementation of the 20-round ChaCha cipher.

The cipher is due to Daniel J. Bernstein and described in https://cr.yp.to/chacha/chacha-20080128.pdf.
The implementation follows a slight variation specified in https://tools.ietf.org/html/rfc7539.
"""

import jax.numpy as jnp
import numpy as np
import jax
from typing import Callable, Union, Type, Tuple, Any
import functools

ChaChaStateShape = (4, 4)
ChaChaStateElementCount = np.prod(ChaChaStateShape)
ChaChaStateElementType = jnp.uint32
ChaChaStateElementBitWidth = jnp.iinfo(ChaChaStateElementType).bits
ChaChaStateBitSize = ChaChaStateElementCount * ChaChaStateElementBitWidth

ChaChaKeySizeInBits = 256
ChaChaKeySizeInBytes = ChaChaKeySizeInBits >> 3
ChaChaKeySizeInWords = ChaChaKeySizeInBytes >> 2

ChaChaNonceSizeInBits = 96
ChaChaNonceSizeInBytes = ChaChaNonceSizeInBits >> 3
ChaChaNonceSizeInWords = ChaChaNonceSizeInBytes >> 2

ChaChaCounterSizeInBits = 32
ChaChaCounterSizeInBytes = ChaChaCounterSizeInBits >> 3
ChaChaCounterSizeInWords = ChaChaCounterSizeInBytes >> 2


class ChaChaState(jnp.ndarray):

    def __new__(cls, *args: Any, **kwargs: Any) -> "ChaChaState":
        arr = jnp.array(*args, **kwargs)
        if arr.shape != ChaChaStateShape:
            raise ValueError(f"ChaChaState must have shape {ChaChaStateShape}; got {arr.shape}.")
        if arr.dtype != ChaChaStateElementType:
            raise TypeError(f"ChaChaState must have dtype {ChaChaStateElementType}; got {arr.dtype}.")
        return arr


def state_verified(state_arg_pos: int = 0) -> Callable:
    """ Decorator that ensures that a valid ChaChaState is passed into the function.

    Args:
      state_arg_pos: The position of the state argument in the decorated function's argument list.
    """
    def inner_decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapped_func(*args: Any, **kwargs: Any) -> Any:
            arg_list = list(args)
            arg_list[state_arg_pos] = ChaChaState(args[state_arg_pos])
            return func(*arg_list, **kwargs)
        return wrapped_func
    return inner_decorator


#### CORE CHACHA ROUND FUNCTIONS ####

def rotate_left(x: jnp.ndarray, num_bits: int) -> jnp.ndarray:
    dtype = jax.lax.dtype(x)
    issubclass(dtype.type, jnp.unsignedinteger)
    type_info = jnp.iinfo(dtype)
    return (x << num_bits) ^ (x >> (type_info.bits - num_bits))


@jax.jit
def _quarterround(x: jnp.ndarray) -> jnp.ndarray:
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
    state = ChaChaState(qr_vmap(state))

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
    state += ori_state  # type: ignore
    return state


#### STATE SETUP AND MANIPULATION FUNCTIONS ####
KEY_GEN_CONSTANTS = {
    32: np.frombuffer("expand 32-byte k".encode("ascii"), dtype=np.uint32),
    16: np.frombuffer("expand 16-byte k".encode("ascii"), dtype=np.uint32)
}


def setup_state(
        key: Union[bytes, jnp.ndarray],
        iv: Union[bytes, jnp.ndarray],
        counter: Union[int, jnp.ndarray] = 0
    ) -> ChaChaState:  # noqa:E121,E125
    """ Initializes and returns a ChaChaState given key, iv/nonce and an optional initial counter.

    Args:
      key: A 256 bit key, either as a bytes object or a size 32 array of 32 bit integers.
      iv: A 96 bit initialization vector / nonce, either as a bytes object or a size 3 array of 32 bit integers.
      counter: An optional initial counter values. Defaults to 0.

    Returns:
      The initialized ChaCha cipher state for the given parameters.
    """

    if isinstance(key, bytes):
        if len(key) not in [16, 32]:
            raise ValueError("key must consist of 16 or 32 bytes")
        key_array = _from_buffer(key)
    elif jax.lax.dtype(key) != ChaChaStateElementType or jnp.size(key) not in [4, 8]:
        raise ValueError("key must be a buffer or an array of 32-bit unsinged integers, totalling to 16 or 32 bytes!")
    else:
        assert isinstance(key, jnp.ndarray)
        key_array = jnp.array(key)

    if isinstance(iv, bytes):
        if len(iv) != ChaChaNonceSizeInBytes:
            raise ValueError("iv must consits of 12 bytes")
        iv_array = _from_buffer(iv)
    elif jax.lax.dtype(iv) != ChaChaStateElementType or jnp.size(iv) != ChaChaNonceSizeInWords:
        raise ValueError("iv must be three 32-bit unsigned integers or a buffer of 12 bytes!")
    else:
        assert isinstance(iv, jnp.ndarray)
        iv_array = iv

    if isinstance(counter, int):
        counter = jnp.uint32(counter)
    elif jax.lax.dtype(counter) != ChaChaStateElementType or jnp.size(counter) != ChaChaCounterSizeInWords:
        raise ValueError("counter must be a single 32-bit unsigned integer!")

    key_bits = key_array.size * 4
    if key_bits == 16:
        key_array = jnp.tile(key_array, 2)
    key_array = key_array.reshape(2, 4)  # type: ignore # numpy seems confused about this
    inputs = jnp.hstack((counter, iv_array))
    state = ChaChaState(jnp.vstack((KEY_GEN_CONSTANTS[key_bits], key_array, inputs)))
    return state


def _from_buffer(buffer: bytes) -> jnp.ndarray:
    return jnp.array(np.frombuffer(buffer, dtype=np.uint32))


@state_verified()
def increase_counter(state: ChaChaState, amount: Union[int, jnp.ndarray]) -> ChaChaState:
    """ Increases the counter value in a given ChaCha cipher state by the given amount.

    Args:
      state: The ChaCha cipher state.
      amount: The amount by which to increase the counter.

    Returns:
      The new ChaCha cipher state with increased counter value.
    """
    return jax.ops.index_add(state, (3, 0), amount, True, True)


@state_verified()
def increment_counter(state: ChaChaState) -> ChaChaState:
    """ Increments the counter value in a given ChaCha cipher state.

    Args:
      state: The ChaCha cipher state.

    Returns:
      The new ChaCha cipher state with incremented counter value.
    """
    return increase_counter(state, 1)


@state_verified()
def set_counter(state: ChaChaState, counter: Union[int, jnp.ndarray]) -> ChaChaState:
    """ Sets the counter value in a given ChaCha cipher state to the given value.

    Args:
      state: The ChaCha cipher state.
      counter: The new counter value.

    Returns:
      The new ChaCha cipher state with the given counter value.
    """
    return jax.ops.index_update(state, (3, 0), counter, True, True)


@state_verified()
def get_counter(state: ChaChaState) -> int:
    """ Returns the counter value of a given ChaCha cipher state.

    Args:
      state: The ChaCha cipher state.

    Returns:
      The counter value of the given state.
    """
    return int(state[3, 0])


@state_verified()
def set_nonce(state: ChaChaState, nonce: jnp.ndarray) -> ChaChaState:
    """ Sets the nonce/IV of the given ChaCha cipher state.

    Args:
      state: The ChaCha cipher state.
      nonce: A 96 bit nonce given as an array of three 32 bit integers.

    Returns:
      The new ChaCha cipher state with the given nonce value.
    """
    assert jnp.shape(nonce) == (3,)
    assert jnp.dtype(state) == jnp.dtype(nonce)
    return jax.ops.index_update(state, jax.ops.index[3, 1:4], nonce, True, True)


@state_verified()
def get_nonce(state: ChaChaState) -> jnp.ndarray:
    """ Returns the nonce/IV of the given ChaCha cipher state.

    Args:
      state: The ChaCha cipher state.

    Returns:
      An array of three 32 bit integers containing the value of the nonce/IV.
    """
    return state[3, 1:4]


def serialize(arr: jnp.ndarray, out_dtype: Type = jnp.uint8) -> jnp.ndarray:
    """Converts an array of arbitrary shape into a linear array of the specified type.

    Args:
      arr: An array of arbitrary shape and dtype `uint32`.
      out_dtype: The desired `numpy.dtype` of the output array.

    Returns:
      A one-dimensional array with the given type, containing a serialized
      representation of the given array.
    """
    assert jax.lax.dtype(arr) == ChaChaStateElementType
    type_info = jnp.iinfo(out_dtype)

    serialization = arr.flatten()
    bit_width = type_info.bits
    if bit_width == 64:
        serialization = jnp.array([jax.lax.convert_element_type(x, out_dtype) for x in jnp.split(serialization, 2)])
        serialization = jax.lax.shift_left(serialization[0], out_dtype(32)) | serialization[1]
    elif bit_width in [8, 16]:
        serialization = jax.lax.shift_right_logical(
            jax.lax.broadcast(serialization, (1,)),
            jax.lax.mul(
                jnp.uint32(bit_width),
                jax.lax.broadcasted_iota(jnp.uint32, (32 // bit_width, 1), 0)
            )
        )
        serialization = jax.lax.reshape(serialization, (jnp.size(serialization),), (1, 0))
        serialization = jax.lax.convert_element_type(serialization, out_dtype)

    return serialization  # type: ignore # numpy seems confused about np.ndarray and jnp.ndarray here


@state_verified(state_arg_pos=1)
def encrypt(message: bytes, state: ChaChaState) -> Tuple[bytes, ChaChaState]:
    """Encrypts a message of arbitrary length with the given ChaCha cipher state.

    Args:
      message: The message to encrypt as bytes object.
      state: The ChaCha cipher state used for encrypting the message.

    Returns:
      A tuple consisting of:
      - The ciphertext resulting from the encryption as a bytes object.
      - The ChaCha cipher with updated counter value after the encryption.
    """
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
        counter: Union[int, jnp.ndarray] = 0,
    ) -> Tuple[bytes, int]:  # noqa:E121,E125
    """ Encrypts a message of arbitrary length with given key, IV and counter.

    Args:
      message: The message to encrypt as bytes object.
      key: A 256 bit key, either as a bytes object or a size 32 array of 32 bit integers.
      iv: A 96 bit initialization vector / nonce, either as a bytes object or a size 3 array of 32 bit integers.
      counter: An optional initial counter values. Defaults to 0.

    Returns:
      A tuple consisting of:
      - The ciphertext resulting from the encryption as a bytes object.
      - The counter value after the encryption.
    """

    state = setup_state(key, iv, counter)
    ciphertext, state = encrypt(message, state)

    return ciphertext, get_counter(state)


@state_verified(state_arg_pos=1)
def decrypt(ciphertext: bytes, state: ChaChaState) -> Tuple[bytes, ChaChaState]:
    """Decrypts a ciphertext of arbitrary length with the given ChaCha cipher state.

    Args:
      ciphertext: The ciphertext to decrypt as bytes object.
      state: The ChaCha cipher state used for decrypting the message.

    Returns:
      A tuple consisting of:
      - The plaintext resulting from the decryption as a bytes object.
      - The ChaCha cipher with updated counter value after the decryption.
    """

    return encrypt(ciphertext, state)


def decrypt_with_key(
        ciphertext: bytes,
        key: Union[bytes, jnp.ndarray],
        iv: Union[bytes, jnp.ndarray],
        counter: Union[int, jnp.ndarray] = 0,
    ) -> Tuple[bytes, int]:  # noqa:E121,E125
    """ Decrypts a ciphertext of arbitrary length with given key, IV and counter.


    Args:
      ciphertext: The ciphertext to decrypt as bytes object.
      key: A 256 bit key, either as a bytes object or a size 32 array of 32 bit integers.
      iv: A 96 bit initialization vector / nonce, either as a bytes object or a size 3 array of 32 bit integers.
      counter: An optional initial counter values. Defaults to 0.

    Returns:
      A tuple consisting of:
      - The plaintext resulting from the decryption as a bytes object.
      - The counter value after the decryption.
    """
    return encrypt_with_key(ciphertext, key, iv, counter)
