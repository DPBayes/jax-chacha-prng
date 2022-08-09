# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2021,2022 Aalto University

""" A JAX-accelerated implementation of the 20-round ChaCha cipher.

Basic definitions/constants.
"""

import jax.numpy as jnp
import numpy as np  # type: ignore
from typing import Any, Callable
import functools

ChaChaStateShape = (4, 4)
ChaChaStateElementCount = np.prod(ChaChaStateShape)
ChaChaStateElementType = jnp.uint32
ChaChaStateElementBitWidth = jnp.iinfo(ChaChaStateElementType).bits
ChaChaStateBitSize = ChaChaStateElementCount * ChaChaStateElementBitWidth

# "Word" == "Element", i.e., WordSize = ChaChaStateElementType (= 32 bits)
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
