# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2021 Aalto University

""" A JAX-accelerated implementation of the 20-round ChaCha cipher.

This module sets up the native implementation of the ChaCha20 block function as JAX ops.
"""

from chacha.defs import ChaChaState
from functools import partial

from enum import Enum
from typing import Any, Tuple, Union, Sequence

import jax
from jax.lib import xla_client
from jax.interpreters import xla, batching, mlir
from jaxlib.mlir import ir
import jax.numpy as jnp
import numpy as np  # type: ignore

import chacha.native

# importing ShapedArray and dtypes
try:
    # pre jax v0.2.14 location
    from jax.abstract_arrays import ShapedArray  # type: ignore
    from jax import dtypes  # type: ignore
except (AttributeError, ImportError):  # pragma: no cover
    # post jax v0.2.14 location
    try:
        from jax._src.abstract_arrays import ShapedArray  # type: ignore
        from jax._src import dtypes  # type: ignore
    except (AttributeError, ImportError):  # pragma: no cover
        raise ImportError("Cannot import ShapedArray and dtypes. "
                          "You are probably using an incompatible version of jax.")

# importing XlaOp
try:
    # pre jax v0.2.13 location
    XlaOp = jax.xla.XlaOp  # type: ignore
except (AttributeError, ImportError):  # pragma: no cover
    # post jax v0.2.13 location
    try:
        from jaxlib.xla_client import XlaOp  # type: ignore
    except (AttributeError, ImportError):  # pragma: no cover
        raise ImportError("Cannot import XlaOp. "
                          "You are probably using an incompatible version of jax.")

try:
    from jaxlib.hlo_helpers import custom_call
except (AttributeError, ImportError):
    try:
        from jaxlib.mhlo_helpers import custom_call
    except (AttributeError, ImportError):
        raise ImportError("Cannot import custom_call. "
                          "You are probably using an incompatible version of jax.")

xla_client.register_cpu_custom_call_target(
    b"cpu_chacha20_block", chacha.native.cpu_chacha20_block_factory()
)


class Platform(Enum):
    CPU = 1
    GPU = 2


def _chacha20_block_translation(
        platform: Platform,
        ctx: mlir.LoweringRuleContext, 
        states: Union[ir.Value, Sequence[ir.Value]]
        ) -> Sequence[Union[ir.Value, Sequence[ir.Value]]]:
    state_type = mlir.ir.RankedTensorType(states.type)
    state_shape = state_type.shape
    layout = tuple(range(len(state_shape) - 1, -1, -1))

    assert len(state_shape) >= 2
    assert tuple(state_shape)[-2:] == (4, 4)

    batch_dims = state_shape[:-2]
    num_states = np.prod(batch_dims).astype(np.uint32)

    if platform == Platform.CPU:
        call_ret = custom_call(
            b"cpu_chacha20_block",
            out_types=[state_type],
            operands=[mlir.ir_constant(num_states), states],
            operand_layouts=[(), layout],
            result_layouts=[layout]
        )
    else:
        call_ret = custom_call(
            b"gpu_chacha20_block",
            out_types=[state_type],
            operands=[states],
            operand_layouts=[layout],
            result_layouts=[layout]
        )
    return (call_ret,)


def _chacha20_block_cpu_translation(
        ctx: mlir.LoweringRuleContext, 
        states: Union[ir.Value, Sequence[ir.Value]]
        ) -> Sequence[Union[ir.Value, Sequence[ir.Value]]]:
    return _chacha20_block_translation(Platform.CPU, ctx, states)


def _chacha20_block_gpu_translation(
        ctx: mlir.LoweringRuleContext, 
        states: Union[ir.Value, Sequence[ir.Value]]
        ) -> Sequence[Union[ir.Value, Sequence[ir.Value]]]:
    return _chacha20_block_translation(Platform.GPU, ctx, states)


def _chacha20_block_abstract_eval(states: jnp.ndarray) -> ShapedArray:
    state_shape = states.shape
    dtype = dtypes.canonicalize_dtype(states.dtype)
    abstract_output = ShapedArray(state_shape, dtype)
    return abstract_output
   

chacha20_p = jax.core.Primitive("chacha20")
chacha20_p.multiple_results = False
chacha20_p.def_impl(partial(xla.apply_primitive, chacha20_p))
chacha20_p.def_abstract_eval(_chacha20_block_abstract_eval)

# xla.backend_specific_translations["cpu"][chacha20_p] = _chacha20_block_cpu_translation
mlir.register_lowering(chacha20_p, _chacha20_block_cpu_translation, platform="cpu")

if chacha.native.cuda_supported():
    xla_client.register_custom_call_target(
        b"gpu_chacha20_block", chacha.native.gpu_chacha20_block_factory(), platform="gpu"
    )
    mlir.register_lowering(chacha20_p, _chacha20_block_gpu_translation, platform="gpu")


## BATCHING RULE, FOR VMAP
def _chacha20_block_batch(states: jnp.ndarray, batch_axes: Tuple[int]) -> Tuple[jnp.ndarray, int]:
    # Our CPU/GPU primitive can already batch over arbitrary leading
    # dimensions. So there's not really anything to do here.
    # We don't even need to really care about which axis should be batched over:
    # jax will take care of taking out the correct axis for nested calls to vmap
    # and mapping it back to outputs, and in the end we will map over all potential batch
    # dimensions, so we do not need to move any axes around here.

    states = states[0]
    batch_axis = batch_axes[0]

    assert batch_axis < len(states.shape) - 2
    assert len(states.shape) >= 3
    assert states.shape[-2:] == (4, 4)

    out_states = chacha20_p.bind(states)
    return out_states, batch_axis


batching.primitive_batchers[chacha20_p] = _chacha20_block_batch


def chacha20_block(state: ChaChaState) -> ChaChaState:
    if jnp.shape(state) not in ((4, 4), (16, 1), (16,)):
        raise ValueError(
            f"Argument to chacha20_block did have unexpected shape. Did you pass a ChaCha state? "
            f"Got: {jnp.shape(state)}, expected (4,4)."
        )
    if jnp.dtype(state) != jnp.uint32:
        raise ValueError(
            f"Argument to chacha20_block did have unexpected type. Did you pass a ChaCha state? "
            f"Got: {jnp.dtype(state)}, expected uint32."
        )

    # CAUTION: currently implicitly assumes that 4x4 matrix is represented as row-major array
    ret = chacha20_p.bind(state)
    return ret
