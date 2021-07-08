# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2021 Aalto University

""" A JAX-accelerated implementation of the 20-round ChaCha cipher.

This module sets up the native implementation of the ChaCha20 block function as JAX ops.
"""

from functools import partial

import jax
from jax.lib import xla_client
from jax.interpreters import xla, batching
import jax.numpy as jnp
import numpy as np  #type: ignore

from chacha.native import gpu_chacha20_block_factory, cpu_chacha20_block_factory

# importing ShapedArray and dtypes
try:
    # pre jax v0.2.14 location
    from jax.abstract_arrays import ShapedArray  # type: ignore
    from jax import dtypes  # type: ignore
except (AttributeError, ImportError):
    # post jax v0.2.14 location
    try:
        from jax._src.abstract_arrays import ShapedArray  # type: ignore
        from jax._src import dtypes  # type: ignore
    except (AttributeError, ImportError):
        raise ImportError("Cannot import ShapedArray and dtypes. "
                          "You are probably using an incompatible version of jax.")


xla_client.register_custom_call_target(b"gpu_chacha20_block", gpu_chacha20_block_factory(), platform="gpu")
xla_client.register_cpu_custom_call_target(b"cpu_chacha20_block", cpu_chacha20_block_factory())


def _chacha20_block_gpu_translation(c, state):
    state_xla_shape = c.get_shape(state)
    state_shape = state_xla_shape.dimensions()

    if len(state_shape) < 2 or state_shape[-2:] != (4, 4):
        raise ValueError("state must be at least two-dimensional and last two dimensions must have size 4")
    batch_dims = state_shape[:-2]
    num_states_bytes = int(np.prod(batch_dims)).to_bytes(4, 'little')

    return xla_client.ops.CustomCallWithLayout(
        c, b"gpu_chacha20_block", operands=(state,), operand_shapes_with_layout=(state_xla_shape,),
        shape_with_layout=state_xla_shape,
        opaque=num_states_bytes
    )


def _chacha20_block_cpu_translation(c, state):
    state_xla_shape = c.get_shape(state)
    state_shape = state_xla_shape.dimensions()

    if len(state_shape) < 2 or state_shape[-2:] != (4, 4):
        raise ValueError("state must be at least two-dimensional and last two dimensions must have size 4")
    batch_dims = state_shape[:-2]
    num_states = np.prod(batch_dims).astype(np.uint32)

    num_states = xla_client.ops.ConstantLiteral(c, num_states)
    num_states_xla_shape = xla_client.Shape.array_shape(np.dtype(np.uint32), (), ())

    call_ret = xla_client.ops.CustomCallWithLayout(
        c, b"cpu_chacha20_block", operands=(num_states, state),
        operand_shapes_with_layout=(num_states_xla_shape, state_xla_shape),
        shape_with_layout=state_xla_shape
    )
    return call_ret


def _chacha20_block_abstract_eval(state):
    state_shape = state.shape
    dtype = dtypes.canonicalize_dtype(state.dtype)
    return ShapedArray(state_shape, dtype)


chacha20_p = jax.core.Primitive("chacha20")
chacha20_p.multiple_results = False
chacha20_p.def_abstract_eval(_chacha20_block_abstract_eval)
chacha20_p.def_impl(partial(xla.apply_primitive, chacha20_p))

xla.backend_specific_translations["cpu"][chacha20_p] = _chacha20_block_cpu_translation
xla.backend_specific_translations["gpu"][chacha20_p] = _chacha20_block_gpu_translation


def chacha20_block(state):
    assert jnp.shape(state) in ((4, 4), (16, 1), (16,))
    assert jnp.dtype(state) == jnp.uint32

    # CAUTION: currently implicitly assumes that 4x4 matrix is represented as row-major array

    return chacha20_p.bind(state)


## BATCHING RULE, FOR VMAP
def _chacha20_block_batch(states, batch_axes):
    # TODO: currently somewhat limited, only allows batching over one additional axis
    states = states[0]
    batch_axis = batch_axes[0]

    # assert len(states.shape) >= 2
    # assert batch_axis < len(states.shape) - 2
    assert len(states.shape) == 3
    assert batch_axis == 0
    assert states.shape[-2:] == (4, 4)

    out_states = chacha20_p.bind(states)

    return out_states, batch_axis


batching.primitive_batchers[chacha20_p] = _chacha20_block_batch
