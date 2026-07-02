# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Aalto University

""" A JAX-accelerated implementation of the 20-round ChaCha cipher.

This module sets up the native implementation of the ChaCha20 block function as JAX ops.
"""

from chacha.defs import ChaChaState, ChaChaStateShape
from functools import partial

from enum import Enum
from typing import Callable

import jax
import jax.numpy as jnp
import jax._src.core

import chacha.native

import jax.ffi
jax.ffi.register_ffi_target("cpu_chacha20_block", chacha.native.cpu_chacha20_block_factory(), platform="cpu") # jax.ffi.register_ffi_target(

if chacha.native.cuda_supported():
    jax.ffi.register_ffi_target("cuda_chacha20_block", chacha.native.cuda_chacha20_block_factory(), platform="CUDA")

if chacha.native.hip_supported():
    jax.ffi.register_ffi_target("rocm_chacha20_block", chacha.native.rocm_chacha20_block_factory(), platform="ROCM")

def chacha20_block(state: ChaChaState) -> ChaChaState:
    state_shape = jnp.shape(state)
    # even if the total `state` array that ends up being passed in here has arbitrary leading
    # batch dimensions, under vmap shape only shows up as that of a single state
    if state_shape != ChaChaStateShape:
        raise ValueError(
            "Argument to chacha20_block has wrong shape. Did you pass a ChaCha state? "
            f"Must be {ChaChaStateShape} but was {state_shape}"
        )

    if state.dtype != jnp.uint32:
        raise ValueError(
            "Argument to chacha20_block did have unexpected type. Did you pass a ChaCha state? "
            f"Got: {state.type}, expected uint32."
        )
    
    shapeDtype = jax.ShapeDtypeStruct(state_shape, jnp.uint32)
    
    # CAUTION: currently implicitly assumes that 4x4 matrix is represented as row-major array
    def make_ffi_call(implementation_name: str) -> Callable[[ChaChaState], ChaChaState]:
        return jax.ffi.ffi_call(
            implementation_name, shapeDtype, vmap_method="broadcast_all"
        )

    return jax.lax.platform_dependent(
        state, cpu=make_ffi_call("cpu_chacha20_block"), cuda=make_ffi_call("cuda_chacha20_block"), rocm=make_ffi_call("rocm_chacha20_block")
    )
