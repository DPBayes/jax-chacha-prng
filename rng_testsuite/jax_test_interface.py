# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2021 Aalto University
""" Interfacing module for accessing JAX's default RNG implementation from TestU01 framework. """

import jax.config
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random
import numpy as np
from typing import Tuple

def create_context(seed: bytes) -> jax.random.PRNGKey:
    seed = int.from_bytes(seed, "big")
    seed = seed % (1 << 32)
    return jax.random.PRNGKey(seed)

def uniform_and_state_update(rng_key: jax.random.PRNGKey, count: int) -> Tuple[jnp.array, jax.random.PRNGKey]:
    """ must generate 64bit floats (doubles) as TestU01 expects at least precision 2^{-32},
        which is not the case for 32bit floats """
    next_rng_key, local_rng_key = jax.random.split(rng_key, 2)
    random_data = jax.random.uniform(local_rng_key, (count,), np.float64)
    return random_data, next_rng_key

def bits_and_state_update(rng_key: jax.random.PRNGKey, count: int) -> Tuple[jnp.array, jax.random.PRNGKey]:
    """ must generate 32bit integers """
    next_rng_key, local_rng_key = jax.random.split(rng_key, 2)
    random_data = jax.random._random_bits(local_rng_key, 32, (count,))
    return random_data, next_rng_key

def to_string(rng_key: jax.random) -> str:
    return f"rng key = {rng_key}"
