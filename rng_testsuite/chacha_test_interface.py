""" Interfacing module for accessing Python ChaCha20RNG implementation from TestU01 framework. """

import jax.config
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import chacha
import numpy as np
from typing import Tuple

def create_context(seed: bytes) -> jnp.array:
    ctx = chacha.PRNGKey(seed)
    return ctx

def uniform_and_state_update(rng_key: jnp.array, count: int) -> Tuple[jnp.array, jnp.array]:
    """ must generate 64bit floats (doubles) as TestU01 expects at least precision 2^{-32},
        which is not the case for 32bit floats """
    num_blocks = (count+7) // 8 # 512 bit state means 8 64 bit floats
    next_rng_key = chacha.chacha_increase_counter(rng_key, num_blocks) # skipping only half as far (num_blocks/2) passes small crush (but /3 or /4 do not) .. test again with more extensive test suites

    random_data = chacha.uniform(rng_key, (count,), np.float64)
    return random_data, next_rng_key

def bits_and_state_update(rng_key: jnp.array, count: int) -> Tuple[jnp.array, jnp.array]:
    """ must generate 32bit integers """
    num_blocks = (count+15) // 16 # 512 bit state means 16 32 bit uints
    next_rng_key = chacha.chacha_increase_counter(rng_key, num_blocks)

    random_data = chacha.random_bits(rng_key, 32, (count,))
    return random_data, next_rng_key

def to_string(rng_key: jnp.array) -> str:
    return str(rng_key)
