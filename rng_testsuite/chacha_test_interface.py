# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2021,2022 Aalto University
""" Interfacing module for accessing Python ChaCha20RNG implementation from TestU01 framework. """

import jax.config
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from chacha.random import PRNGKey, uniform, random_bits, split, ErrorFlag
from chacha.cipher import increase_counter, get_counter
import numpy as np
from typing import Tuple


def create_context(seed: bytes) -> jnp.array:
    ctx = PRNGKey(seed)
    return ctx


def uniform_and_state_update(rng_key: jnp.array, count: int) -> Tuple[jnp.array, jnp.array]:
    """ must generate 64bit floats (doubles) as TestU01 expects at least precision 2^{-32},
        which is not the case for 32bit floats """
    num_blocks = (count+7) // 8  # 512 bit state means 8 64 bit floats
    next_rng_key = increase_counter(rng_key, num_blocks)

    # making many repeated requests might overflow the state's counter,
    # in that case we split the state and start with a fresh counter
    if get_counter(next_rng_key) < get_counter(rng_key):
        rng_key = split(rng_key, 1)[0]
        next_rng_key = increase_counter(rng_key, num_blocks)

    random_data = uniform(rng_key, (count,), np.float64)
    return np.array(random_data), next_rng_key


def bits_and_state_update(rng_key: jnp.array, count: int) -> Tuple[jnp.array, jnp.array]:
    """ must generate 32bit integers """
    num_blocks = (count+15) // 16  # 512 bit state means 16 32 bit uints
    next_rng_key = increase_counter(rng_key, num_blocks)

    # making many repeated requests might overflow the state's counter,
    # in that case we split the state and start with a fresh counter
    if get_counter(next_rng_key) < get_counter(rng_key):
        rng_key = split(rng_key, 1)[0]
        next_rng_key = increase_counter(rng_key, num_blocks)

    random_data, error = random_bits(rng_key, 32, (count,))

    if error:
        ef = ErrorFlag(error)
        raise Exception(f"An unexpected error occured when generating randomness {str(ef)}")

    return np.array(random_data), next_rng_key


def to_string(rng_key: jnp.array) -> str:
    return str(rng_key)
