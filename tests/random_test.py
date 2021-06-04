# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2021 Aalto University

import unittest
import testconfig  # noqa
import jax.numpy as jnp

from chacha.random import _split, PRNGKey, random_bits, uniform
from chacha.cipher import set_counter
import numpy as np


class ChaChaRNGTests(unittest.TestCase):

    def test_random_bits(self) -> None:
        rng_key = jnp.array([
            [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574],
            [0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c],
            [0x13121110, 0x17161514, 0x1b1a1918, 0x1f1e1d1c],
            [0x00000001, 0x09000000, 0x4a000000, 0x00000000],
        ], dtype=jnp.uint32)

        shape = (17, 9)
        x = random_bits(rng_key, 8, shape)
        self.assertEqual(x.dtype, jnp.uint8)
        self.assertEqual(x.shape, shape)

        x = random_bits(rng_key, 16, shape)
        self.assertEqual(x.dtype, jnp.uint16)
        self.assertEqual(x.shape, shape)

        x = random_bits(rng_key, 32, shape)
        self.assertEqual(x.dtype, jnp.uint32)
        self.assertEqual(x.shape, shape)

        # needs 64bit jax mode
        x = random_bits(rng_key, 64, shape)
        self.assertEqual(x.dtype, jnp.uint64)
        self.assertEqual(x.shape, shape)

    def test_split(self) -> None:
        rng_key = jnp.array([
            [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574],
            [0x00000000, 0x00000000, 0x00000000, 0x00000000],
            [0x00000000, 0x00000000, 0x00000000, 0x00000000],
            [0x00000000, 0x00000000, 0x00000000, 0x00000000],
        ], dtype=jnp.uint32)

        # 6 splits require two blocks of randomness
        first_rng_key, second_rng_key, third_rng_key, fourth_rng_key, fifth_rng_key, sixth_rng_key = _split(rng_key, 6)

        expected_first = jnp.array([
            [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574],
            [0x00000000, 0x00000000, 0x00000000, 0x00000000],
            [0x00000000, 0x00000000, 0x00000000, 0x00000000],
            [0x00000000, 0xade0b876, 0x903df1a0, 0xe56a5d40]
        ], dtype=jnp.uint32)
        self.assertTrue(np.all(expected_first == first_rng_key))

        expected_second = jnp.array([
            [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574],
            [0x00000000, 0x00000000, 0x00000000, 0x00000000],
            [0x00000000, 0x00000000, 0x00000000, 0x00000000],
            [0x00000000, 0x28bd8653, 0xb819d2bd, 0x1aed8da0]
        ], dtype=jnp.uint32)
        self.assertTrue(np.all(expected_second == second_rng_key))

        expected_third = jnp.array([
            [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574],
            [0x00000000, 0x00000000, 0x00000000, 0x00000000],
            [0x00000000, 0x00000000, 0x00000000, 0x00000000],
            [0x00000000, 0xccef36a8, 0xc70d778b, 0x7c5941da]
        ], dtype=jnp.uint32)
        self.assertTrue(np.all(expected_third == third_rng_key))

        expected_fourth = jnp.array([
            [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574],
            [0x00000000, 0x00000000, 0x00000000, 0x00000000],
            [0x00000000, 0x00000000, 0x00000000, 0x00000000],
            [0x00000000, 0x8d485751, 0x3fe02477, 0x374ad8b8]
        ], dtype=jnp.uint32)
        self.assertTrue(np.all(expected_fourth == fourth_rng_key))

        expected_fifth = jnp.array([
            [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574],
            [0x00000000, 0x00000000, 0x00000000, 0x00000000],
            [0x00000000, 0x00000000, 0x00000000, 0x00000000],
            [0x00000000, 0xf4b8436a, 0x1ca11815, 0x69b687c3]
        ], dtype=jnp.uint32)
        self.assertTrue(np.all(expected_fifth == fifth_rng_key))

        expected_sixth = jnp.array([
            [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574],
            [0x00000000, 0x00000000, 0x00000000, 0x00000000],
            [0x00000000, 0x00000000, 0x00000000, 0x00000000],
            [0x00000000, 0x8665eeb2, 0xbee7079f, 0x7a385155]
        ], dtype=jnp.uint32)
        self.assertTrue(np.all(expected_sixth == sixth_rng_key))

    def test_PRNGKey_from_int(self) -> None:
        x = 9651354789628635673475235

        expected_rng_key = jnp.array([
            [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574],
            [0x00000000, 0x00000000, 0x00000000, 0x00000000],
            [0x00000000, 0xc0fb0700, 0x0412d4fd, 0xa338f3bf],
            [0x00000000, 0x00000000, 0x00000000, 0x00000000]
        ], dtype=jnp.uint32)

        rng_key = PRNGKey(x)
        self.assertTrue(np.all(expected_rng_key == rng_key))

    def test_PRNGKey_from_bytes(self) -> None:
        x = bytes(range(30))

        expected_rng_key = jnp.array([
            [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574],
            [0x03020100, 0x07060504, 0x0b0a0908, 0xf0e0d0c],
            [0x13121110, 0x17161514, 0x1b1a1918, 0x00001d1c],
            [0x00000000, 0x00000000, 0x00000000, 0x00000000]
        ], dtype=jnp.uint32)

        rng_key = PRNGKey(x)

        self.assertTrue(np.all(expected_rng_key == rng_key))

    def test_PRNGKey_from_bytes_rejects_too_long(self) -> None:
        x = bytes(range(33))

        with self.assertRaises(ValueError):
            PRNGKey(x)

    def test_uniform(self) -> None:
        """ verifies that the outputs of uniform have correct type, shape and bounds.

        does not test for uniformity; we have the rng testsuite for that.
        """
        rng_key = PRNGKey(0)
        shape = (3, 12)
        minval = 1
        maxval = 3
        dtype = jnp.float32
        data = uniform(rng_key, shape, dtype, minval, maxval)

        self.assertEqual(jnp.float32, data.dtype)
        self.assertEqual(shape, data.shape)
        self.assertTrue(np.all(data >= minval))
        self.assertTrue(np.all(data <= maxval))

    def test_random_bits_consistency(self) -> None:
        """ verifies that there is no difference between sampling two blocks directly
            or separately (while manually increasing counter) """
        rng_key = PRNGKey(0)
        single_vals = random_bits(rng_key, 32, (32,))
        multi_vals_1 = random_bits(rng_key, 32, (16,))
        rng_key_2 = set_counter(rng_key, 1)
        multi_vals_2 = random_bits(rng_key_2, 32, (16,))
        self.assertTrue(np.all(single_vals[:16] == multi_vals_1))
        self.assertTrue(np.all(single_vals[16:] == multi_vals_2))


if __name__ == '__main__':
    unittest.main()
