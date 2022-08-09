# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2021,2022 Aalto University

import unittest
import testconfig  # noqa
import jax.numpy as jnp
import jax

from chacha.random import split, PRNGKey, random_bits, uniform, _uniform, is_state_invalidated, ErrorFlag
from chacha.cipher import set_counter
import numpy as np  # type: ignore


class ChaChaRNGTests(unittest.TestCase):

    def test_random_bits(self) -> None:
        rng_key = jnp.array([
            [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574],
            [0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c],
            [0x13121110, 0x17161514, 0x1b1a1918, 0x1f1e1d1c],
            [0x00000001, 0x09000000, 0x4a000000, 0x00000000],
        ], dtype=jnp.uint32)

        shape = (17, 9)
        x, error = random_bits(rng_key, 8, shape)
        self.assertEqual(0, error)
        self.assertEqual(x.dtype, jnp.uint8)
        self.assertEqual(x.shape, shape)

        x, error = random_bits(rng_key, 16, shape)
        self.assertEqual(0, error)
        self.assertEqual(x.dtype, jnp.uint16)
        self.assertEqual(x.shape, shape)

        x, error = random_bits(rng_key, 32, shape)
        self.assertEqual(0, error)
        self.assertEqual(x.dtype, jnp.uint32)
        self.assertEqual(x.shape, shape)

        # needs 64bit jax mode
        x, error = random_bits(rng_key, 64, shape)
        self.assertEqual(0, error)
        self.assertEqual(x.dtype, jnp.uint64)
        self.assertEqual(x.shape, shape)

    def test_is_state_invalidated(self) -> None:
        rng_key = jnp.array([
            [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574],
            [0x00000000, 0x00000000, 0x00000000, 0x00000000],
            [0x00000000, 0x00000000, 0x00000000, 0x00000000],
            [0x00000000, 0x00000000, 0x00000000, 0x00000001],
        ], dtype=jnp.uint32)

        self.assertFalse(is_state_invalidated(rng_key))

        rng_key = jnp.array([
            [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574],
            [0x00000000, 0x00000000, 0x00000000, 0x00000000],
            [0x00000000, 0x00000000, 0x00000000, 0x00000000],
            [0x00000000, 0x00000000, 0x00000000, 0x00000000],
        ], dtype=jnp.uint32)

        self.assertTrue(is_state_invalidated(rng_key))

    def test_split_from_initial(self) -> None:
        rng_key = jnp.array([
            [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574],
            [0x00000000, 0x00000000, 0x00000000, 0x00000000],
            [0x00000000, 0x00000000, 0x00000000, 0x00000000],
            [0xe56a5d40, 0x00000000, 0x00000000, 0x00000001],
        ], dtype=jnp.uint32)

        # 6 splits from initial
        split_keys = split(rng_key, 6)

        expected = np.array([
            [
                [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574],
                [0x00000000, 0x00000000, 0x00000000, 0x00000000],
                [0x00000000, 0x00000000, 0x00000000, 0x00000000],
                [0x00000000, 0x00000000, 0x00000000, 0x00000008]
            ],
            [
                [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574],
                [0x00000000, 0x00000000, 0x00000000, 0x00000000],
                [0x00000000, 0x00000000, 0x00000000, 0x00000000],
                [0x00000000, 0x00000000, 0x00000000, 0x00000009]
            ],
            [
                [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574],
                [0x00000000, 0x00000000, 0x00000000, 0x00000000],
                [0x00000000, 0x00000000, 0x00000000, 0x00000000],
                [0x00000000, 0x00000000, 0x00000000, 0x0000000a]
            ],
            [
                [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574],
                [0x00000000, 0x00000000, 0x00000000, 0x00000000],
                [0x00000000, 0x00000000, 0x00000000, 0x00000000],
                [0x00000000, 0x00000000, 0x00000000, 0x0000000b]
            ],
            [
                [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574],
                [0x00000000, 0x00000000, 0x00000000, 0x00000000],
                [0x00000000, 0x00000000, 0x00000000, 0x00000000],
                [0x00000000, 0x00000000, 0x00000000, 0x0000000c]
            ],
            [
                [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574],
                [0x00000000, 0x00000000, 0x00000000, 0x00000000],
                [0x00000000, 0x00000000, 0x00000000, 0x00000000],
                [0x00000000, 0x00000000, 0x00000000, 0x0000000d]
            ],
        ], dtype=np.uint32)

        self.assertTrue(np.all(expected == split_keys))

    def test_split_word_boundaries(self) -> None:
        rng_key = jnp.array([
            [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574],
            [0xbbbbbbbb, 0x00000000, 0x00000000, 0x00000000],
            [0x00000000, 0x00000000, 0x00000000, 0xaaaaaaaa],
            [0x00000000, 0x00000000, 0x40000000, 0x80000000],
        ], dtype=jnp.uint32)

        split_keys = split(rng_key, 3)

        expected = np.array([
            [
                [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574],
                [0xbbbbbbbb, 0x00000000, 0x00000000, 0x00000000],
                [0x00000000, 0x00000000, 0x00000000, 0xaaaaaaaa],
                [0x00000000, 0x00000001, 0x00000002, 0x00000000]
            ],
            [
                [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574],
                [0xbbbbbbbb, 0x00000000, 0x00000000, 0x00000000],
                [0x00000000, 0x00000000, 0x00000000, 0xaaaaaaaa],
                [0x00000000, 0x00000001, 0x00000002, 0x00000001]
            ],
            [
                [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574],
                [0xbbbbbbbb, 0x00000000, 0x00000000, 0x00000000],
                [0x00000000, 0x00000000, 0x00000000, 0xaaaaaaaa],
                [0x00000000, 0x00000001, 0x00000002, 0x00000002]
            ],

        ], dtype=np.uint32)

        self.assertTrue(np.all(expected == split_keys))

    def test_split_invalidates_when_split_limit_exceeded(self) -> None:
        rng_key = jnp.array([
            [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574],
            [0xbbbbbbbb, 0x00000000, 0x00000000, 0x00000000],
            [0x00000000, 0x00000000, 0x00000000, 0xaaaaaaaa],
            [0xbe57823a, 0x40000000, 0x00000000, 0x00000000],
        ], dtype=jnp.uint32)

        split_keys = split(rng_key, 5)

        for split_key in split_keys:
            self.assertTrue(is_state_invalidated(split_key))

    def test_split_reproduces_invalid_states(self) -> None:
        rng_key = jnp.array([
            [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574],
            [0xbbbbbbbb, 0x00000000, 0x00000000, 0x00000000],
            [0x00000000, 0x00000000, 0x00000000, 0xaaaaaaaa],
            [0xbe57823a, 0x00000000, 0x00000000, 0x00000000],
        ], dtype=jnp.uint32)
        assert is_state_invalidated(rng_key)

        split_keys = split(rng_key, 2)

        for split_key in split_keys:
            self.assertTrue(is_state_invalidated(split_key))

    def test_split_rejects_too_large_split(self) -> None:
        rng_key = PRNGKey(23456)
        with self.assertRaises(ValueError):
            split(rng_key, 2**32)

    def test_PRNGKey_from_int(self) -> None:
        x = 9651354789628635673475235

        expected_rng_key = jnp.array([
            [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574],
            [0x00000000, 0x00000000, 0x00000000, 0x00000000],
            [0x00000000, 0xc0fb0700, 0x0412d4fd, 0xa338f3bf],
            [0x00000000, 0x00000000, 0x00000000, 0x00000001]
        ], dtype=jnp.uint32)

        rng_key = PRNGKey(x)
        self.assertTrue(np.all(expected_rng_key == rng_key))

    def test_PRNGKey_from_bytes(self) -> None:
        x = bytes(range(30))

        expected_rng_key = jnp.array([
            [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574],
            [0x03020100, 0x07060504, 0x0b0a0908, 0xf0e0d0c],
            [0x13121110, 0x17161514, 0x1b1a1918, 0x00001d1c],
            [0x00000000, 0x00000000, 0x00000000, 0x00000001]
        ], dtype=jnp.uint32)

        rng_key = PRNGKey(x)

        self.assertTrue(np.all(expected_rng_key == rng_key))

    def test_PRNGKey_from_bytes_rejects_too_long(self) -> None:
        x = bytes(range(33))

        with self.assertRaises(ValueError):
            PRNGKey(x)

    def test_PRNGKey_invalid_seed_type(self) -> None:
        x = 8.0

        with self.assertRaises(TypeError):
            PRNGKey(x)  # type: ignore # ignore mypy error for invalid type here

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

    def test_uniform_scalar(self) -> None:
        rng_key = PRNGKey(0)
        sample = uniform(rng_key)
        self.assertEqual((), jnp.shape(sample))

    def test_uniform_vmap(self) -> None:
        rng_key = PRNGKey(0)
        batch_keys = split(rng_key, 7)
        shape = (3, 12)
        minval = 1
        maxval = 3
        dtype = jnp.float32
        data = jax.vmap(lambda key: uniform(key, shape, dtype, minval, maxval))(batch_keys)

        self.assertEqual(jnp.float32, data.dtype)
        self.assertEqual((7, *shape), data.shape)
        self.assertTrue(np.all(data >= minval))
        self.assertTrue(np.all(data <= maxval))

    def test_uniform_scalar_vmap(self) -> None:
        rng_key = PRNGKey(0)
        batch_keys = split(rng_key, 7)
        sample = jax.vmap(uniform)(batch_keys)
        self.assertEqual((7,), jnp.shape(sample))

    def test_random_bits_consistency(self) -> None:
        """ verifies that there is no difference between sampling two blocks directly
            or separately (while manually increasing counter) """
        rng_key = PRNGKey(0)
        single_vals, _ = random_bits(rng_key, 32, (32,))
        multi_vals_1, _ = random_bits(rng_key, 32, (16,))
        rng_key_2 = set_counter(rng_key, 1)
        multi_vals_2, _ = random_bits(rng_key_2, 32, (16,))
        self.assertTrue(np.all(single_vals[:16] == multi_vals_1))
        self.assertTrue(np.all(single_vals[16:] == multi_vals_2))

    def test_random_bits_scalar(self) -> None:
        rng_key = PRNGKey(0)
        sample, error = random_bits(rng_key, 32, ())
        self.assertEqual(0, error)
        self.assertEqual((), jnp.shape(sample))
        self.assertEqual(jnp.uint32, jnp.dtype(sample))

    def test_random_bits_invalid_width(self) -> None:
        rng_key = PRNGKey(0)
        with self.assertRaises(ValueError):
            random_bits(rng_key, 13, (1,))

    def test_uniform_invalid_dtype(self) -> None:
        rng_key = PRNGKey(0)

        with self.assertRaises(TypeError):
            _uniform(rng_key, (), jnp.uint8, 0., 1.)
        with self.assertRaises(TypeError):
            uniform(rng_key, (), jnp.uint32)

    def test_random_bits_invalid_state(self) -> None:
        rng_key = jnp.array([
            [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574],
            [0xbbbbbbbb, 0x00000000, 0x00000000, 0x00000000],
            [0x00000000, 0x00000000, 0x00000000, 0xaaaaaaaa],
            [0xbe57823a, 0x00000000, 0x00000000, 0x00000000],
        ], dtype=jnp.uint32)

        x, error = random_bits(rng_key, 32, (1, 5))
        self.assertTrue(jnp.all(x == 0))
        self.assertEqual(ErrorFlag.InvalidState, error)

    def test_random_bits_counter_overflow(self) -> None:
        rng_key = jnp.array([
            [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574],
            [0xbbbbbbbb, 0x00000000, 0x00000000, 0x00000000],
            [0x00000000, 0x00000000, 0x00000000, 0xaaaaaaaa],
            [0xfffffffe, 0x00000000, 0x00000000, 0x00000001],
        ], dtype=jnp.uint32)
        # state has one block left to generate

        x, error = random_bits(rng_key, 32, (4, 5))  # will require generating two blocks
        self.assertTrue(jnp.all(x == 0))
        self.assertEqual(ErrorFlag.CounterOverflow, error)

    def test_uniform_invalid_state(self) -> None:
        rng_key = jnp.array([
            [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574],
            [0xbbbbbbbb, 0x00000000, 0x00000000, 0x00000000],
            [0x00000000, 0x00000000, 0x00000000, 0xaaaaaaaa],
            [0xbe57823a, 0x00000000, 0x00000000, 0x00000000],
        ], dtype=jnp.uint32)

        x = uniform(rng_key, (3, 2, 1))
        self.assertTrue(jnp.all(jnp.isnan(x)))

    def test_uniform_counter_overflow(self) -> None:
        rng_key = jnp.array([
            [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574],
            [0xbbbbbbbb, 0x00000000, 0x00000000, 0x00000000],
            [0x00000000, 0x00000000, 0x00000000, 0xaaaaaaaa],
            [0xffffffff, 0x00000000, 0x00000000, 0x00000001],
        ], dtype=jnp.uint32)
        # state has zero blocks left to generate

        x = uniform(rng_key, (3, 2, 1))
        self.assertTrue(jnp.all(jnp.isnan(x)))


if __name__ == '__main__':
    unittest.main()
