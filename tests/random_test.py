import unittest
import jax.numpy as jnp
import numpy as np
np.set_printoptions(formatter={'int':hex})

from chacha.random import *
from chacha.random import _split
from chacha.cipher import set_counter

class ChaChaRNGTests(unittest.TestCase):

    def test_random_bits(self) -> None:
        ctx = jnp.array([
            [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574],
            [0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c],
            [0x13121110, 0x17161514, 0x1b1a1918, 0x1f1e1d1c],
            [0x00000001, 0x09000000, 0x4a000000, 0x00000000],
        ], dtype=jnp.uint32)

        shape = (17, 9)
        x = random_bits(ctx, 8, shape)
        self.assertEqual(x.dtype, jnp.uint8)
        self.assertEqual(x.shape, shape)

        x = random_bits(ctx, 16, shape)
        self.assertEqual(x.dtype, jnp.uint16)
        self.assertEqual(x.shape, shape)

        x = random_bits(ctx, 32, shape)
        self.assertEqual(x.dtype, jnp.uint32)
        self.assertEqual(x.shape, shape)

        # needs 64bit jax mode
        # x = random_bits(ctx, 64, shape)
        # self.assertEqual(x.dtype, jnp.uint64)
        # self.assertEqual(x.shape, shape)


    def test_split(self) -> None:
        ctx = jnp.array([
            [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574],
            [0x00000000, 0x00000000, 0x00000000, 0x00000000],
            [0x00000000, 0x00000000, 0x00000000, 0x00000000],
            [0x00000000, 0x00000000, 0x00000000, 0x00000000],
        ], dtype=jnp.uint32)

        first_ctx, second_ctx, third_ctx, fourth_ctx, fifth_ctx, sixth_ctx = _split(ctx, 6)
            # 6 splits require two blocks of randomness

        expected_first = jnp.array([
            [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574],
            [0x00000000, 0x00000000, 0x00000000, 0x00000000],
            [0x00000000, 0x00000000, 0x00000000, 0x00000000],
            [0x00000000, 0xade0b876, 0x903df1a0, 0xe56a5d40]
        ], dtype=jnp.uint32)
        self.assertTrue(np.all(expected_first == first_ctx))

        expected_second = jnp.array([
            [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574],
            [0x00000000, 0x00000000, 0x00000000, 0x00000000],
            [0x00000000, 0x00000000, 0x00000000, 0x00000000],
            [0x00000000, 0x28bd8653, 0xb819d2bd, 0x1aed8da0]
        ], dtype=jnp.uint32)
        self.assertTrue(np.all(expected_second == second_ctx))

        expected_third = jnp.array([
            [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574],
            [0x00000000, 0x00000000, 0x00000000, 0x00000000],
            [0x00000000, 0x00000000, 0x00000000, 0x00000000],
            [0x00000000, 0xccef36a8, 0xc70d778b, 0x7c5941da]
        ], dtype=jnp.uint32)
        self.assertTrue(np.all(expected_third == third_ctx))

        expected_fourth = jnp.array([
            [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574],
            [0x00000000, 0x00000000, 0x00000000, 0x00000000],
            [0x00000000, 0x00000000, 0x00000000, 0x00000000],
            [0x00000000, 0x8d485751, 0x3fe02477, 0x374ad8b8]
        ], dtype=jnp.uint32)
        self.assertTrue(np.all(expected_fourth == fourth_ctx))

        expected_fifth = jnp.array([
            [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574],
            [0x00000000, 0x00000000, 0x00000000, 0x00000000],
            [0x00000000, 0x00000000, 0x00000000, 0x00000000],
            [0x00000000, 0xf4b8436a, 0x1ca11815, 0x69b687c3]
        ], dtype=jnp.uint32)
        self.assertTrue(np.all(expected_fifth == fifth_ctx))

        expected_sixth = jnp.array([
            [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574],
            [0x00000000, 0x00000000, 0x00000000, 0x00000000],
            [0x00000000, 0x00000000, 0x00000000, 0x00000000],
            [0x00000000, 0x8665eeb2, 0xbee7079f, 0x7a385155]
        ], dtype=jnp.uint32)
        self.assertTrue(np.all(expected_sixth == sixth_ctx))

    def test_PRNGKey(self) -> None:
        x = 9651354789628635673475235
        # x = jnp.ones(8, dtype=jnp.uint32)
        ctx = PRNGKey(x)
        raise NotImplementedError()

    def test_uniform(self) -> None:
        ctx = PRNGKey(0)
        data = uniform(ctx, (3, 12))
        print(data)
        raise NotImplementedError()

    def test_random_bits_consistency(self) -> None:
        """ verifies that there is no difference between sampling two blocks directly
            or separately (while manually increasing counter) """
        ctx = PRNGKey(0)
        single_vals = random_bits(ctx, 32, (32,))
        multi_vals_1 = random_bits(ctx, 32, (16,))
        ctx_2 = set_counter(ctx, 1)
        multi_vals_2 = random_bits(ctx_2, 32, (16,))
        self.assertTrue(np.all(single_vals[:16] == multi_vals_1))
        self.assertTrue(np.all(single_vals[16:] == multi_vals_2))



if __name__ == '__main__':
    unittest.main()
