import unittest
import jax.numpy as jnp
import numpy as np
np.set_printoptions(formatter={'int':hex})

from chacha.cipher import *
from chacha.cipher import _block, _quarterround

class ChaCha20CipherTests(unittest.TestCase):

    def test_chacha_quarterround(self) -> None:
        x = jnp.array([
            0x11111111, 0x01020304, 0x9b8d6f43, 0x01234567
        ], dtype=jnp.uint32)

        expected = jnp.array([
            0xea2a92f4, 0xcb1cf8ce, 0x4581472e, 0x5881c4bb
        ], dtype=jnp.uint32)

        y = _quarterround(x)
        self.assertTrue(jnp.all(expected == y), "_quarterround function does not produce correct output on test vector")


    def test_chacha_block(self) -> None:
        ctx = jnp.array([
            [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574],
            [0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c],
            [0x13121110, 0x17161514, 0x1b1a1918, 0x1f1e1d1c],
            [0x00000001, 0x09000000, 0x4a000000, 0x00000000],
        ], dtype=jnp.uint32)

        expected = jnp.array([
            [0xe4e7f110, 0x15593bd1, 0x1fdd0f50, 0xc47120a3],
            [0xc7f4d1c7, 0x0368c033, 0x9aaa2204, 0x4e6cd4c3],
            [0x466482d2, 0x09aa9f07, 0x05d7c214, 0xa2028bd9],
            [0xd19c12b5, 0xb94e16de, 0xe883d0cb, 0x4e3c50a2],
        ], dtype=jnp.uint32)

        y = _block(ctx)
        self.assertTrue(jnp.all(y == expected), "_block function does produce correct output on test vector")


    def test_setup_state(self) -> None:
        key = b'\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f'
        iv = b'\x00\x00\x00\x09\x00\x00\x00\x4a\x00\x00\x00\x00'
        counter = jnp.uint32(1)
        ctx = setup_state(key, iv, counter)

        expected = jnp.array([
            [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574],
            [0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c],
            [0x13121110, 0x17161514, 0x1b1a1918, 0x1f1e1d1c],
            [0x00000001, 0x09000000, 0x4a000000, 0x00000000],
        ], dtype=jnp.uint32)

        self.assertTrue(jnp.all(expected == ctx), "setup_state function does not produce correct output on test vector")

    def test_increment_counter(self) -> None:
        ctx = jnp.array([
            [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574],
            [0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c],
            [0x13121110, 0x17161514, 0x1b1a1918, 0x1f1e1d1c],
            [0x00000001, 0x09000000, 0x4a000000, 0x00000000],
        ], dtype=jnp.uint32)

        expected = jnp.array([
            [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574],
            [0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c],
            [0x13121110, 0x17161514, 0x1b1a1918, 0x1f1e1d1c],
            [0x00000002, 0x09000000, 0x4a000000, 0x00000000],
        ], dtype=jnp.uint32)

        y = increment_counter(ctx)
        self.assertTrue(jnp.all(expected == y))


    def test_get_counter(self) -> None:
        ctx = jnp.array([
            [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574],
            [0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c],
            [0x13121110, 0x17161514, 0x1b1a1918, 0x1f1e1d1c],
            [0x00000001, 0x09000000, 0x4a000000, 0x00000000],
        ], dtype=jnp.uint32)

        ctr = get_counter(ctx)
        self.assertEqual(1, ctr)

    def test_serialize(self) -> None:
        ctx = jnp.array([
            [0xe4e7f110, 0x15593bd1, 0x1fdd0f50, 0xc47120a3],
            [0xc7f4d1c7, 0x0368c033, 0x9aaa2204, 0x4e6cd4c3],
            [0x466482d2, 0x09aa9f07, 0x05d7c214, 0xa2028bd9],
            [0xd19c12b5, 0xb94e16de, 0xe883d0cb, 0x4e3c50a2],
        ], dtype=jnp.uint32)

        expected = b'\x10\xf1\xe7\xe4\xd1\x3b\x59\x15\x50\x0f\xdd\x1f\xa3\x20\x71\xc4\xc7\xd1\xf4\xc7\x33\xc0\x68\x03\x04\x22\xaa\x9a\xc3\xd4\x6c\x4e\xd2\x82\x64\x46\x07\x9f\xaa\x09\x14\xc2\xd7\x05\xd9\x8b\x02\xa2\xb5\x12\x9c\xd1\xde\x16\x4e\xb9\xcb\xd0\x83\xe8\xa2\x50\x3c\x4e'
        expected = np.frombuffer(expected, dtype=np.uint8)

        y = serialize(ctx)

        self.assertTrue(np.all(expected == y), "chacha_serialize function does not produce correct output on test vector")

    def test_encrypt_with_key(self) -> None:
        key = b'\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f'
        iv = b'\x00\x00\x00\x00\x00\x00\x00\x4a\x00\x00\x00\x00'
        counter = jnp.uint32(1)

        message = "Ladies and Gentlemen of the class of '99: If I could offer you only one tip for the future, sunscreen would be it.".encode("ascii")
        expected = b'\x6e\x2e\x35\x9a\x25\x68\xf9\x80\x41\xba\x07\x28\xdd\x0d\x69\x81\xe9\x7e\x7a\xec\x1d\x43\x60\xc2\x0a\x27\xaf\xcc\xfd\x9f\xae\x0b\xf9\x1b\x65\xc5\x52\x47\x33\xab\x8f\x59\x3d\xab\xcd\x62\xb3\x57\x16\x39\xd6\x24\xe6\x51\x52\xab\x8f\x53\x0c\x35\x9f\x08\x61\xd8\x07\xca\x0d\xbf\x50\x0d\x6a\x61\x56\xa3\x8e\x08\x8a\x22\xb6\x5e\x52\xbc\x51\x4d\x16\xcc\xf8\x06\x81\x8c\xe9\x1a\xb7\x79\x37\x36\x5a\xf9\x0b\xbf\x74\xa3\x5b\xe6\xb4\x0b\x8e\xed\xf2\x78\x5e\x42\x87\x4d'

        y = encrypt_with_key(message, key, iv, counter)


        self.assertEqual(len(expected), len(y), "encrypt_with_key function produces ciphertext of wrong length")
        self.assertEqual(expected, y, "encrypt_with_key function does not produce correct output on test vector")


    def test_set_nonce(self) -> None:
        ctx = jnp.array([
            [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574],
            [0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c],
            [0x13121110, 0x17161514, 0x1b1a1918, 0x1f1e1d1c],
            [0x00000001, 0x09000000, 0x4a000000, 0x00000000],
        ], dtype=jnp.uint32)
        nonce = jnp.array([0xffffffff, 0xeeeeeeee, 0xdddddddd], dtype=jnp.uint32)

        new_ctx = set_nonce(ctx, nonce)

        expected_ctx = jnp.array([
            [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574],
            [0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c],
            [0x13121110, 0x17161514, 0x1b1a1918, 0x1f1e1d1c],
            [0x00000001, 0xffffffff, 0xeeeeeeee, 0xdddddddd],
        ], dtype=jnp.uint32)

        self.assertTrue(np.all(expected_ctx == new_ctx))

if __name__ == '__main__':
    unittest.main()
