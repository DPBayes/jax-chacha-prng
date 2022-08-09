# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2021,2022 Aalto University

import unittest
import testconfig  # noqa
import numpy as np  # type: ignore
import jax
import jax.numpy as jnp

from chacha.cipher import \
    CounterOverflowException, setup_state, encrypt, decrypt, encrypt_with_key,\
    decrypt_with_key, set_nonce, get_nonce, get_counter, increment_counter, serialize
from chacha.cipher import _block
from chacha.defs import ChaChaState, ChaChaStateElementType, ChaChaStateShape


class ChaChaStateTests(unittest.TestCase):

    def test_construction(self) -> None:
        data = jnp.zeros(ChaChaStateShape, dtype=ChaChaStateElementType)
        state = ChaChaState(data)
        self.assertTrue(jnp.all(data == state))

    def test_invalid_shape(self) -> None:
        data = jnp.array([1, 2, 3, 4, 5], dtype=ChaChaStateElementType)
        with self.assertRaises(ValueError):
            ChaChaState(data)

    def test_invalid_dtype(self) -> None:
        data = jnp.zeros(ChaChaStateShape, dtype=jnp.uint16)
        with self.assertRaises(TypeError):
            ChaChaState(data)


class ChaCha20CipherTests(unittest.TestCase):

    def test_chacha_block(self) -> None:
        """ Test vector 2.3.2 from RFC 7539. """
        state = jnp.array([
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

        y = _block(state)
        self.assertTrue(jnp.all(y == expected), "_block function does give correct output on test vector")

    def test_serialize(self) -> None:
        """ According to test vector 2.1.1 from RFC 7539. """
        state = jnp.array([
            [0xe4e7f110, 0x15593bd1, 0x1fdd0f50, 0xc47120a3],
            [0xc7f4d1c7, 0x0368c033, 0x9aaa2204, 0x4e6cd4c3],
            [0x466482d2, 0x09aa9f07, 0x05d7c214, 0xa2028bd9],
            [0xd19c12b5, 0xb94e16de, 0xe883d0cb, 0x4e3c50a2],
        ], dtype=jnp.uint32)

        expected_raw = \
            b'\x10\xf1\xe7\xe4\xd1\x3b\x59\x15\x50\x0f\xdd\x1f\xa3\x20\x71\xc4\xc7\xd1\xf4\xc7\x33\xc0\x68\x03\x04'\
            b'\x22\xaa\x9a\xc3\xd4\x6c\x4e\xd2\x82\x64\x46\x07\x9f\xaa\x09\x14\xc2\xd7\x05\xd9\x8b\x02\xa2\xb5\x12'\
            b'\x9c\xd1\xde\x16\x4e\xb9\xcb\xd0\x83\xe8\xa2\x50\x3c\x4e'
        expected = np.frombuffer(expected_raw, dtype=np.uint8)

        y = serialize(state)

        self.assertTrue(np.all(expected == y), "chacha_serialize function does not give correct output on test vector")

    def test_setup_state(self) -> None:
        """ According to 2.4.2 of RFC 7539. """
        key = b'\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18'\
              b'\x19\x1a\x1b\x1c\x1d\x1e\x1f'
        iv = b'\x00\x00\x00\x00\x00\x00\x00\x4a\x00\x00\x00\x00'
        counter = jnp.uint32(1)
        state = setup_state(key, iv, counter)

        expected = jnp.array([
            [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574],
            [0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c],
            [0x13121110, 0x17161514, 0x1b1a1918, 0x1f1e1d1c],
            [0x00000001, 0x00000000, 0x4a000000, 0x00000000],
        ], dtype=jnp.uint32)

        self.assertTrue(jnp.all(expected == state), "setup_state function does not give correct output on test vector")

    def test_setup_state_128bit_key(self) -> None:
        key = b'\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f'
        iv = b'\x00\x00\x00\x00\x00\x00\x00\x4a\x00\x00\x00\x00'
        counter = jnp.uint32(1)
        state = setup_state(key, iv, counter)

        expected = jnp.array([
            [0x61707865, 0x3120646e, 0x79622d36, 0x6b206574],
            [0x13121110, 0x17161514, 0x1b1a1918, 0x1f1e1d1c],
            [0x13121110, 0x17161514, 0x1b1a1918, 0x1f1e1d1c],
            [0x00000001, 0x00000000, 0x4a000000, 0x00000000],
        ], dtype=jnp.uint32)

        self.assertTrue(jnp.all(expected == state), "setup_state function does not give correct output for 128 bit key")

    def test_setup_state_invalid_key_length_bytes(self) -> None:
        key = b'\x00\x00\x00'
        iv = b'\x00\x00\x00\x00\x00\x00\x00\x4a\x00\x00\x00\x00'
        counter = 1
        with self.assertRaises(ValueError):
            setup_state(key, iv, counter)

    def test_setup_state_invalid_key_length_array(self) -> None:
        key = jnp.array([0, 1, 2], dtype=jnp.uint32)
        iv = b'\x00\x00\x00\x00\x00\x00\x00\x4a\x00\x00\x00\x00'
        counter = 1
        with self.assertRaises(ValueError):
            setup_state(key, iv, counter)

    def test_setup_state_invalid_key_dtype_array(self) -> None:
        key = jnp.array([0, 1, 2, 3], dtype=jnp.float32)
        iv = b'\x00\x00\x00\x00\x00\x00\x00\x4a\x00\x00\x00\x00'
        counter = 1
        with self.assertRaises(ValueError):
            setup_state(key, iv, counter)

    def test_setup_state_invalid_iv_length_bytes(self) -> None:
        key = b'\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18'\
              b'\x19\x1a\x1b\x1c\x1d\x1e\x1f'
        iv = b'\x00\x00\x00\x00\x00\x00\x00\x4a\x00'
        counter = jnp.uint32(1)

        with self.assertRaises(ValueError):
            setup_state(key, iv, counter)

    def test_setup_state_invalid_iv_length_array(self) -> None:
        key = b'\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18'\
              b'\x19\x1a\x1b\x1c\x1d\x1e\x1f'
        iv = jnp.array([0, 1], dtype=jnp.uint32)
        counter = jnp.uint32(1)

        with self.assertRaises(ValueError):
            setup_state(key, iv, counter)

    def test_setup_state_invalid_iv_dtype_array(self) -> None:
        key = b'\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18'\
              b'\x19\x1a\x1b\x1c\x1d\x1e\x1f'
        iv = jnp.array([0, 1, 2], dtype=jnp.float32)
        counter = jnp.uint32(1)

        with self.assertRaises(ValueError):
            setup_state(key, iv, counter)

    def test_setup_state_invalid_counter_type(self) -> None:
        key = b'\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18'\
              b'\x19\x1a\x1b\x1c\x1d\x1e\x1f'
        iv = b'\x00\x00\x00\x00\x00\x00\x00\x4a\x00\x00\x00\x00'
        counter = 1.0

        with self.assertRaises(ValueError):
            setup_state(key, iv, counter)  # type: ignore # ignore mypy typre error for float here

    def test_setup_state_invalid_counter_dtype_array(self) -> None:
        key = b'\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18'\
              b'\x19\x1a\x1b\x1c\x1d\x1e\x1f'
        iv = b'\x00\x00\x00\x00\x00\x00\x00\x4a\x00\x00\x00\x00'
        counter = jnp.array([1.0])

        with self.assertRaises(ValueError):
            setup_state(key, iv, counter)

    def test_setup_state_invalid_counter_length_array(self) -> None:
        key = b'\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18'\
              b'\x19\x1a\x1b\x1c\x1d\x1e\x1f'
        iv = b'\x00\x00\x00\x00\x00\x00\x00\x4a\x00\x00\x00\x00'
        counter = jnp.array([1, 2], dtype=jnp.uint32)

        with self.assertRaises(ValueError):
            setup_state(key, iv, counter)

    def test_increment_counter(self) -> None:
        state = jnp.array([
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

        y = increment_counter(state)
        self.assertTrue(jnp.all(expected == y))

    def test_get_counter(self) -> None:
        state = jnp.array([
            [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574],
            [0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c],
            [0x13121110, 0x17161514, 0x1b1a1918, 0x1f1e1d1c],
            [0x00000001, 0x09000000, 0x4a000000, 0x00000000],
        ], dtype=jnp.uint32)

        ctr = get_counter(state)
        self.assertEqual(1, ctr)

    def test_set_get_nonce(self) -> None:
        state = jnp.array([
            [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574],
            [0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c],
            [0x13121110, 0x17161514, 0x1b1a1918, 0x1f1e1d1c],
            [0x00000001, 0x09000000, 0x4a000000, 0x00000000],
        ], dtype=jnp.uint32)
        nonce = jnp.array([0xffffffff, 0xeeeeeeee, 0xdddddddd], dtype=jnp.uint32)

        new_state = set_nonce(state, nonce)

        expected_state = jnp.array([
            [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574],
            [0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c],
            [0x13121110, 0x17161514, 0x1b1a1918, 0x1f1e1d1c],
            [0x00000001, 0xffffffff, 0xeeeeeeee, 0xdddddddd],
        ], dtype=jnp.uint32)

        self.assertTrue(np.all(expected_state == new_state))

        gotten_nonce = get_nonce(new_state)
        self.assertTrue(np.all(nonce == gotten_nonce))

    def test_encrypt(self) -> None:
        """ Test vector 2.4.2 from RFC 7539. """
        state = jnp.array([
            [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574],
            [0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c],
            [0x13121110, 0x17161514, 0x1b1a1918, 0x1f1e1d1c],
            [0x00000001, 0x00000000, 0x4a000000, 0x00000000],
        ], dtype=jnp.uint32)

        expected_state = jnp.array([
            [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574],
            [0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c],
            [0x13121110, 0x17161514, 0x1b1a1918, 0x1f1e1d1c],
            [0x00000003, 0x00000000, 0x4a000000, 0x00000000],
        ], dtype=jnp.uint32)

        message = "Ladies and Gentlemen of the class of '99: If I could offer you only one tip for the future, "\
                  "sunscreen would be it.".encode("ascii")
        expected = \
            b'\x6e\x2e\x35\x9a\x25\x68\xf9\x80\x41\xba\x07\x28\xdd\x0d\x69\x81\xe9\x7e\x7a\xec\x1d\x43\x60\xc2\x0a\x27'\
            b'\xaf\xcc\xfd\x9f\xae\x0b\xf9\x1b\x65\xc5\x52\x47\x33\xab\x8f\x59\x3d\xab\xcd\x62\xb3\x57\x16\x39\xd6\x24'\
            b'\xe6\x51\x52\xab\x8f\x53\x0c\x35\x9f\x08\x61\xd8\x07\xca\x0d\xbf\x50\x0d\x6a\x61\x56\xa3\x8e\x08\x8a\x22'\
            b'\xb6\x5e\x52\xbc\x51\x4d\x16\xcc\xf8\x06\x81\x8c\xe9\x1a\xb7\x79\x37\x36\x5a\xf9\x0b\xbf\x74\xa3\x5b\xe6'\
            b'\xb4\x0b\x8e\xed\xf2\x78\x5e\x42\x87\x4d'

        ciphertext, new_state = encrypt(message, state)

        self.assertTrue(np.all(expected_state == new_state))
        self.assertEqual(len(expected), len(ciphertext), "encrypt function gives ciphertext of wrong length")
        self.assertEqual(expected, ciphertext, "encrypt function does not give correct output on test vector")

    def test_decrypt(self) -> None:
        state = jnp.array([
            [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574],
            [0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c],
            [0x13121110, 0x17161514, 0x1b1a1918, 0x1f1e1d1c],
            [0x00000001, 0x00000000, 0x4a000000, 0x00000000],
        ], dtype=jnp.uint32)

        expected_state = jnp.array([
            [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574],
            [0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c],
            [0x13121110, 0x17161514, 0x1b1a1918, 0x1f1e1d1c],
            [0x00000003, 0x00000000, 0x4a000000, 0x00000000],
        ], dtype=jnp.uint32)

        message = "Ladies and Gentlemen of the class of '99: If I could offer you only one tip for the future, "\
            "sunscreen would be it.".encode("ascii")
        ciphertext = \
            b'\x6e\x2e\x35\x9a\x25\x68\xf9\x80\x41\xba\x07\x28\xdd\x0d\x69\x81\xe9\x7e\x7a\xec\x1d\x43\x60\xc2\x0a\x27'\
            b'\xaf\xcc\xfd\x9f\xae\x0b\xf9\x1b\x65\xc5\x52\x47\x33\xab\x8f\x59\x3d\xab\xcd\x62\xb3\x57\x16\x39\xd6\x24'\
            b'\xe6\x51\x52\xab\x8f\x53\x0c\x35\x9f\x08\x61\xd8\x07\xca\x0d\xbf\x50\x0d\x6a\x61\x56\xa3\x8e\x08\x8a\x22'\
            b'\xb6\x5e\x52\xbc\x51\x4d\x16\xcc\xf8\x06\x81\x8c\xe9\x1a\xb7\x79\x37\x36\x5a\xf9\x0b\xbf\x74\xa3\x5b\xe6'\
            b'\xb4\x0b\x8e\xed\xf2\x78\x5e\x42\x87\x4d'

        recovered_message, new_state = decrypt(ciphertext, state)

        self.assertTrue(np.all(expected_state == new_state))
        self.assertEqual(message, recovered_message)

    def test_encrypt_with_key(self) -> None:
        """ Test vector 2.4.2 from RFC 7539. """
        key = b'\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18'\
              b'\x19\x1a\x1b\x1c\x1d\x1e\x1f'
        iv = b'\x00\x00\x00\x00\x00\x00\x00\x4a\x00\x00\x00\x00'
        counter = 1
        expected_counter = 3

        message = "Ladies and Gentlemen of the class of '99: If I could offer you only one tip for the future, "\
                  "sunscreen would be it.".encode("ascii")
        expected = \
            b'\x6e\x2e\x35\x9a\x25\x68\xf9\x80\x41\xba\x07\x28\xdd\x0d\x69\x81\xe9\x7e\x7a\xec\x1d\x43\x60\xc2\x0a\x27'\
            b'\xaf\xcc\xfd\x9f\xae\x0b\xf9\x1b\x65\xc5\x52\x47\x33\xab\x8f\x59\x3d\xab\xcd\x62\xb3\x57\x16\x39\xd6\x24'\
            b'\xe6\x51\x52\xab\x8f\x53\x0c\x35\x9f\x08\x61\xd8\x07\xca\x0d\xbf\x50\x0d\x6a\x61\x56\xa3\x8e\x08\x8a\x22'\
            b'\xb6\x5e\x52\xbc\x51\x4d\x16\xcc\xf8\x06\x81\x8c\xe9\x1a\xb7\x79\x37\x36\x5a\xf9\x0b\xbf\x74\xa3\x5b\xe6'\
            b'\xb4\x0b\x8e\xed\xf2\x78\x5e\x42\x87\x4d'

        ciphertext, new_counter = encrypt_with_key(message, key, iv, counter)

        self.assertEqual(expected_counter, new_counter)
        self.assertEqual(len(expected), len(ciphertext), "encrypt_with_key function gives ciphertext of wrong length")
        self.assertEqual(expected, ciphertext, "encrypt_with_key function does not give correct output on test vector")

    def test_decrypt_with_key(self) -> None:
        key = b'\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18'\
              b'\x19\x1a\x1b\x1c\x1d\x1e\x1f'
        iv = b'\x00\x00\x00\x00\x00\x00\x00\x4a\x00\x00\x00\x00'
        counter = 1
        expected_counter = 3

        message = "Ladies and Gentlemen of the class of '99: If I could offer you only one tip for the future, "\
            "sunscreen would be it.".encode("ascii")
        ciphertext = \
            b'\x6e\x2e\x35\x9a\x25\x68\xf9\x80\x41\xba\x07\x28\xdd\x0d\x69\x81\xe9\x7e\x7a\xec\x1d\x43\x60\xc2\x0a\x27'\
            b'\xaf\xcc\xfd\x9f\xae\x0b\xf9\x1b\x65\xc5\x52\x47\x33\xab\x8f\x59\x3d\xab\xcd\x62\xb3\x57\x16\x39\xd6\x24'\
            b'\xe6\x51\x52\xab\x8f\x53\x0c\x35\x9f\x08\x61\xd8\x07\xca\x0d\xbf\x50\x0d\x6a\x61\x56\xa3\x8e\x08\x8a\x22'\
            b'\xb6\x5e\x52\xbc\x51\x4d\x16\xcc\xf8\x06\x81\x8c\xe9\x1a\xb7\x79\x37\x36\x5a\xf9\x0b\xbf\x74\xa3\x5b\xe6'\
            b'\xb4\x0b\x8e\xed\xf2\x78\x5e\x42\x87\x4d'

        recovered_message, new_counter = decrypt_with_key(ciphertext, key, iv, counter)

        self.assertEqual(expected_counter, new_counter)
        self.assertEqual(message, recovered_message)

    def test_encrypt_counter_overflow(self) -> None:
        state = jnp.array([
            [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574],
            [0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c],
            [0x13121110, 0x17161514, 0x1b1a1918, 0x1f1e1d1c],
            [0xfffffffe, 0x00000000, 0x4a000000, 0x00000000],
        ], dtype=jnp.uint32)


        message = "Ladies and Gentlemen of the class of '99: If I could offer you only one tip for the future, "\
                  "sunscreen would be it.".encode("ascii")
        with self.assertRaises(CounterOverflowException):
            encrypt(message, state)

class ChaCha20CipherAdditionalVectorTests(unittest.TestCase):

    def test_chacha_block_additional_vector_1(self) -> None:
        """ Test Vector 1, from RFC 7539 A.1. """
        key = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'\
              b'\x00\x00\x00\x00\x00\x00\x00'
        iv = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
        counter = 0

        state = setup_state(key, iv, counter)

        expected = jnp.array([
            [0xade0b876, 0x903df1a0, 0xe56a5d40, 0x28bd8653],
            [0xb819d2bd, 0x1aed8da0, 0xccef36a8, 0xc70d778b],
            [0x7c5941da, 0x8d485751, 0x3fe02477, 0x374ad8b8],
            [0xf4b8436a, 0x1ca11815, 0x69b687c3, 0x8665eeb2],
        ], dtype=jnp.uint32)

        y = _block(state)
        self.assertTrue(jnp.all(y == expected), "_block function does give correct output on test vector")

    def test_chacha_block_additional_vector_2(self) -> None:
        """ Test Vector 2, from RFC 7539 A.1. """
        key = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'\
              b'\x00\x00\x00\x00\x00\x00\x00'
        iv = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
        counter = 1

        state = setup_state(key, iv, counter)

        expected = jnp.array([
            [0xbee7079f, 0x7a385155, 0x7c97ba98, 0x0d082d73],
            [0xa0290fcb, 0x6965e348, 0x3e53c612, 0xed7aee32],
            [0x7621b729, 0x434ee69c, 0xb03371d5, 0xd539d874],
            [0x281fed31, 0x45fb0a51, 0x1f0ae1ac, 0x6f4d794b],
        ], dtype=jnp.uint32)

        y = _block(state)
        self.assertTrue(jnp.all(y == expected), "_block function does give correct output on test vector")

    def test_chacha_block_additional_vector_3(self) -> None:
        """ Test Vector 3, from RFC 7539 A.1. """
        key = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'\
              b'\x00\x00\x00\x00\x00\x00\x01'
        iv = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
        counter = 1

        state = setup_state(key, iv, counter)

        expected = jnp.array([
            [0x2452eb3a, 0x9249f8ec, 0x8d829d9b, 0xddd4ceb1],
            [0xe8252083, 0x60818b01, 0xf38422b8, 0x5aaa49c9],
            [0xbb00ca8e, 0xda3ba7b4, 0xc4b592d1, 0xfdf2732f],
            [0x4436274e, 0x2561b3c8, 0xebdd4aa6, 0xa0136c00],
        ], dtype=jnp.uint32)

        y = _block(state)
        self.assertTrue(jnp.all(y == expected), "_block function does give correct output on test vector")

    def test_chacha_block_additional_vector_4(self) -> None:
        """ Test Vector 4, from RFC 7539 A.1. """
        key = b'\x00\xff\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'\
              b'\x00\x00\x00\x00\x00\x00\x00'
        iv = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
        counter = 2

        state = setup_state(key, iv, counter)

        expected = jnp.array([
            [0xfb4dd572, 0x4bc42ef1, 0xdf922636, 0x327f1394],
            [0xa78dea8f, 0x5e269039, 0xa1bebbc1, 0xcaf09aae],
            [0xa25ab213, 0x48a6b46c, 0x1b9d9bcb, 0x092c5be6],
            [0x546ca624, 0x1bec45d5, 0x87f47473, 0x96f0992e],
        ], dtype=jnp.uint32)

        y = _block(state)
        self.assertTrue(jnp.all(y == expected), "_block function does give correct output on test vector")

    def test_chacha_block_additional_vector_5(self) -> None:
        """ Test Vector 5, from RFC 7539 A.1. """
        key = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'\
              b'\x00\x00\x00\x00\x00\x00\x00'
        iv = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02'
        counter = 0

        state = setup_state(key, iv, counter)

        expected = jnp.array([
            [0x374dc6c2, 0x3736d58c, 0xb904e24a, 0xcd3f93ef],
            [0x88228b1a, 0x96a4dfb3, 0x5b76ab72, 0xc727ee54],
            [0x0e0e978a, 0xf3145c95, 0x1b748ea8, 0xf786c297],
            [0x99c28f5f, 0x628314e8, 0x398a19fa, 0x6ded1b53],
        ], dtype=jnp.uint32)

        y = _block(state)
        self.assertTrue(jnp.all(y == expected), "_block function does give correct output on test vector")

    def test_encrypt_with_key_additional_vector_1(self) -> None:
        """ Test Vector 1, from RFC 7539 A.2. """
        key = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'\
              b'\x00\x00\x00\x00\x00\x00\x00'
        iv = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
        counter = 0

        message_blocksize = 1
        message = b'\x00' * 64
        expected = \
            b'\x76\xb8\xe0\xad\xa0\xf1\x3d\x90\x40\x5d\x6a\xe5\x53\x86\xbd\x28\xbd\xd2\x19\xb8\xa0\x8d\xed\x1a\xa8\x36'\
            b'\xef\xcc\x8b\x77\x0d\xc7\xda\x41\x59\x7c\x51\x57\x48\x8d\x77\x24\xe0\x3f\xb8\xd8\x4a\x37\x6a\x43\xb8\xf4'\
            b'\x15\x18\xa1\x1c\xc3\x87\xb6\x69\xb2\xee\x65\x86'

        ciphertext, new_counter = encrypt_with_key(message, key, iv, counter)
        self.assertEqual(counter + message_blocksize, new_counter)
        self.assertEqual(expected, ciphertext)

    def test_encrypt_with_key_additional_vector_2(self) -> None:
        """ Test Vector 2, from RFC 7539 A.2. """
        key = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'\
              b'\x00\x00\x00\x00\x00\x00\x01'
        iv = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02'
        counter = 1

        message_blocksize = 6
        message =  \
            'Any submission to the IETF intended by the Contributor for publication as all or part of an IETF Internet'\
            '-Draft or RFC and any statement made within the context of an IETF activity is considered an "IETF Contri'\
            'bution". Such statements include oral statements in IETF sessions, as well as written and electronic comm'\
            'unications made at any time or place, which are addressed to'.encode("ascii")
        expected = \
            b'\xa3\xfb\xf0\x7d\xf3\xfa\x2f\xde\x4f\x37\x6c\xa2\x3e\x82\x73\x70\x41\x60\x5d\x9f\x4f\x4f\x57\xbd\x8c\xff'\
            b'\x2c\x1d\x4b\x79\x55\xec\x2a\x97\x94\x8b\xd3\x72\x29\x15\xc8\xf3\xd3\x37\xf7\xd3\x70\x05\x0e\x9e\x96\xd6'\
            b'\x47\xb7\xc3\x9f\x56\xe0\x31\xca\x5e\xb6\x25\x0d\x40\x42\xe0\x27\x85\xec\xec\xfa\x4b\x4b\xb5\xe8\xea\xd0'\
            b'\x44\x0e\x20\xb6\xe8\xdb\x09\xd8\x81\xa7\xc6\x13\x2f\x42\x0e\x52\x79\x50\x42\xbd\xfa\x77\x73\xd8\xa9\x05'\
            b'\x14\x47\xb3\x29\x1c\xe1\x41\x1c\x68\x04\x65\x55\x2a\xa6\xc4\x05\xb7\x76\x4d\x5e\x87\xbe\xa8\x5a\xd0\x0f'\
            b'\x84\x49\xed\x8f\x72\xd0\xd6\x62\xab\x05\x26\x91\xca\x66\x42\x4b\xc8\x6d\x2d\xf8\x0e\xa4\x1f\x43\xab\xf9'\
            b'\x37\xd3\x25\x9d\xc4\xb2\xd0\xdf\xb4\x8a\x6c\x91\x39\xdd\xd7\xf7\x69\x66\xe9\x28\xe6\x35\x55\x3b\xa7\x6c'\
            b'\x5c\x87\x9d\x7b\x35\xd4\x9e\xb2\xe6\x2b\x08\x71\xcd\xac\x63\x89\x39\xe2\x5e\x8a\x1e\x0e\xf9\xd5\x28\x0f'\
            b'\xa8\xca\x32\x8b\x35\x1c\x3c\x76\x59\x89\xcb\xcf\x3d\xaa\x8b\x6c\xcc\x3a\xaf\x9f\x39\x79\xc9\x2b\x37\x20'\
            b'\xfc\x88\xdc\x95\xed\x84\xa1\xbe\x05\x9c\x64\x99\xb9\xfd\xa2\x36\xe7\xe8\x18\xb0\x4b\x0b\xc3\x9c\x1e\x87'\
            b'\x6b\x19\x3b\xfe\x55\x69\x75\x3f\x88\x12\x8c\xc0\x8a\xaa\x9b\x63\xd1\xa1\x6f\x80\xef\x25\x54\xd7\x18\x9c'\
            b'\x41\x1f\x58\x69\xca\x52\xc5\xb8\x3f\xa3\x6f\xf2\x16\xb9\xc1\xd3\x00\x62\xbe\xbc\xfd\x2d\xc5\xbc\xe0\x91'\
            b'\x19\x34\xfd\xa7\x9a\x86\xf6\xe6\x98\xce\xd7\x59\xc3\xff\x9b\x64\x77\x33\x8f\x3d\xa4\xf9\xcd\x85\x14\xea'\
            b'\x99\x82\xcc\xaf\xb3\x41\xb2\x38\x4d\xd9\x02\xf3\xd1\xab\x7a\xc6\x1d\xd2\x9c\x6f\x21\xba\x5b\x86\x2f\x37'\
            b'\x30\xe3\x7c\xfd\xc4\xfd\x80\x6c\x22\xf2\x21'

        ciphertext, new_counter = encrypt_with_key(message, key, iv, counter)
        self.assertEqual(counter + message_blocksize, new_counter)
        self.assertEqual(expected, ciphertext)

    def test_encrypt_with_key_additional_vector_3(self) -> None:
        """ Test Vector 3, from RFC 7539 A.2. """
        key = b'\x1c\x92\x40\xa5\xeb\x55\xd3\x8a\xf3\x33\x88\x86\x04\xf6\xb5\xf0\x47\x39\x17\xc1\x40\x2b\x80\x09\x9d'\
              b'\xca\x5c\xbc\x20\x70\x75\xc0'
        iv = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02'
        counter = 42

        message_blocksize = 2
        message = \
            "'Twas brillig, and the slithy toves\nDid gyre and gimble in the wabe:\nAll mimsy were the borogoves,\n"\
            "And the mome raths outgrabe.".encode("ascii")
        expected = \
            b'\x62\xe6\x34\x7f\x95\xed\x87\xa4\x5f\xfa\xe7\x42\x6f\x27\xa1\xdf\x5f\xb6\x91\x10\x04\x4c\x0d\x73\x11\x8e'\
            b'\xff\xa9\x5b\x01\xe5\xcf\x16\x6d\x3d\xf2\xd7\x21\xca\xf9\xb2\x1e\x5f\xb1\x4c\x61\x68\x71\xfd\x84\xc5\x4f'\
            b'\x9d\x65\xb2\x83\x19\x6c\x7f\xe4\xf6\x05\x53\xeb\xf3\x9c\x64\x02\xc4\x22\x34\xe3\x2a\x35\x6b\x3e\x76\x43'\
            b'\x12\xa6\x1a\x55\x32\x05\x57\x16\xea\xd6\x96\x25\x68\xf8\x7d\x3f\x3f\x77\x04\xc6\xa8\xd1\xbc\xd1\xbf\x4d'\
            b'\x50\xd6\x15\x4b\x6d\xa7\x31\xb1\x87\xb5\x8d\xfd\x72\x8a\xfa\x36\x75\x7a\x79\x7a\xc1\x88\xd1'

        ciphertext, new_counter = encrypt_with_key(message, key, iv, counter)
        self.assertEqual(counter + message_blocksize, new_counter)
        self.assertEqual(expected, ciphertext)


class ChaCha20CipherVectorizationTest(unittest.TestCase):

    def test_block_accepts_only_single_state(self) -> None:
        key = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'\
            b'\x00\x00\x00\x00\x00\x00\x00'
        iv = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'

        states = jnp.array([setup_state(key, iv, counter) for counter in range(7)])

        with self.assertRaises(ValueError):
            _block(states)

    def test_block_exception_for_vmap_when_input_is_single_state(self) -> None:
        key = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'\
            b'\x00\x00\x00\x00\x00\x00\x00'
        iv = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'

        state = setup_state(key, iv, 0)
        with self.assertRaises(ValueError):
            jax.vmap(_block)(state)

    def test_block_singly_vectorized(self) -> None:
        key = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'\
            b'\x00\x00\x00\x00\x00\x00\x00'
        iv = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'

        states = jnp.array([setup_state(key, iv, counter) for counter in range(7)])
        expected = [
            jnp.array([
                [0xade0b876, 0x903df1a0, 0xe56a5d40, 0x28bd8653],
                [0xb819d2bd, 0x1aed8da0, 0xccef36a8, 0xc70d778b],
                [0x7c5941da, 0x8d485751, 0x3fe02477, 0x374ad8b8],
                [0xf4b8436a, 0x1ca11815, 0x69b687c3, 0x8665eeb2],
            ], dtype=jnp.uint32),
            jnp.array([
                [0xbee7079f, 0x7a385155, 0x7c97ba98, 0x0d082d73],
                [0xa0290fcb, 0x6965e348, 0x3e53c612, 0xed7aee32],
                [0x7621b729, 0x434ee69c, 0xb03371d5, 0xd539d874],
                [0x281fed31, 0x45fb0a51, 0x1f0ae1ac, 0x6f4d794b],
            ], dtype=jnp.uint32)
        ]

        results = jax.vmap(_block)(states)
        self.assertEqual(results.shape, (7, *ChaChaStateShape))
        self.assertTrue(jnp.all(expected[0] == results[0]))
        self.assertTrue(jnp.all(expected[1] == results[1]))

    def test_block_double_batchdim_fails_for_single_vmap(self) -> None:
        key = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'\
            b'\x00\x00\x00\x00\x00\x00\x00'
        iv = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'

        states = jnp.array([setup_state(key, iv, counter) for counter in range(7)])
        states = jnp.array((states, states))

        with self.assertRaises(ValueError):
            jax.vmap(_block)(states)

    def test_block_doubly_vectorized(self) -> None:
        key = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'\
            b'\x00\x00\x00\x00\x00\x00\x00'
        iv = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'

        states = jnp.array([setup_state(key, iv, counter) for counter in range(7)])
        states = jnp.array((states, states))
        expected = [
            jnp.array([
                [0xade0b876, 0x903df1a0, 0xe56a5d40, 0x28bd8653],
                [0xb819d2bd, 0x1aed8da0, 0xccef36a8, 0xc70d778b],
                [0x7c5941da, 0x8d485751, 0x3fe02477, 0x374ad8b8],
                [0xf4b8436a, 0x1ca11815, 0x69b687c3, 0x8665eeb2],
            ], dtype=jnp.uint32),
            jnp.array([
                [0xbee7079f, 0x7a385155, 0x7c97ba98, 0x0d082d73],
                [0xa0290fcb, 0x6965e348, 0x3e53c612, 0xed7aee32],
                [0x7621b729, 0x434ee69c, 0xb03371d5, 0xd539d874],
                [0x281fed31, 0x45fb0a51, 0x1f0ae1ac, 0x6f4d794b],
            ], dtype=jnp.uint32)
        ]

        results = jax.vmap(jax.vmap(_block))(states)
        self.assertEqual(results.shape, (2, 7, *ChaChaStateShape))
        self.assertTrue(jnp.all(expected[0] == results[0][0]))
        self.assertTrue(jnp.all(expected[1] == results[0][1]))
        self.assertTrue(jnp.all(results[0] == results[1]))

    def test_block_doubly_vectorized_out_of_order(self) -> None:
        key = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'\
            b'\x00\x00\x00\x00\x00\x00\x00'
        iv = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'

        states = jnp.array([[setup_state(key, iv, counter + i * 7) for counter in range(7)] for i in range(5)])

        baseline = jnp.array([
            [
                _block(jnp.array(state))  # casting to jnp.array explicity to prevent failure with old jax versions
                for state in i_states
            ] for i_states in states
        ])

        assert states.shape == (5, 7, 4, 4)
        expected = [
            jnp.array([
                [0xade0b876, 0x903df1a0, 0xe56a5d40, 0x28bd8653],
                [0xb819d2bd, 0x1aed8da0, 0xccef36a8, 0xc70d778b],
                [0x7c5941da, 0x8d485751, 0x3fe02477, 0x374ad8b8],
                [0xf4b8436a, 0x1ca11815, 0x69b687c3, 0x8665eeb2],
            ], dtype=jnp.uint32),
            jnp.array([
                [0xbee7079f, 0x7a385155, 0x7c97ba98, 0x0d082d73],
                [0xa0290fcb, 0x6965e348, 0x3e53c612, 0xed7aee32],
                [0x7621b729, 0x434ee69c, 0xb03371d5, 0xd539d874],
                [0x281fed31, 0x45fb0a51, 0x1f0ae1ac, 0x6f4d794b],
            ], dtype=jnp.uint32)
        ]

        results = jax.vmap(jax.vmap(_block), in_axes=1)(states)
        self.assertEqual(results.shape, (7, 5, *ChaChaStateShape))
        self.assertTrue(jnp.all(expected[0] == results[0][0]))
        self.assertTrue(jnp.all(expected[1] == results[1][0]))

        for c in range(7):
            for i in range(5):
                self.assertTrue(jnp.all(results[c, i] == baseline[i, c]), f"Wrong output for i={i}, c={c}")

    def test_block_triply_vectorized_out_of_order(self) -> None:
        key = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'\
            b'\x00\x00\x00\x00\x00\x00\x00'
        iv = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'

        states = jnp.array([
            [
                [
                    setup_state(key, iv, counter + i * 7 + j * 7 * 2)
                    for counter in range(7)
                ] for i in range(2)
            ] for j in range(3)
        ])

        baseline = jnp.array([
            [
                [
                    _block(jnp.array(state))  # casting to jnp.array explicity to prevent failure with old jax versions
                    for state in ij_states
                ] for ij_states in j_states
            ] for j_states in states
        ])

        assert states.shape == (3, 2, 7, 4, 4)
        expected = [
            jnp.array([
                [0xade0b876, 0x903df1a0, 0xe56a5d40, 0x28bd8653],
                [0xb819d2bd, 0x1aed8da0, 0xccef36a8, 0xc70d778b],
                [0x7c5941da, 0x8d485751, 0x3fe02477, 0x374ad8b8],
                [0xf4b8436a, 0x1ca11815, 0x69b687c3, 0x8665eeb2],
            ], dtype=jnp.uint32),
            jnp.array([
                [0xbee7079f, 0x7a385155, 0x7c97ba98, 0x0d082d73],
                [0xa0290fcb, 0x6965e348, 0x3e53c612, 0xed7aee32],
                [0x7621b729, 0x434ee69c, 0xb03371d5, 0xd539d874],
                [0x281fed31, 0x45fb0a51, 0x1f0ae1ac, 0x6f4d794b],
            ], dtype=jnp.uint32)
        ]

        results = jax.vmap(jax.vmap(jax.vmap(_block), in_axes=1), in_axes=1)(states)
        self.assertEqual(results.shape, (2, 7, 3, *ChaChaStateShape))
        self.assertTrue(jnp.all(expected[0] == results[0][0][0]))
        self.assertTrue(jnp.all(expected[1] == results[0][1][0]))

        for i in range(2):
            for c in range(7):
                for j in range(3):
                    self.assertTrue(
                        jnp.all(results[i, c, j] == baseline[j, i, c]),
                        f"Wrong output for i={i}, j={j}, c={c}"
                    )


if __name__ == '__main__':
    unittest.main()
