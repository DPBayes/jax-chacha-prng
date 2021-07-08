// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2021 Aalto University

#include <cstddef>
#include <stdint.h>

typedef unsigned int uint;

#include <immintrin.h>


struct VectorizedState
{
    __m128i values[4];
    VectorizedState() = default;
    VectorizedState(__m128i a, __m128i b, __m128i c, __m128i d) : values{a, b, c, d} { }
    __m128i& operator[](uint i) { return values[i]; }
};

__m128i rotate_left(__m128i values, uint num_bits)
{
    return _mm_xor_si128(
        _mm_slli_epi32(values, num_bits),
        _mm_srli_epi32(values, 32 - num_bits)
    ); // (value << num_bits) ^ (value >> (32 - num_bits));
}

static VectorizedState quarterround_sse(VectorizedState state)
{
    __m128i a = state[0];
    __m128i b = state[1];
    __m128i c = state[2];
    __m128i d = state[3];
    a = _mm_add_epi32(a, b); // a += b;
    d = _mm_xor_si128(d, a); // d ^= a;
    d = rotate_left(d, 16);
    c = _mm_add_epi32(c, d); // c += d;
    b = _mm_xor_si128(b, c); // b ^= c;
    b = rotate_left(b, 12);
    a = _mm_add_epi32(a, b); // a += b;
    d = _mm_xor_si128(d, a); //d ^= a;
    d = rotate_left(d, 8);
    c = _mm_add_epi32(c, d); // c += d;
    b = _mm_xor_si128(b, c); // b ^= c;
    b = rotate_left(b, 7);
    return VectorizedState(a, b, c, d);
}

template <uint num_positions>
inline __m128i rotate_elements_left(__m128i vec)
{
    constexpr uint8_t rotation_lookup[4] = {
        0b11100100,
        0b00111001,
        0b01001110,
        0b10010011
    };
    constexpr uint8_t rotation_immediate = rotation_lookup[num_positions];
    return _mm_shuffle_epi32(vec, rotation_immediate);
}

template <uint num_positions>
inline __m128i rotate_elements_right(__m128i vec)
{
    return rotate_elements_left<(4 - num_positions) % 4>(vec);
}

static void pack_diagonals(VectorizedState& out_state, VectorizedState in_state)
{
    out_state[0] = rotate_elements_left<0>(in_state[0]);
    out_state[1] = rotate_elements_left<1>(in_state[1]);
    out_state[2] = rotate_elements_left<2>(in_state[2]);
    out_state[3] = rotate_elements_left<3>(in_state[3]);
}

static void unpack_diagonals(VectorizedState& out_state, VectorizedState in_state)
{
    out_state[0] = rotate_elements_right<0>(in_state[0]);
    out_state[1] = rotate_elements_right<1>(in_state[1]);
    out_state[2] = rotate_elements_right<2>(in_state[2]);
    out_state[3] = rotate_elements_right<3>(in_state[3]);
}

static VectorizedState double_round_sse(VectorizedState state)
{
    state = quarterround_sse(state);
    pack_diagonals(state, state);
    state = quarterround_sse(state);
    unpack_diagonals(state, state);

    return state;
}

static VectorizedState add_states_sse(VectorizedState x, VectorizedState y)
{
    VectorizedState out;
    for (uint i = 0; i < 4; ++i)
    {
        out[i] = _mm_add_epi32(x[i], y[i]);
    }
    return out;
}

static VectorizedState vectorize_state(const uint32_t state[16])
{
    VectorizedState vec_state;
    for (uint i = 0; i < 4; ++i)
    {
        vec_state[i] = _mm_load_si128(&(reinterpret_cast<const __m128i*>(state)[i]));
    }
    return vec_state;
}

static void unvectorize_state(uint32_t out_state[16], VectorizedState vec_state)
{
    for (uint i = 0; i < 4; ++i)
    {
        _mm_store_si128(&(reinterpret_cast<__m128i*>(out_state)[i]), vec_state[i]);
    }
}

static void chacha20_block_sse(uint32_t out_state[16], const uint32_t in_state[16])
{
    VectorizedState vec_in_state = vectorize_state(in_state);
    VectorizedState vec_tmp_state = double_round_sse(vec_in_state);
    for (uint i = 0; i < 9; ++i)
    {
        vec_tmp_state = double_round_sse(vec_tmp_state);
    }
    vec_tmp_state = add_states_sse(vec_in_state, vec_tmp_state);
    unvectorize_state(out_state, vec_tmp_state);
}


// inline uint32_t rotate_left(uint32_t value, uint num_bits)
// {
//     return (value << num_bits) ^ (value >> (32 - num_bits));
// }

// static void quarterround(uint32_t out_state[16], const uint32_t in_state[16], const uint indices[4])
// {
//     uint32_t a = in_state[indices[0]];
//     uint32_t b = in_state[indices[1]];
//     uint32_t c = in_state[indices[2]];
//     uint32_t d = in_state[indices[3]];

//     a += b;
//     d ^= a;
//     d = rotate_left(d, 16);
//     c += d;
//     b ^= c;
//     b = rotate_left(b, 12);
//     a += b;
//     d ^= a;
//     d = rotate_left(d, 8);
//     c += d;
//     b ^= c;
//     b = rotate_left(b, 7);

//     out_state[indices[0]] = a;
//     out_state[indices[1]] = b;
//     out_state[indices[2]] = c;
//     out_state[indices[3]] = d;
// }

// inline uint get_diagonal_index(uint i, uint diagonal)
// {
//     return i * 4 + (i + diagonal) % 4;
// }

// static void double_round(uint32_t out_state[16], const uint32_t in_state[16])
// {
//     for (uint i = 0; i < 4; ++i)
//     {
//         // quarterround on columns
//         uint first_round_indices[4] = { 0 + i, 4 + i, 8 + i, 12 + i };
//         quarterround(out_state, in_state, first_round_indices);

//         // quarterround on diagonals
//         uint second_round_indices[4] = {
//             get_diagonal_index(0, i),
//             get_diagonal_index(1, i),
//             get_diagonal_index(2, i),
//             get_diagonal_index(3, i)
//         };
//         quarterround(out_state, out_state, second_round_indices);
//     }
// }

// static void add_states(uint32_t out[16], const uint32_t x[16], const uint32_t y[16])
// {
//     for (uint i = 0; i < 16; ++i)
//     {
//         out[i] = x[i] + y[i];
//     }
// }

// void chacha20_block(uint32_t out_state[16], const uint32_t in_state[16])
// {
//     double_round(out_state, in_state);
//     for (int i = 0; i < 9; ++i)
//     {
//         double_round(out_state, out_state);
//     }
//     add_states(out_state, in_state, out_state);
// }

void cpu_chacha20_block(void* out_buffer, const void** in_buffers)
{
    const uint32_t* in_state = reinterpret_cast<const uint32_t*>(in_buffers[0]);
    uint32_t* out_state = reinterpret_cast<uint32_t*>(out_buffer);
    chacha20_block_sse(out_state, in_state);
}

// TODO: some ad-hoc test code below, move into separate test file
// #include <cstdio>
// int main(int argc, const char** argv)
// {
//     uint32_t host_state[16] = {
//         0x61707865, 0x3320646e, 0x79622d32, 0x6b206574,
//         0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c,
//         0x13121110, 0x17161514, 0x1b1a1918, 0x1f1e1d1c,
//         0x00000001, 0x09000000, 0x4a000000, 0x00000000,
//     };

//     uint32_t host_result[16];
//     chacha20_block_sse(host_result, host_state);


//     uint32_t host_expected[16] = {
//         0xe4e7f110, 0x15593bd1, 0x1fdd0f50, 0xc47120a3,
//         0xc7f4d1c7, 0x0368c033, 0x9aaa2204, 0x4e6cd4c3,
//         0x466482d2, 0x09aa9f07, 0x05d7c214, 0xa2028bd9,
//         0xd19c12b5, 0xb94e16de, 0xe883d0cb, 0x4e3c50a2,
//     };

//     for (size_t i = 0; i < 16; ++i)
//     {
//         printf("%u, ", host_result[i]);
//         if (i % 4 == 3) printf("\n");
//         if (host_expected[i] != host_result[i])
//         {
//             printf("\nDiffers from expected result %x at position %i", host_expected[i], i);
//             break;
//         }
//     }
//     printf("\n");

// }
