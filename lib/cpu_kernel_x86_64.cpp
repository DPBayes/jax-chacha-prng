// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2022 Aalto University

#include "cpu_kernel_x86_64.hpp"

inline __m128i rotate_left(__m128i values, uint num_bits)
{
    return _mm_xor_si128(
        _mm_slli_epi32(values, num_bits),
        _mm_srli_epi32(values, 32 - num_bits)
    ); // (value << num_bits) ^ (value >> (32 - num_bits));
}

VectorizedState quarterround_sse(VectorizedState state)
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
    d = _mm_xor_si128(d, a); // d ^= a;
    d = rotate_left(d, 8);
    c = _mm_add_epi32(c, d); // c += d;
    b = _mm_xor_si128(b, c); // b ^= c;
    b = rotate_left(b, 7);
    return VectorizedState(a, b, c, d);
}

void pack_diagonals(VectorizedState& out_state, VectorizedState in_state)
{
    out_state[0] = rotate_elements_left<0>(in_state[0]);
    out_state[1] = rotate_elements_left<1>(in_state[1]);
    out_state[2] = rotate_elements_left<2>(in_state[2]);
    out_state[3] = rotate_elements_left<3>(in_state[3]);
}

void unpack_diagonals(VectorizedState& out_state, VectorizedState in_state)
{
    out_state[0] = rotate_elements_right<0>(in_state[0]);
    out_state[1] = rotate_elements_right<1>(in_state[1]);
    out_state[2] = rotate_elements_right<2>(in_state[2]);
    out_state[3] = rotate_elements_right<3>(in_state[3]);
}

VectorizedState double_round_sse(VectorizedState state)
{
    state = quarterround_sse(state);
    pack_diagonals(state, state);
    state = quarterround_sse(state);
    unpack_diagonals(state, state);

    return state;
}

VectorizedState add_states_sse(VectorizedState x, VectorizedState y)
{
    VectorizedState out;
    for (uint i = 0; i < 4; ++i)
    {
        out[i] = _mm_add_epi32(x[i], y[i]);
    }
    return out;
}

VectorizedState vectorize_state(const uint32_t state[16])
{
    VectorizedState vec_state;
    for (uint i = 0; i < 4; ++i)
    {
        vec_state[i] = _mm_load_si128(&(reinterpret_cast<const __m128i*>(state)[i]));
    }
    return vec_state;
}

void unvectorize_state(uint32_t out_state[16], VectorizedState vec_state)
{
    for (uint i = 0; i < 4; ++i)
    {
        _mm_store_si128(&(reinterpret_cast<__m128i*>(out_state)[i]), vec_state[i]);
    }
}

void chacha20_block_sse(uint32_t out_state[16], const uint32_t in_state[16])
{
    VectorizedState vec_in_state = vectorize_state(in_state);
    VectorizedState vec_tmp_state = double_round_sse(vec_in_state);
    for (uint i = 0; i < ChaChaDoubleRoundCount - 1; ++i)
    {
        vec_tmp_state = double_round_sse(vec_tmp_state);
    }
    vec_tmp_state = add_states_sse(vec_in_state, vec_tmp_state);
    unvectorize_state(out_state, vec_tmp_state);
}

void cpu_chacha20_block(void* out_buffer, const void** in_buffers)
{
    uint32_t num_states = *reinterpret_cast<const uint32_t*>(in_buffers[0]);
    const uint32_t* in_states = reinterpret_cast<const uint32_t*>(in_buffers[1]);
    uint32_t* out_state = reinterpret_cast<uint32_t*>(out_buffer);
    #pragma omp parallel for
    for (uint32_t i = 0; i < num_states; ++i)
    {
        uint32_t offset = ChaChaStateSizeInWords * i;
        chacha20_block_sse(out_state + offset, in_states + offset);
    }
}
