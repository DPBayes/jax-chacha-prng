// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2022 Aalto University

#include "cpu_kernel_arm.hpp"

inline uint32x4_t rotate_left(uint32x4_t values, uint num_bits)
{
    return vorrq_u32(
        vshlq_n_u32(values, num_bits),
        vshrq_n_u32(values, 32 - num_bits)
    );
}

VectorizedState quarterround(VectorizedState state)
{
    uint32x4_t a = state[0];
    uint32x4_t b = state[1];
    uint32x4_t c = state[2];
    uint32x4_t d = state[3];
    a = vaddq_u32(a, b); // a += b;
    d = veorq_u32(d, a); // d ^= a;
    d = rotate_left(d, 16);
    c = vaddq_u32(c, d); // c += d;
    b = veorq_u32(b, c); // b ^= c;
    b = rotate_left(b, 12);
    a = vaddq_u32(a, b); // a += b;
    d = veorq_u32(d, a); // d ^= a;
    d = rotate_left(d, 8);
    c = vaddq_u32(c, d); // c += d;
    b = veorq_u32(b, c); // b ^= c;
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

VectorizedState double_round(VectorizedState state)
{
    state = quarterround(state);
    pack_diagonals(state, state);
    state = quarterround(state);
    unpack_diagonals(state, state);

    return state;
}

VectorizedState add_states(VectorizedState x, VectorizedState y)
{
    VectorizedState out;
    for (uint i = 0; i < 4; ++i)
    {
        out[i] = vaddq_u32(x[i], y[i]);
    }
    return out;
}

VectorizedState vectorize_state(const uint32_t state[16])
{
    return VectorizedState(vld4q_u32(state));
}

void unvectorize_state(uint32_t out_state[16], VectorizedState vec_state)
{
    vst4q_u32(out_state, vec_state.values);
}

void chacha20_block(uint32_t out_state[16], const uint32_t in_state[16])
{
    VectorizedState vec_in_state = vectorize_state(in_state);
    VectorizedState vec_tmp_state = double_round(vec_in_state);
    for (uint i = 0; i < ChaChaDoubleRoundCount - 1; ++i)
    {
        vec_tmp_state = double_round(vec_tmp_state);
    }
    vec_tmp_state = add_states(vec_in_state, vec_tmp_state);
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
        chacha20_block(out_state + offset, in_states + offset);
    }
}
