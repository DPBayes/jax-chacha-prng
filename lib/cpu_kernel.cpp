// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2022 Aalto University

#include "cpu_kernel.hpp"

VectorizedState::VectorizedState(StateRow a, StateRow b, StateRow c, StateRow d) : rows{a, b, c, d} { }

VectorizedState::VectorizedState(const uint32_t state[ChaChaStateSizeInWords]) : rows{
    StateRow(state + 0 * ChaChaStateSizeInRows),
    StateRow(state + 1 * ChaChaStateSizeInRows),
    StateRow(state + 2 * ChaChaStateSizeInRows),
    StateRow(state + 3 * ChaChaStateSizeInRows)
} { }

void VectorizedState::unvectorize(uint32_t out_state[ChaChaStateSizeInWords]) const
{
    for (uint i = 0; i < ChaChaStateSizeInRows; ++i)
    {
        rows[i].unvectorize(out_state + i * ChaChaStateSizeInRows);
    }
}

StateRow& VectorizedState::operator[](uint i)
{
    return rows[i];
}

VectorizedState& VectorizedState::operator+=(VectorizedState other)
{
    for (uint i = 0; i < ChaChaStateSizeInRows; ++i)
    {
        rows[i] += other.rows[i];
    }
    return *this;
}

VectorizedState VectorizedState::operator+(VectorizedState other) const
{
    VectorizedState result(*this);
    result += other;
    return result;
}

/// This implements what Bernstein calls a quarterround, but does so in a
/// vectorized manner, i.e., it performs all quarterrounds over the
/// state matrix's rows concurrently.
VectorizedState round(VectorizedState state)
{
    StateRow a = state[0];
    StateRow b = state[1];
    StateRow c = state[2];
    StateRow d = state[3];
    a += b;
    d ^= a;
    d <<= 16;
    c += d;
    b ^= c;
    b <<= 12;
    a += b;
    d ^= a;
    d <<= 8;
    c += d;
    b ^= c;
    b <<= 7;
    return VectorizedState(a, b, c, d);
}

void pack_diagonals(VectorizedState& out_state, VectorizedState in_state)
{
    out_state[0] = in_state[0].rotate_elements_left<0>();
    out_state[1] = in_state[1].rotate_elements_left<1>();
    out_state[2] = in_state[2].rotate_elements_left<2>();
    out_state[3] = in_state[3].rotate_elements_left<3>();
}

void unpack_diagonals(VectorizedState& out_state, VectorizedState in_state)
{
    out_state[0] = in_state[0].rotate_elements_right<0>();
    out_state[1] = in_state[1].rotate_elements_right<1>();
    out_state[2] = in_state[2].rotate_elements_right<2>();
    out_state[3] = in_state[3].rotate_elements_right<3>();
}

VectorizedState double_round(VectorizedState state)
{
    state = round(state);
    pack_diagonals(state, state);
    state = round(state);
    unpack_diagonals(state, state);

    return state;
}

void chacha20_block(uint32_t out_state[16], const uint32_t in_state[16])
{
    VectorizedState vec_in_state(in_state);
    VectorizedState vec_tmp_state = double_round(vec_in_state);
    for (uint i = 0; i < ChaChaDoubleRoundCount - 1; ++i)
    {
        vec_tmp_state = double_round(vec_tmp_state);
    }
    vec_tmp_state += vec_in_state;
    vec_tmp_state.unvectorize(out_state);
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
