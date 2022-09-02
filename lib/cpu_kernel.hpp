// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2022 Aalto University

#pragma once

#include "defs.hpp"

#include <cpu_kernel_arch.hpp>


inline StateRow operator+(StateRow left, StateRow right)
{
    StateRow result(left);
    result += right;
    return result;
}

inline StateRow operator^(StateRow left, StateRow right)
{
    StateRow result(left);
    result ^= right;
    return result;
}

inline StateRow operator<<(StateRow row, uint num_bits)
{
    StateRow result(row);
    result <<= num_bits;
    return result;
}

template <>
inline StateRow StateRow::rotate_elements_left<0>() const
{
    return *this;
}


struct VectorizedState
{
private:
    StateRow rows[ChaChaStateSizeInRows];

public:
    VectorizedState() = default;
    VectorizedState(StateRow a, StateRow b, StateRow c, StateRow d);
    VectorizedState(const uint32_t state[ChaChaStateSizeInWords]);

    void unvectorize(uint32_t out_state[ChaChaStateSizeInWords]) const;

    StateRow& operator[](uint i);

    VectorizedState& operator+=(VectorizedState other);
    VectorizedState operator+(VectorizedState other) const;
};

void unpack_diagonals(VectorizedState& out_state, VectorizedState in_state);
void pack_diagonals(VectorizedState& out_state, VectorizedState in_state);
void chacha20_block(uint32_t out_state[16], const uint32_t in_state[16]);
void cpu_chacha20_block(void* out_buffer, const void** in_buffers);
