// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2022 Aalto University

#pragma once

#include <stdint.h>

#include "defs.hpp"
#include <arm_neon.h>

struct VectorizedState
{
    uint32x4x4_t values;

    VectorizedState() = default;
    VectorizedState(uint32x4_t a, uint32x4_t b, uint32x4_t c, uint32x4_t d) : values()
    {
        values.val[0] = a;
        values.val[1] = b;
        values.val[2] = c;
        values.val[3] = d;
    }

    VectorizedState(uint32x4x4_t vals) : values(vals) { }
    uint32x4_t& operator[](uint i) { return values.val[i]; }
};

uint32x4_t rotate_left(uint32x4_t values, uint num_bits);

VectorizedState quarterround(VectorizedState state);

// Rotate elements in a 4-vec
template <uint num_positions>
inline uint32x4_t rotate_elements_left(uint32x4_t vec)
{
    return vextq_u32(vec, vec, (4 - num_positions) % 4);
}

template <uint num_positions>
inline uint32x4_t rotate_elements_right(uint32x4_t vec)
{
    return rotate_elements_left<(4 - num_positions) % 4>(vec);
}

void pack_diagonals(VectorizedState& out_state, VectorizedState in_state);

void unpack_diagonals(VectorizedState& out_state, VectorizedState in_state);

VectorizedState double_round(VectorizedState state);

VectorizedState add_states(VectorizedState x, VectorizedState y);

VectorizedState vectorize_state(const uint32_t state[16]);

void unvectorize_state(uint32_t out_state[16], VectorizedState vec_state);

void chacha20_block(uint32_t out_state[16], const uint32_t in_state[16]);

void cpu_chacha20_block(void* out_buffer, const void** in_buffers);
