// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2022 Aalto University

#pragma once

#include "../defs.hpp"
#include <stdint.h>
#include <arm_neon.h>


typedef uint32x4_t vector_t;

constexpr uint ChaChaWordsPerVector = 4;
constexpr uint ChaChaStateSizeInVectors = ChaChaStateSizeInWords / ChaChaWordsPerVector;

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
    VectorizedState(const uint32_t state[ChaChaStateSizeInWords]) : VectorizedState()
    {
        for (uint i = 0; i < ChaChaStateSizeInVectors; ++i)
        {
            values.val[i] = vld1q_u32(state + i * ChaChaStateSizeInVectors);
        }
    }

    void unvectorize(uint32_t out_state[ChaChaStateSizeInWords])
    {
        for (uint i = 0; i < ChaChaStateSizeInVectors; ++i)
        {
            vst1q_u32(out_state + i * ChaChaStateSizeInVectors, values.val[i]);
        }
    }

    uint32x4_t& operator[](uint i) { return values.val[i]; }
};

// Rotate elements in a 4-vec
template <uint num_positions>
inline uint32x4_t rotate_elements_left(uint32x4_t vec)
{
    return vextq_u32(vec, vec, num_positions);
}

template <uint num_positions>
inline uint32x4_t rotate_elements_right(uint32x4_t vec)
{
    return rotate_elements_left<(ChaChaWordsPerVector - num_positions) % ChaChaWordsPerVector>(vec);
}


inline uint32x4_t rotate_left(uint32x4_t values, uint num_bits)
{
    return vorrq_u32(
        vshlq_n_u32(values, num_bits),
        vshrq_n_u32(values, 32 - num_bits)
    );
}

inline uint32x4_t vadd(uint32x4_t x, uint32x4_t y)
{
    return vaddq_u32(x, y);
}

inline uint32x4_t vxor(uint32x4_t x, uint32x4_t y)
{
    return veorq_u32(x, y);
}
