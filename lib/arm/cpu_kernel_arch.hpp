// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2022 Aalto University

#pragma once

#include "../defs.hpp"
#include <stdint.h>
#include <utility>
#include <arm_neon.h>


template <uint num_positions>
inline uint32x4_t rotate_vector_left(uint32x4_t vec)
{
    return vextq_u32(vec, vec, num_positions);
}

template <>
inline uint32x4_t rotate_vector_left<0>(uint32x4_t vec)
{
    return vec;
}

struct StateRow
{
private:
    uint32x4_t values;

public:
    StateRow() { }
    StateRow(uint32x4_t vec) : values(std::move(vec)) { }
    StateRow(const uint32_t row_values[ChaChaStateWordsPerRow]) : values(vld1q_u32(row_values)) { }

    inline StateRow& operator+=(const StateRow other)
    {
        values = vaddq_u32(values, other.values);
        return *this;
    }

    inline StateRow& operator^=(const StateRow other)
    {
        values = veorq_u32(values, other.values);
        return *this;
    }

    inline StateRow& operator<<=(int num_bits)
    {
        values = vorrq_u32(
            vshlq_n_u32(values, num_bits),
            vshrq_n_u32(values, 32 - num_bits)
        ); // (value << num_bits) ^ (value >> (32 - num_bits));
        return *this;
    }

    template <uint num_positions>
    inline StateRow rotate_elements_left() const
    {
        return StateRow(rotate_vector_left<num_positions>(values));
    }

    template <uint num_positions>
    inline StateRow rotate_elements_right() const
    {
        return rotate_elements_left<(ChaChaStateWordsPerRow - num_positions) % ChaChaStateWordsPerRow>();
    }

    inline void unvectorize(uint32_t out_buffer[ChaChaStateWordsPerRow]) const
    {
        vst1q_u32(out_buffer, values);
    }

};
