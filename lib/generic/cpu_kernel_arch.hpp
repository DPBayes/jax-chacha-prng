// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2022 Aalto University

#pragma once

#include "../defs.hpp"
#include <stdint.h>
#include <algorithm>

struct StateRow
{
    uint32_t values[ChaChaStateWordsPerRow];

    StateRow() : values() { }
    StateRow(const uint32_t row_values[ChaChaStateWordsPerRow]) : StateRow()
    {
        std::copy_n(row_values, ChaChaStateWordsPerRow, values);
    }

    inline StateRow& operator+=(const StateRow other)
    {
        for (uint i = 0; i < ChaChaStateWordsPerRow; ++i)
        {
            values[i] += other.values[i];
        }
        return *this;
    }

    inline StateRow& operator^=(const StateRow other)
    {
        for (uint i = 0; i < ChaChaStateWordsPerRow; ++i)
        {
            values[i] ^= other.values[i];
        }
        return *this;
    }

    inline StateRow& operator<<=(int num_bits)
    {
        for (uint i = 0; i < ChaChaStateWordsPerRow; ++i)
        {
            uint32_t val = values[i];
            values[i] = (val << num_bits) | (val >> (32 - num_bits));
        }
        return *this;
    }

    template <int num_positions>
    inline StateRow rotate_elements_left() const
    {
        StateRow res;
        for (uint i = 0; i < ChaChaStateWordsPerRow; ++i)
        {
            res.values[i] = values[(num_positions + i) % ChaChaStateWordsPerRow];
        }
        return res;
    }

    template <uint num_positions>
    inline StateRow rotate_elements_right() const
    {
        return rotate_elements_left<(ChaChaStateWordsPerRow - num_positions) % ChaChaStateWordsPerRow>();
    }

    inline void unvectorize(uint32_t out_buffer[ChaChaStateWordsPerRow]) const
    {
        std::copy_n(values, ChaChaStateWordsPerRow, out_buffer);
    }

};
