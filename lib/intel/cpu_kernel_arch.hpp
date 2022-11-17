// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2022 Aalto University

#pragma once

#include "../defs.hpp"
#include <stdint.h>
#include <immintrin.h>
#include <utility>


// Wrap _mm_shuffle_epi32 in a template to enforce that rotation_immediate
// is always known at compile time.
template <uint8_t rotation_immediate>
inline __m128i _mm_shuffle_epi32_templated(__m128i val)
{
    return _mm_shuffle_epi32(val, rotation_immediate);
}

// Rotate elements in a 4-vec
template <uint num_positions>
inline __m128i rotate_m128i_left(__m128i vec)
{
    constexpr uint8_t rotation_lookup[ChaChaStateWordsPerRow] = {
        0b11100100,
        0b00111001,
        0b01001110,
        0b10010011
    };
    constexpr uint8_t rotation_immediate = rotation_lookup[num_positions];
    // using the templated wrapper for _mm_shuffle_epi32, otherwise
    // gcc may be confused and decide to not treat rotation_immediate as
    // compile-time known in debug mode (-O0) for some reason, resulting
    // in errors from _mm_shuffle_epi32:
    return _mm_shuffle_epi32_templated<rotation_immediate>(vec);
}

template <>
inline __m128i rotate_m128i_left<0>(__m128i vec)
{
    return vec;
}

struct StateRow
{
private:
    __m128i values;

public:
    StateRow() {}
    StateRow(const uint32_t row_values[ChaChaStateWordsPerRow])
        : values(_mm_load_si128(reinterpret_cast<const __m128i*>(row_values))) { }
    StateRow(__m128i vals) : values(std::move(vals)) { }

    inline StateRow& operator+=(const StateRow other)
    {
        values = _mm_add_epi32(values, other.values);
        return *this;
    }

    inline StateRow& operator^=(const StateRow other)
    {
        values = _mm_xor_si128(values, other.values);
        return *this;
    }

    template <uint num_bits>
    inline StateRow rotate_values_left() const
    {
        return StateRow(
            _mm_xor_si128(
                _mm_slli_epi32(values, num_bits),
                _mm_srli_epi32(values, 32 - num_bits)
            ) // (value << num_bits) ^ (value >> (32 - num_bits));
        );
    }

    template <uint num_positions>
    inline StateRow rotate_elements_left() const
    {
        return StateRow(rotate_m128i_left<num_positions>(values));
    }

    template <uint num_positions>
    inline StateRow rotate_elements_right() const
    {
        return rotate_elements_left<(ChaChaStateWordsPerRow - num_positions) % ChaChaStateWordsPerRow>();
    }

    inline void unvectorize(uint32_t out_buffer[ChaChaStateWordsPerRow]) const
    {
        _mm_store_si128(reinterpret_cast<__m128i*>(out_buffer), values);
    }
};
