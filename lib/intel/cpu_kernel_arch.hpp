// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2022 Aalto University

#pragma once

#include "../defs.hpp"
#include <stdint.h>
#include <immintrin.h>

typedef __m128i vector_t;

constexpr uint ChaChaWordsPerVector = 4;
constexpr uint ChaChaStateSizeInVectors = ChaChaStateSizeInWords / ChaChaWordsPerVector;


struct VectorizedState
{
    __m128i values[ChaChaStateSizeInVectors];
    VectorizedState() = default;
    VectorizedState(__m128i a, __m128i b, __m128i c, __m128i d) : values{a, b, c, d} { }

    VectorizedState(const uint32_t state[ChaChaStateSizeInWords]) : values()
    {
        for (uint i = 0; i < ChaChaStateSizeInVectors; ++i)
        {
            values[i] = _mm_load_si128(&(reinterpret_cast<const __m128i*>(state)[i]));
        }
    }

    void unvectorize(uint32_t out_state[ChaChaStateSizeInWords])
    {
        for (uint i = 0; i < ChaChaStateSizeInVectors; ++i)
        {
            _mm_store_si128(&(reinterpret_cast<__m128i*>(out_state)[i]), values[i]);
        }
    }

    __m128i& operator[](uint i) { return values[i]; }
};

// Wrap _mm_shuffle_epi32 in a template to enforce that rotation_immediate
// is always known at compile time.
template <uint8_t rotation_immediate>
inline __m128i _mm_shuffle_epi32_templated(__m128i val)
{
    return _mm_shuffle_epi32(val, rotation_immediate);
}

// Rotate elements in a 4-vec
template <uint num_positions>
inline __m128i rotate_elements_left(__m128i vec)
{
    constexpr uint8_t rotation_lookup[ChaChaWordsPerVector] = {
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
inline __m128i rotate_elements_left<0>(__m128i vec)
{
    return vec;
}

template <uint num_positions>
inline __m128i rotate_elements_right(__m128i vec)
{
    return rotate_elements_left<(ChaChaWordsPerVector - num_positions) % ChaChaWordsPerVector>(vec);
}

inline __m128i rotate_left(__m128i values, uint num_bits)
{
    return _mm_xor_si128(
        _mm_slli_epi32(values, num_bits),
        _mm_srli_epi32(values, 32 - num_bits)
    ); // (value << num_bits) ^ (value >> (32 - num_bits));
}

inline vector_t vadd(vector_t x, vector_t y)
{
    return _mm_add_epi32(x, y);
}

inline vector_t vxor(vector_t x, vector_t y)
{
    return _mm_xor_si128(x, y);
}