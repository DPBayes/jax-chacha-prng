// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2021 Aalto University

#pragma once

#include <stdint.h>
#include <immintrin.h>

#include "defs.hpp"


struct VectorizedState
{
    __m128i values[4];
    VectorizedState() = default;
    VectorizedState(__m128i a, __m128i b, __m128i c, __m128i d) : values{a, b, c, d} { }
    __m128i& operator[](uint i) { return values[i]; }
};

__m128i rotate_left(__m128i values, uint num_bits);

VectorizedState quarterround_sse(VectorizedState state);

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
    constexpr uint8_t rotation_lookup[4] = {
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

template <uint num_positions>
inline __m128i rotate_elements_right(__m128i vec)
{
    return rotate_elements_left<(4 - num_positions) % 4>(vec);
}

void pack_diagonals(VectorizedState& out_state, VectorizedState in_state);

void unpack_diagonals(VectorizedState& out_state, VectorizedState in_state);

VectorizedState double_round_sse(VectorizedState state);

VectorizedState add_states_sse(VectorizedState x, VectorizedState y);

VectorizedState vectorize_state(const uint32_t state[16]);

void unvectorize_state(uint32_t out_state[16], VectorizedState vec_state);

void chacha20_block_sse(uint32_t out_state[16], const uint32_t in_state[16]);

void cpu_chacha20_block(void* out_buffer, const void** in_buffers);
