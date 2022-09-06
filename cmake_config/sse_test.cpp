// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2022 Aalto University

#include <array>
#include <cstdint>
#include <immintrin.h>

int main(int argc, const char** argv)
{
    std::array<uint32_t, 4> x { 1, 2, 3, 4 };
    std::array<uint32_t, 4> y { 5, 6, 7, 8 };

    __m128i x_vec = _mm_load_si128(reinterpret_cast<const __m128i*>(x.data()));
    __m128i y_vec = _mm_load_si128(reinterpret_cast<const __m128i*>(y.data()));

    __m128i z_vec = _mm_add_epi32(x_vec, y_vec);

    std::array<uint32_t, 4> z;
    _mm_store_si128(reinterpret_cast<__m128i*>(z.data()), z_vec);

    for (int i = 0; i < 4; ++i)
    {
        if (z[i] != x[i] + y[i])
            return 1;
    }
    return 0;
}
