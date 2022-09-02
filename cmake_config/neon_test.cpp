// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2022 Aalto University

#include <array>
#include <cstdint>
#include <arm_neon.h>

int main(int argc, const char** argv)
{
    std::array<uint32_t, 4> x { 1, 2, 3, 4 };
    std::array<uint32_t, 4> y { 5, 6, 7, 8 };

    uint32x4_t x_vec = vld1q_u32(x.data());
    uint32x4_t y_vec = vld1q_u32(y.data());

    uint32x4_t z_vec = vaddq_u32(x_vec, y_vec);

    std::array<uint32_t, 4> z;
    vst1q_u32(z.data(), z_vec);

    for (int i = 0; i < 4; ++i)
    {
        if (z[i] != x[i] + y[i])
            return 1;
    }
    return 0;
}
