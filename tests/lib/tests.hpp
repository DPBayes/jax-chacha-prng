// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2021 Aalto University

#pragma once

#include <iostream>
#include <array>
#include <vector>

#include "defs.hpp"

inline int test_assert(bool condition)
{
    if (!condition)
    {
        std::cout << "\033[1;31m" << "fail" <<"\033[0m" << std::endl;
        return 1;
    }
    else
    {
        std::cout << "\033[;32m" << "ok" <<"\033[0m" << std::endl;
        return 0;
    }
}


std::vector<std::array<uint32_t, ChaChaStateSizeInWords>> test_vector_states({
    std::array<uint32_t, ChaChaStateSizeInWords>({
        0x61707865, 0x3320646e, 0x79622d32, 0x6b206574,
        0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c,
        0x13121110, 0x17161514, 0x1b1a1918, 0x1f1e1d1c,
        0x00000001, 0x09000000, 0x4a000000, 0x00000000,
    }),
    std::array<uint32_t, ChaChaStateSizeInWords>({
        0x61707865, 0x3320646e, 0x79622d32, 0x6b206574,
        0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000,
    }),
    std::array<uint32_t, ChaChaStateSizeInWords>({
        0x61707865, 0x3320646e, 0x79622d32, 0x6b206574,
        0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000001, 0x00000000, 0x00000000, 0x00000000,
    }),
    std::array<uint32_t, ChaChaStateSizeInWords>({
        0x61707865, 0x3320646e, 0x79622d32, 0x6b206574,
        0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x01000000,
        0x00000001, 0x00000000, 0x00000000, 0x00000000,
    }),
    std::array<uint32_t, ChaChaStateSizeInWords>({
        0x61707865, 0x3320646e, 0x79622d32, 0x6b206574,
        0x0000ff00, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000002, 0x00000000, 0x00000000, 0x00000000,
    }),
    std::array<uint32_t, ChaChaStateSizeInWords>({
        0x61707865, 0x3320646e, 0x79622d32, 0x6b206574,
        0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x02000000,
    }),
});

std::vector<std::array<uint32_t, ChaChaStateSizeInWords>> test_vector_expected({
    std::array<uint32_t, ChaChaStateSizeInWords>({
        0xe4e7f110, 0x15593bd1, 0x1fdd0f50, 0xc47120a3,
        0xc7f4d1c7, 0x0368c033, 0x9aaa2204, 0x4e6cd4c3,
        0x466482d2, 0x09aa9f07, 0x05d7c214, 0xa2028bd9,
        0xd19c12b5, 0xb94e16de, 0xe883d0cb, 0x4e3c50a2,
    }),
    std::array<uint32_t, ChaChaStateSizeInWords>({
        0xade0b876, 0x903df1a0, 0xe56a5d40, 0x28bd8653,
        0xb819d2bd, 0x1aed8da0, 0xccef36a8, 0xc70d778b,
        0x7c5941da, 0x8d485751, 0x3fe02477, 0x374ad8b8,
        0xf4b8436a, 0x1ca11815, 0x69b687c3, 0x8665eeb2,
    }),
    std::array<uint32_t, ChaChaStateSizeInWords>({
        0xbee7079f, 0x7a385155, 0x7c97ba98, 0x0d082d73,
        0xa0290fcb, 0x6965e348, 0x3e53c612, 0xed7aee32,
        0x7621b729, 0x434ee69c, 0xb03371d5, 0xd539d874,
        0x281fed31, 0x45fb0a51, 0x1f0ae1ac, 0x6f4d794b,
    }),
    std::array<uint32_t, ChaChaStateSizeInWords>({
        0x2452eb3a, 0x9249f8ec, 0x8d829d9b, 0xddd4ceb1,
        0xe8252083, 0x60818b01, 0xf38422b8, 0x5aaa49c9,
        0xbb00ca8e, 0xda3ba7b4, 0xc4b592d1, 0xfdf2732f,
        0x4436274e, 0x2561b3c8, 0xebdd4aa6, 0xa0136c00,
    }),
    std::array<uint32_t, ChaChaStateSizeInWords>({
        0xfb4dd572, 0x4bc42ef1, 0xdf922636, 0x327f1394,
        0xa78dea8f, 0x5e269039, 0xa1bebbc1, 0xcaf09aae,
        0xa25ab213, 0x48a6b46c, 0x1b9d9bcb, 0x092c5be6,
        0x546ca624, 0x1bec45d5, 0x87f47473, 0x96f0992e,
    }),
    std::array<uint32_t, ChaChaStateSizeInWords>({
        0x374dc6c2, 0x3736d58c, 0xb904e24a, 0xcd3f93ef,
        0x88228b1a, 0x96a4dfb3, 0x5b76ab72, 0xc727ee54,
        0x0e0e978a, 0xf3145c95, 0x1b748ea8, 0xf786c297,
        0x99c28f5f, 0x628314e8, 0x398a19fa, 0x6ded1b53,
    }),
});
