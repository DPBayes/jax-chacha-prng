// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2021 Aalto University

// Wrapper for Bernstein's ChaCha reference implementation for TestU01 framework.

#pragma once

extern "C" {
#include "ecrypt-sync.h"
#include <TestU01.h>
}

#include <algorithm>
#include <vector>
#include <iostream>

const size_t BITS_PER_BYTE = 8;
const size_t KEY_LENGTH_IN_BITS = 256;
constexpr size_t KEY_LENGTH_IN_BYTES = KEY_LENGTH_IN_BITS / BITS_PER_BYTE;


typedef uint32_t bitsequence_t;

struct ChachaRNGState
{
    ECRYPT_ctx ctx;
    const size_t valuesPerBatch;
    std::vector<bitsequence_t> bitBuffer;
    size_t bitBufferCursor;
    ChachaRNGState(ECRYPT_ctx ctx, size_t valuesPerBatch)
        : ctx(ctx)
        , valuesPerBatch(valuesPerBatch)
        , bitBuffer(valuesPerBatch)
        , bitBufferCursor(valuesPerBatch)
    {}
};

extern "C" {
unif01_Gen chacha_CreateRNG(const std::vector<uint8_t>& seed, size_t valuesPerBatch);
void chacha_ClearRNG(unif01_Gen& gen);
}