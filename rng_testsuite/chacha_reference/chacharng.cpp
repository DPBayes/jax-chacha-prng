// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2021 Aalto University

extern "C" {
#include "ecrypt-sync.h"
}
#include "chacharng.hpp"

#include <algorithm>
#include <vector>
#include <iostream>


#define BITS_PER_BYTE 8
unsigned long chacha_GenBits(void* param, void* state)
{
    constexpr int BYTES_IN_VALUE = sizeof(bitsequence_t);

    auto rngState = static_cast<ChachaRNGState*>(state);
    if (rngState->bitBufferCursor >= rngState->valuesPerBatch)
    {
        size_t keystreamLengthInBytes = rngState->valuesPerBatch * BYTES_IN_VALUE;
        std::vector<uint8_t> keystream(keystreamLengthInBytes);
        ECRYPT_keystream_bytes(&(rngState->ctx), reinterpret_cast<uint8_t*>(rngState->bitBuffer.data()), keystreamLengthInBytes);
        rngState->bitBufferCursor = 0;
    }
    return rngState->bitBuffer[rngState->bitBufferCursor++];
}

#define unif01_INV32   2.328306436538696289e-10
double chacha_GenU01(void* param, void* state)
{
    return unif01_INV32 * chacha_GenBits(param, state);
}

void chacha_Write(void* state)
{
    std::cout << "C-Chacha with state" << std::endl;
    std::cout << "counter = ";
    auto rngState = static_cast<ChachaRNGState*>(state);
    printf("%x", rngState->ctx.input[12]);
    std::cout << std::endl;
}

unif01_Gen chacha_CreateRNG(const std::vector<uint8_t>& seed, size_t valuesPerBatch)
{
    ECRYPT_init();

    ECRYPT_ctx ctx;
    ECRYPT_keysetup(&ctx, seed.data(), KEY_LENGTH_IN_BYTES*8, 64);
    ChachaRNGState* rngState = new ChachaRNGState(std::move(ctx), valuesPerBatch);

    unif01_Gen gen;
    gen.state = rngState;
    gen.param = nullptr;
    gen.name = const_cast<char*>("C-ChaCha20 with state");
    gen.GetBits = chacha_GenBits;
    gen.GetU01 = chacha_GenU01;
    gen.Write = chacha_Write;
    return gen;
}

void chacha_ClearRNG(unif01_Gen& gen)
{
    delete static_cast<ChachaRNGState*>(gen.state);
}
