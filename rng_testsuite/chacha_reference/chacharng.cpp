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

// ECRYPT_ctx ctx;
// const size_t BLOCK_SIZE = 512;
// uint32_t randomnessBuffer[BLOCK_SIZE];
// size_t bufferCursor = BLOCK_SIZE;

// uint32_t chacha_global_GenBits32Bit()
// {
//     if (bufferCursor >= BLOCK_SIZE)
//     {
//         uint8_t keystream[BLOCK_SIZE*4];
//         ECRYPT_keystream_bytes(&ctx, keystream, BLOCK_SIZE*4);
//         for (size_t i = 0; i < BLOCK_SIZE; ++i)
//         {
//             randomnessBuffer[i] =
//                 (static_cast<uint32_t>(keystream[i*4+3]) << 24) |
//                 (static_cast<uint32_t>(keystream[i*4+2]) << 16) |
//                 (static_cast<uint32_t>(keystream[i*4+1]) <<  8) |
//                 (static_cast<uint32_t>(keystream[i*4+0]) <<  0);
//         }
//         bufferCursor = 0;
//     }
//     return randomnessBuffer[bufferCursor++];
// }

// void chacha_global_rnginit()
// {
//     ECRYPT_init();
//     const size_t KEY_LENGTH_IN_BYTES = 32;
//     uint8_t key[KEY_LENGTH_IN_BYTES];
//     std::fill_n(key, KEY_LENGTH_IN_BYTES, 0);
//     ECRYPT_keysetup(&ctx, key, KEY_LENGTH_IN_BYTES*8, 64);
// }
