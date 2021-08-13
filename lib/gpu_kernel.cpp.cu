// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2021 Aalto University

#include <cuda_runtime.h>
#include <cstdlib>
#include <stdint.h>
#include <stdexcept>

#include "defs.hpp"

__device__ __inline__
uint32_t rotate_left(uint32_t value, uint num_bits)
{
    return (value << num_bits) ^ (value >> (32 - num_bits));
}

union uint32_vec4
{
    struct
    {
        uint32_t a;
        uint32_t b;
        uint32_t c;
        uint32_t d;
    } comp;
    uint32_t arr[4];

    __device__ uint32_t& operator[](int i)
    {
        return arr[i];
    }
};

__device__
uint32_vec4 quarterround_with_shuffle(uint32_vec4 vec)
{
    uint32_t a = vec.comp.a;
    uint32_t b = vec.comp.b;
    uint32_t c = vec.comp.c;
    uint32_t d = vec.comp.d;

    a += b;
    d ^= a;
    d = rotate_left(d, 16);
    c += d;
    b ^= c;
    b = rotate_left(b, 12);
    a += b;
    d ^= a;
    d = rotate_left(d, 8);
    c += d;
    b ^= c;
    b = rotate_left(b, 7);

    return (uint32_vec4){ a, b, c, d };
}

__device__
uint32_vec4 double_round_with_shuffle(uint32_vec4 state)
{
    int state_thread_id = threadIdx.x % ThreadsPerState;

    // quarterround on column
    state = quarterround_with_shuffle(state);

    // shuffle so that thread holds diagonal
    for (int i = 1; i < WordsPerThread; ++i)
    {
        state[i] = __shfl_sync((uint)-1,
            state[i], /*srcLane=*/state_thread_id + i, /*width=*/ThreadsPerState
        );
    }

    // quarterround on diagonal
    state = quarterround_with_shuffle(state);

    // shuffle back to columns
    for (int i = 1; i < WordsPerThread; ++i)
    {
        state[i] = __shfl_sync((uint)-1,
            state[i], /*srcLane=*/state_thread_id - i, /*width=*/ThreadsPerState
        );
    }

    return state;
}

__global__
void chacha20_block_with_shuffle(uint32_t* out_state, const uint32_t* in_state, uint num_threads)
{
    // Each block consists of TargetThreadsPerBlock threads and each group of ThreadsPerState threads
    // handle a single state (for a total of StatesPerBlock states in a block).
    // We index into the state buffer by block id and thread group count:
    const uint thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= num_threads) return;

    const uint state_thread_id = threadIdx.x % ThreadsPerState;
    const uint block_state_index = threadIdx.x / ThreadsPerState;
    const uint global_state_index = blockIdx.x * StatesPerBlock + block_state_index;
    const uint global_buffer_offset = global_state_index * ChaChaStateSizeInWords;

    uint32_vec4 in_state_vec;

    for (uint i = 0; i < WordsPerThread; ++i)
    {
        in_state_vec[i] = in_state[global_buffer_offset + i*WordsPerThread + state_thread_id];
    }

    uint32_vec4 state_vec = in_state_vec;

    for (uint i = 0; i < ChaChaDoubleRoundCount; ++i)
    {
        state_vec = double_round_with_shuffle(state_vec);
    }

    for (uint i = 0; i < WordsPerThread; ++i)
    {
        out_state[global_buffer_offset + i*WordsPerThread + state_thread_id] = in_state_vec[i] + state_vec[i];
    }
}

void gpu_chacha20_block(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_length)
{
    uint32_t num_states = 1;
    if (opaque_length > 0)
    {
        if (opaque_length != sizeof(uint32_t))
        {
            throw std::runtime_error(
                "gpu_chacha20_block requires the opaque argument to be either null or a pointer to a 32-bit integer "
                "indicating the number of states on which to operate."
            );
        }
        num_states = *reinterpret_cast<const uint32_t*>(opaque);
    }
    const uint32_t* in_states = reinterpret_cast<const uint32_t*>(buffers[0]);
    uint32_t* out_state = reinterpret_cast<uint32_t*>(buffers[1]);

    uint num_threads = (num_states * ThreadsPerState);
    uint num_blocks =  (num_threads + TargetThreadsPerBlock - 1) / TargetThreadsPerBlock; // = ceil(num_threads / TargetThreadsPerBlock)

    uint threads_per_block = std::min(num_threads, TargetThreadsPerBlock);
    chacha20_block_with_shuffle<<<num_blocks, threads_per_block, 0, stream>>>(out_state, in_states, num_threads);
}

// // TODO: some ad-hoc test code below, move into separate test file
// #include <iostream>
// int main(int argc, const char** argv)
// {
//     uint num_states = 260;
//     uint32_t host_state[ChaChaStateSizeInWords] = {
//         0x61707865, 0x3320646e, 0x79622d32, 0x6b206574,
//         0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c,
//         0x13121110, 0x17161514, 0x1b1a1918, 0x1f1e1d1c,
//         0x00000001, 0x09000000, 0x4a000000, 0x00000000,
//     };
//     // for (size_t i = 0; i < ChaChaStateSizeInWords; ++i)
//     // {
//     //     host_state[i] = i;
//     // }

//     uint32_t* gpu_state;
//     if (cudaMalloc((void**)&gpu_state, num_states*ChaChaStateSizeInBytes) != cudaSuccess)
//     {
//         std::cout << "failed to allocate device memory!" << std::endl;
//         return 1;
//     }
//     for (uint i = 0; i < num_states; ++i)
//     {
//         uint32_t* gpu_state_ptr = gpu_state + i * ChaChaStateSizeInWords;
//         if (cudaMemcpy(gpu_state_ptr, host_state, ChaChaStateSizeInBytes, cudaMemcpyHostToDevice) != cudaSuccess)
//         {
//             std::cout << "failed to copy to device memory!" << std::endl;
//             return 1;
//         }
//     }

//     uint32_t* gpu_state_out;
//     if (cudaMalloc((void**)&gpu_state_out, num_states*ChaChaStateSizeInBytes) != cudaSuccess)
//     {
//         std::cout << "failed to allocate device memory for out buffer!" << std::endl;
//         return 1;
//     }

//     uint32_t* buffers[2] = { gpu_state, gpu_state_out };
//     gpu_chacha20_block(/*stream=*/nullptr, reinterpret_cast<void**>(buffers), /*opaque=*/reinterpret_cast<const char*>(&num_states), /*opaque_length=*/sizeof(num_states));
//     auto err = cudaGetLastError();
//     if (err != cudaSuccess)
//         std::cout << "Error running kernel: " << cudaGetErrorString(err) << std::endl;

//     // chacha20_block<<<1, 4>>>(gpu_state); // could alternatively run 16 threads for faster final summation and idle unused while processing the 4 columns/diagonals

//     std::cout << "states per block: " << StatesPerBlock << " threads per block: " << TargetThreadsPerBlock << std::endl;

//     uint32_t host_result[ChaChaStateSizeInWords];
//     cudaMemcpy(host_result, gpu_state_out + 257 * ChaChaStateSizeInWords, ChaChaStateSizeInBytes, cudaMemcpyDeviceToHost);

//     uint32_t host_expected[ChaChaStateSizeInWords] = {
//         0xe4e7f110, 0x15593bd1, 0x1fdd0f50, 0xc47120a3,
//         0xc7f4d1c7, 0x0368c033, 0x9aaa2204, 0x4e6cd4c3,
//         0x466482d2, 0x09aa9f07, 0x05d7c214, 0xa2028bd9,
//         0xd19c12b5, 0xb94e16de, 0xe883d0cb, 0x4e3c50a2,
//     };

//     for (size_t i = 0; i < ChaChaStateSizeInWords; ++i)
//     {
//         printf("%x, ", host_result[i]);
//         if (host_expected[i] != host_result[i])
//         {
//             printf("\nDiffers from expected result %x at position %i", host_expected[i], i);
//             break;
//         }
//     }
//     std::cout << std::endl;
//     cudaFree(gpu_state);

// }

