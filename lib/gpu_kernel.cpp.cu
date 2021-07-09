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

struct uint32_vec4
{
    uint32_t x;
    uint32_t y;
    uint32_t z;
    uint32_t w;
};

__device__
uint32_vec4 quarterround_with_shuffle(uint32_vec4 vec)
{
    uint32_t a = vec.x;
    uint32_t b = vec.y;
    uint32_t c = vec.z;
    uint32_t d = vec.w;

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
void double_round_with_shuffle(
    uint32_t out_state[ChaChaStateSizeInBytes],
    const uint32_t in_state[ChaChaStateSizeInBytes])
{
    int thread_id = threadIdx.x;
    uint32_vec4 first_vec = {
        in_state[0*4 + thread_id],
        in_state[1*4 + thread_id],
        in_state[2*4 + thread_id],
        in_state[3*4 + thread_id],
    };
    uint32_vec4 second_vec = quarterround_with_shuffle(first_vec);
    second_vec.y = __shfl_sync((uint)-1, second_vec.y, (thread_id + 1) % 4);
    second_vec.z = __shfl_sync((uint)-1, second_vec.z, (thread_id + 2) % 4);
    second_vec.w = __shfl_sync((uint)-1, second_vec.w, (thread_id + 3) % 4);
    uint32_vec4 third_vec = quarterround_with_shuffle(second_vec);
    out_state[0*4 + thread_id] = third_vec.x;
    out_state[1*4 + (thread_id + 1) % 4] = third_vec.y;
    out_state[2*4 + (thread_id + 2) % 4] = third_vec.z;
    out_state[3*4 + (thread_id + 3) % 4] = third_vec.w;
    __syncthreads();
}

__device__
void quarterround(
    uint32_t out_state[ChaChaStateSizeInBytes],
    const uint32_t in_state[ChaChaStateSizeInBytes],
    const uint indices[4])
{
    uint32_t a = in_state[indices[0]];
    uint32_t b = in_state[indices[1]];
    uint32_t c = in_state[indices[2]];
    uint32_t d = in_state[indices[3]];

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

    out_state[indices[0]] = a;
    out_state[indices[1]] = b;
    out_state[indices[2]] = c;
    out_state[indices[3]] = d;
}


__device__ __inline__
uint get_diagonal_index(uint i, uint diagonal)
{
    return i * 4 + (i + diagonal) % 4;
}

__device__
void double_round(uint32_t out_state[ChaChaStateSizeInBytes], const uint32_t in_state[ChaChaStateSizeInBytes])
{
    uint thread_id = threadIdx.x;

    // quarterround on columns
    uint first_round_indices[4] = { 0 + thread_id, 4 + thread_id, 8 + thread_id, 12 + thread_id };
    quarterround(out_state, in_state, first_round_indices);
    __syncthreads();

    // quarterround on diagonals
    uint second_round_indices[4] = {
        get_diagonal_index(0, thread_id),
        get_diagonal_index(1, thread_id),
        get_diagonal_index(2, thread_id),
        get_diagonal_index(3, thread_id)
    };
    quarterround(out_state, out_state, second_round_indices);
    __syncthreads();
}

__device__
void add_states(
    uint32_t out[ChaChaStateSizeInBytes],
    const uint32_t x[ChaChaStateSizeInBytes],
    const uint32_t y[ChaChaStateSizeInBytes])
// add two 4x4 matrices using 4 threads (each processing a column in row-major layout)
{
    int thread_id = threadIdx.x;
    for (int i = 0; i < 4; ++i)
    {
        int idx = 4*i + thread_id;
        out[idx] = x[idx] + y[idx];
    }
}

__global__
void chacha20_block(uint32_t* out_state, const uint32_t* in_state)
{
    // currently this still has consecutive stores and reads to shared memory
    // when transitioning between double_round calls.
    // I think with use of __shfl these could entirely be eliminated and mapped entirely
    // to registers; would that be worthwhile, i.e., would the potential performance
    // gain outweight the fact that we would probably need to remove the functional abstractions
    // entirely (or even directly code ptx) to enforce this?

    uint buffer_offset = blockIdx.x * ChaChaStateSizeInBytes;

    __shared__ uint32_t tmp_state[16];
    // double_round_with_shuffle(tmp_state, in_state + buffer_offset);
    double_round(tmp_state, in_state + buffer_offset);
    for (uint i = 0; i < ChaChaDoubleRoundCount - 1; ++i)
    {
        // double_round_with_shuffle(tmp_state, tmp_state);
        double_round(tmp_state, tmp_state);
    }
    add_states(out_state + buffer_offset, in_state + buffer_offset, tmp_state);
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
    chacha20_block<<<num_states, 4, 0, stream>>>(out_state, in_states);
}

// TODO: some ad-hoc test code below, move into separate test file
// int main(int argc, const char** argv)
// {
//     uint32_t host_state[16] = {
//         0x61707865, 0x3320646e, 0x79622d32, 0x6b206574,
//         0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c,
//         0x13121110, 0x17161514, 0x1b1a1918, 0x1f1e1d1c,
//         0x00000001, 0x09000000, 0x4a000000, 0x00000000,
//     };
//     // for (size_t i = 0; i < 16; ++i)
//     // {
//     //     host_state[i] = i;
//     // }

//     uint32_t* gpu_state;
//     if (cudaMalloc((void**)&gpu_state, 16*sizeof(uint32_t)) != cudaSuccess)
//     {
//         std::cout << "failed to allocate device memory!" << std::endl;
//         return 1;
//     }
//     if (cudaMemcpy(gpu_state, host_state, 16*sizeof(uint32_t), cudaMemcpyHostToDevice) != cudaSuccess)
//     {
//         std::cout << "failed to copy to device memory!" << std::endl;
//         return 1;
//     }

//     uint32_t* buffers[2] = { gpu_state, gpu_state };
//     gpu_chacha20_block(/*stream=*/nullptr, reinterpret_cast<void**>(buffers), /*opaque=*/nullptr, /*opaque_length=*/0);

//     // chacha20_block<<<1, 4>>>(gpu_state); // could alternatively run 16 threads for faster final summation and idle unused while processing the 4 columns/diagonals

//     uint32_t host_result[16];
//     cudaMemcpy(host_result, gpu_state, 16*sizeof(uint32_t), cudaMemcpyDeviceToHost);

//     uint32_t host_expected[16] = {
//         0xe4e7f110, 0x15593bd1, 0x1fdd0f50, 0xc47120a3,
//         0xc7f4d1c7, 0x0368c033, 0x9aaa2204, 0x4e6cd4c3,
//         0x466482d2, 0x09aa9f07, 0x05d7c214, 0xa2028bd9,
//         0xd19c12b5, 0xb94e16de, 0xe883d0cb, 0x4e3c50a2,
//     };

//     for (size_t i = 0; i < 16; ++i)
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

