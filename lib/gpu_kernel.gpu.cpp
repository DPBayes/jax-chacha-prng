// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2023 Aalto University

#include <cstdlib>
#include <stdexcept>

#include "gpu_kernel.hpp"

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
        state[i] = shfl(
            state[i], /*srcLane=*/(state_thread_id + i) % ThreadsPerState, /*width=*/ThreadsPerState
        );
    }

    // quarterround on diagonal
    state = quarterround_with_shuffle(state);

    // shuffle back to columns
    for (int i = 1; i < WordsPerThread; ++i)
    {
        state[i] = shfl(
            state[i], /*srcLane=*/(state_thread_id + ThreadsPerState - i) % ThreadsPerState, /*width=*/ThreadsPerState
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

void gpu_chacha20_block(gpuStream_t stream, void** buffers, const char* opaque, std::size_t opaque_length)
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
