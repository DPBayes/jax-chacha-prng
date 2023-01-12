// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2023 Aalto University

#pragma once

#ifdef CUDA_ENABLED
    #include <cuda_runtime.h>
    typedef cudaStream_t gpuStream_t;

    #define shfl(x, y, z) __shfl_sync((uint)(-1), x, y, z)

#elif HIP_ENABLED
    #include <hip/hip_runtime.h>
    typedef hipStream_t gpuStream_t;

    #define shfl __shfl
#endif
#include <stdint.h>

#include "defs.hpp"

__global__
void chacha20_block_with_shuffle(uint32_t* out_state, const uint32_t* in_state, uint num_threads);

void gpu_chacha20_block(gpuStream_t stream, void** buffers, const char* opaque, std::size_t opaque_length);
