// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Aalto University

#pragma once


// cpu_kernel.cpp
void cpu_chacha20_block(uint32_t num_states, const uint32_t* state_buffer, uint32_t* result_buffer);

#if (CUDA_ENABLED || HIP_ENABLED)
    #ifdef CUDA_ENABLED
        #include <cuda_runtime.h>
        typedef cudaStream_t gpuStream_t;
    #endif // CUDA_ENABLED

    #ifdef HIP_ENABLED
        #include <hip/hip_runtime.h>
        typedef hipStream_t gpuStream_t;
    #endif // HIP_ENABLED

    // gpu_kernel.cpp.cu
    extern void gpu_chacha20_block(gpuStream_t stream, uint32_t num_states, const uint32_t* in_states, uint32_t* out_states);
#endif // (CUDA_ENABLED || HIP_ENABLED)
