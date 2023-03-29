// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2023 Aalto University

#pragma once


// cpu_kernel.cpp
extern void cpu_chacha20_block(void* out_buffer, const void** in_buffers);

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
    extern void gpu_chacha20_block(gpuStream_t stream, void** buffers, const char* opaque, std::size_t opaque_length);
#endif // (CUDA_ENABLED || HIP_ENABLED)
