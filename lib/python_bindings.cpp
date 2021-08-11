// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2021 Aalto University

#include <pybind11/pybind11.h>

// cpu_kernel.cpp
extern void cpu_chacha20_block(void* out_buffer, const void** in_buffers);

#ifdef CUDA_ENABLED
#include <cuda_runtime.h>
// gpu_kernel.cpp.cu
extern void gpu_chacha20_block(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_length);
#endif // CUDA_ENABLED

PYBIND11_MODULE(native, m)
{
    m.def("cpu_chacha20_block_factory",
          []() { return pybind11::capsule(reinterpret_cast<void*>(cpu_chacha20_block), "xla._CUSTOM_CALL_TARGET"); } );

#ifdef CUDA_ENABLED
    m.def("gpu_chacha20_block_factory",
          []() { return pybind11::capsule(reinterpret_cast<void*>(gpu_chacha20_block), "xla._CUSTOM_CALL_TARGET"); });
#endif // CUDA_ENABLED
}
