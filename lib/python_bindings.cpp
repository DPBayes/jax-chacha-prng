// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Aalto University

#include <pybind11/pybind11.h>
#include <xla/ffi/api/ffi.h>
#include <xla/ffi/api/c_api.h>
#include <numeric>

#include "chacha_kernels.hpp"

constexpr bool cuda_supported()
{
#ifdef CUDA_ENABLED
    return true;
#else
    return false;
#endif
}

constexpr bool hip_supported()
{
#ifdef HIP_ENABLED
    return true;
#else
    return false;
#endif
}

constexpr bool openmp_accelerated()
{
#ifdef OPENMP_AVAILABLE
    return true;
#else
    return false;
#endif
}

namespace ffi = xla::ffi;

template <ffi::DataType T>
ffi::Error get_num_states(const ffi::Buffer<T>& buffer, uint32_t *num_states)
{
    auto dims = buffer.dimensions();
    if (dims.size() < 2)
        return ffi::Error::InvalidArgument("A valid input must have at least two dimensions");

    // uint32_t _num_states = 1;

    uint32_t _num_states = std::accumulate(dims.begin(), dims.end() - 2, 1, std::multiplies<int>());

    // for (size_t i = 0; i < dims.size() - 2; ++i)
    // {
    //     _num_states *= dims[i];
    // }
    *num_states = _num_states;
    return ffi::Error::Success();
}

ffi::Error cpu_chacha20_block_ffi_impl(ffi::Buffer<ffi::U32> state_buffer, ffi::ResultBuffer<ffi::U32> result)
{
    uint32_t num_states = 1;
    auto error = get_num_states(state_buffer, &num_states);
    if (error.failure())
        return error;

    cpu_chacha20_block(num_states, state_buffer.typed_data(), result->typed_data());
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cpu_chacha20_block_ffi, cpu_chacha20_block_ffi_impl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::U32>>()  // state_buffer
        .Ret<ffi::Buffer<ffi::U32>>()  // result
);

#if (CUDA_ENABLED || HIP_ENABLED)
ffi::Error gpu_chacha20_block_ffi_impl(gpuStream_t stream, ffi::Buffer<ffi::U32> state_buffer, ffi::ResultBuffer<ffi::U32> result)
{
    uint32_t num_states = 1;
    auto error = get_num_states(state_buffer, &num_states);
    if (error.failure())
        return error;

    gpu_chacha20_block(stream, num_states, state_buffer.typed_data(), result->typed_data());
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    gpu_chacha20_block_ffi, gpu_chacha20_block_ffi_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<gpuStream_t>>()
        .Arg<ffi::Buffer<ffi::U32>>()  // state_buffer
        .Ret<ffi::Buffer<ffi::U32>>()  // result
);
#endif // (CUDA_ENABLED || HIP_ENABLED)

PYBIND11_MODULE(native, m)
{
    m.def("cpu_chacha20_block_factory",
          []() { return pybind11::capsule(reinterpret_cast<void*>(cpu_chacha20_block_ffi)); } );

#if (CUDA_ENABLED)
    m.def("cuda_chacha20_block_factory",
          []() { return pybind11::capsule(reinterpret_cast<void*>(gpu_chacha20_block_ffi)); });
#elif (HIP_ENABLED)
    m.def("rocm_chacha20_block_factory",
          []() { return pybind11::capsule(reinterpret_cast<void*>(gpu_chacha20_block_ffi)); });
#endif // (CUDA_ENABLED || HIP_ENABLED)

    m.def("cuda_supported", &cuda_supported, "Returns true if CUDA GPU kernels were compiled.");
    m.def("hip_supported", &hip_supported, "Returns true if AMD GPU kernels were compiled.");
    m.def("openmp_accelerated", &openmp_accelerated, "Returns true if CPU kernels are accelerated using OpenMP.");
}
