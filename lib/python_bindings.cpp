// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2023 Aalto University

#include <pybind11/pybind11.h>

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

PYBIND11_MODULE(native, m)
{
    m.def("cpu_chacha20_block_factory",
          []() { return pybind11::capsule(reinterpret_cast<void*>(cpu_chacha20_block), "xla._CUSTOM_CALL_TARGET"); } );

#if (CUDA_ENABLED || HIP_ENABLED)
    m.def("gpu_chacha20_block_factory",
          []() { return pybind11::capsule(reinterpret_cast<void*>(gpu_chacha20_block), "xla._CUSTOM_CALL_TARGET"); });
#endif // (CUDA_ENABLED || HIP_ENABLED)

    m.def("cuda_supported", &cuda_supported, "Returns true if CUDA GPU kernels were compiled.");
    m.def("hip_supported", &hip_supported, "Returns true if AMD GPU kernels were compiled.");
    m.def("openmp_accelerated", &openmp_accelerated, "Returns true if CPU kernels are accelerated using OpenMP.");
}
