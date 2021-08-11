#pragma once

typedef unsigned int uint;

constexpr uint ChaChaDoubleRoundCount = 10;
constexpr uint ChaChaStateSizeInWords = 16;

#ifdef CUDA_ENABLED
// Constants for Cuda kernels
constexpr uint ThreadsPerState = 4;
#endif // CUDA_ENABLED
