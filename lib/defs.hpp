// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2021 Aalto University

#pragma once

typedef unsigned int uint;

constexpr uint ChaChaDoubleRoundCount = 10;
constexpr uint ChaChaStateSizeInWords = 16;
constexpr uint ChaChaStateSizeInBytes = 4 * ChaChaStateSizeInWords;
constexpr uint ChaChaStateWordsPerRow = 4;
constexpr uint ChaChaStateSizeInRows = ChaChaStateSizeInWords / ChaChaStateWordsPerRow;

#ifdef CUDA_ENABLED
// Constants for Cuda kernels
// For hardware constraints, check
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications__technical-specifications-per-compute-capability

#include <algorithm>
// Basic constraints
constexpr uint ThreadsPerState = 4;
constexpr uint WordsPerThread = ChaChaStateSizeInWords / ThreadsPerState;
constexpr uint CUDAMaximumThreadsPerBlock = 1024; // CUDA limit
constexpr uint CUDAMaximumSharedMemorySizeInBytes = 48 * 1024; // for compute capability 3.5, which we currently compile for

// Derived configuration constants
constexpr uint TargetThreadsPerBlock = 256;
constexpr uint StatesPerBlock = TargetThreadsPerBlock / ThreadsPerState;

// Sanity assertions
static_assert(TargetThreadsPerBlock < CUDAMaximumThreadsPerBlock);
static_assert(ChaChaStateSizeInWords % ThreadsPerState == 0); // thread workload per state is even
static_assert(TargetThreadsPerBlock % ThreadsPerState == 0);  // block size is multiple of threads groups for states
static_assert(StatesPerBlock * ThreadsPerState <= TargetThreadsPerBlock);
#endif // CUDA_ENABLED
