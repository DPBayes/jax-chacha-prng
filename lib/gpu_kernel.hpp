// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2021 Aalto University

#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#include "defs.hpp"

__global__
void chacha20_block_with_shuffle(uint32_t* out_state, const uint32_t* in_state, uint num_threads);

void gpu_chacha20_block(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_length);
