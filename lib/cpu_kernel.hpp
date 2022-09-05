// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2022 Aalto University

#pragma once

#include "defs.hpp"

#include <cpu_kernel_arch.hpp>

void chacha20_block(uint32_t out_state[16], const uint32_t in_state[16]);
void cpu_chacha20_block(void* out_buffer, const void** in_buffers);
