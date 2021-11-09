# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2021 Aalto University

from typing import Callable, Sequence, Any

XlaCustomCallCPU = Callable[[bytes, Sequence[bytes]], None]
XlaCustomCallGPU = Callable[[Any, Sequence[bytes], bytes, int], None]

def cpu_chacha20_block_factory() -> XlaCustomCallCPU: ...
def gpu_chacha20_block_factory() -> XlaCustomCallGPU: ...
def cuda_supported() -> bool: ...
