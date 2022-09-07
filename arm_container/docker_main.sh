#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2022 Aalto University

cd /work/build
cmake \
    -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
    -DCMAKE_C_COMPILER=/usr/bin/aarch64-linux-gnu-gcc \
    -DCMAKE_CXX_COMPILER=/usr/bin/aarch64-linux-gnu-g++ \
    -DCMAKE_FIND_ROOT_PATH=/usr/aarch64-linux-gnu/ \
    -DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=BOTH \
    -DCMAKE_CROSSCOMPILING_EMULATOR="/usr/bin/qemu-aarch64-static;-L;/usr/aarch64-linux-gnu/" \
    -DBUILD_TESTING=On \
    -DFORCE_GENERIC=$FORCE_GENERIC \
    -DCMAKE_BUILD_TYPE=Release ../src/
make
/usr/bin/qemu-aarch64-static -L /usr/aarch64-linux-gnu/ ./cpu_kernel_tests