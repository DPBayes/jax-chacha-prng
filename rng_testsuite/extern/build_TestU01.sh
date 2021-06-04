#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2021 Aalto University
# Script for building the TestU01 libraries
set -e

cd TestU01-1.2.3
./configure --prefix=`pwd`/build
make clean
make -j
make install
