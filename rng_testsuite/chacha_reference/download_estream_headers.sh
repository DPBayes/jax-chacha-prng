#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2021 Aalto University
# Script for downloading eStream header files
set -e

wget https://www.ecrypt.eu.org/stream/call/api/ecrypt-config.h
wget https://www.ecrypt.eu.org/stream/call/api/ecrypt-machine.h
wget https://www.ecrypt.eu.org/stream/call/api/ecrypt-portable.h
