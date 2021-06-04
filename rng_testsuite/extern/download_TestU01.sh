#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2021 Aalto University
# Script for downloading TestU01 source code
set -e

wget http://simul.iro.umontreal.ca/testu01/TestU01.zip
unzip TestU01.zip
rm TestU01.zip
