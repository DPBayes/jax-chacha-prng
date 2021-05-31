#!/bin/sh
set -e

cd TestU01-1.2.3
./configure --prefix=`pwd`/build
make clean
make -j
make install
