#!/bin/sh

curl -o installer.run -s https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda_11.1.1_455.32.00_linux.run
/bin/sh ./installer.run --toolkit --silent