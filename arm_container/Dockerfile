# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Aalto University

FROM ubuntu:jammy AS qemu-arm-base

RUN sed -i 's/http/[arch=i386,amd64] http/g' /etc/apt/sources.list
ADD arm-cross-compile-sources.list /etc/apt/sources.list.d/

RUN dpkg --add-architecture arm64
RUN apt update && apt upgrade -y && apt install -y qemu binfmt-support qemu-user-static libc6:arm64

FROM qemu-arm-base

RUN apt install -y gcc-aarch64-linux-gnu g++-aarch64-linux-gnu libc6-dev-arm64-cross cmake python3 python3-dev libpython3-dev:arm64
RUN mkdir -p /work/build/ /work/src/
ADD docker_main.sh /work/docker_main.sh
RUN chmod +x /work/docker_main.sh
WORKDIR /work

ENTRYPOINT [ "/work/docker_main.sh" ]
