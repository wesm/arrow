# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

FROM openjdk:8-jre-slim-stretch

# install build essentials
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update -y -q && \
    apt-get install -y -q --no-install-recommends \
        apt-transport-https \
        ca-certificates \
        wget \
        software-properties-common \
        gnupg \
        gnupg2 && \
    wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - && \
    apt-add-repository -y "deb http://apt.llvm.org/stretch/ llvm-toolchain-stretch-7 main" && \
    apt-get update -y -q && \
    apt-get install -y -q --no-install-recommends \
        cmake \
        git \
        make \
        pkg-config \
        wget \
        clang-7 \
        clang-format-7 \
        clang-tidy-7 \
        clang-tools-7 \
        python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /

ENV CC=clang-7 \
    CXX=clang++-7

CMD ["arrow/ci/docker_lint.sh"]
