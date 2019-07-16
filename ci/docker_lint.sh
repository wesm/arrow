#!/usr/bin/env bash
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

set -ex

pushd /arrow

ci/travis_release_audit.sh || exit 1

ARROW_CPP_DIR=/arrow/cpp
ARROW_DEV_DIR=/arrow/dev
ARROW_INTEGRATION_DIR=/arrow/integration
ARROW_PYTHON_DIR=/arrow/python
ARROW_R_DIR=/arrow/r

pip3 install -U setuptools wheel cmake_format==0.5.2 flake8
pip3 install pre_commit

pre-commit install

# TODO: Add hadolint to this Docker container

# TODO: Move more checks into pre-commit as this gives a nice summary
# and doesn't abort on the first failed check.
# pre-commit run hadolint -a

# CMake formatting check
python3 /arrow/run-cmake-format.py --check

# C++ code linting
if [ "$ARROW_CI_CPP_AFFECTED" != "0" ]; then
  mkdir /cpp-lint
  pushd /cpp-lint

  cmake $ARROW_CPP_DIR -DARROW_ONLY_LINT=ON
  make lint
  make check-format

  python $ARROW_CPP_DIR/build-support/lint_cpp_cli.py $ARROW_CPP_DIR/src

  popd
fi

# Python style checks
# (need Python 3 for crossbow)
FLAKE8="python3 -m flake8"

if [ "$ARROW_CI_DEV_AFFECTED" != "0" ]; then
  $FLAKE8 --count $ARROW_DEV_DIR
fi

if [ "$ARROW_CI_INTEGRATION_AFFECTED" != "0" ]; then
  $FLAKE8 --count $ARROW_INTEGRATION_DIR
fi

if [ "$ARROW_CI_PYTHON_AFFECTED" != "0" ]; then
  $FLAKE8 --count $ARROW_PYTHON_DIR
  # Check Cython files with some checks turned off
  $FLAKE8 --count \
          --config=$ARROW_PYTHON_DIR/.flake8.cython \
          $ARROW_PYTHON_DIR
fi

if [ "$ARROW_CI_R_AFFECTED" != "0" ]; then
  pushd $ARROW_R_DIR
  ./lint.sh
  popd
fi

/arrow/ci/travis_release_test.sh
