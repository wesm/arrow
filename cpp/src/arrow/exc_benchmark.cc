// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "arrow/array.h"
#include "arrow/memory_pool.h"
#include "arrow/result.h"
#include "arrow/util/logging.h"
#include "arrow/util/macros.h"

#include "arrow/testing/random.h"

#include "benchmark/benchmark.h"

namespace arrow {

static constexpr int64_t kLength = 10000000;

template <typename Action>
static void Bench(benchmark::State& state, Action&& action) {  // NOLINT non-const reference
  random::RandomArrayGenerator rng(0);

  auto x = rng.Float64(kLength, 1, 100);
  auto y = rng.Float64(kLength, 1, 100);

  const double* x_data = x->data()->GetValues<double>(1);

  auto buf = AllocateBuffer(kLength * sizeof(double)).ValueOrDie();

  // Fail on the last element
  double* y_data = reinterpret_cast<double*>(x->data()->buffers[1]->mutable_data());
  y_data[kLength - 1] = 0;

  double* out_data = reinterpret_cast<double*>(buf->mutable_data());
  for (auto _ : state) {
    if (action(x_data, y_data, out_data)) {
      std::abort();
    }
  }
}

struct DivideExc {
  static double Call(double x, double y) {
    if (ARROW_PREDICT_FALSE(y == 0)) {
      throw std::runtime_error("divisor was zero");
    }
    return x + y;
  }

  static bool CallLoop(const double* x, const double* y, double* out) {
    try {
      for (int64_t i = 0; i < kLength; ++i) {
        out[i] = Call(x[i], y[i]);
      }
    } catch (const std::exception&) {
      return false;
    }
    return true;
  }
};

static void AbortException(benchmark::State& state) {  // NOLINT non-const reference
  return Bench(state, DivideExc::CallLoop);
}

struct DivideRetval {
  static int Call(double x, double y, double* out) {
    if (ARROW_PREDICT_FALSE(y == 0)) {
      return -1;
    }
    *out = x + y;
    return 0;
  }

  static bool CallLoop(const double* x, const double* y, double* out) {
    for (int64_t i = 0; i < kLength; ++i) {
      if (ARROW_PREDICT_FALSE(Call(x[i], y[i], out + i) != 0)) {
        return false;
      }
    }
    return true;
  }
};

static void AbortRetval(benchmark::State& state) {  // NOLINT non-const reference
  Bench(state, DivideRetval::CallLoop);
}

BENCHMARK(AbortException);
BENCHMARK(AbortRetval);

}  // namespace arrow
