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

#include "benchmark/benchmark.h"

#include <vector>

#include "arrow/compute/benchmark_util.h"
#include "arrow/compute/cast.h"
#include "arrow/compute/kernels/test_util.h"
#include "arrow/testing/gtest_util.h"
#include "arrow/testing/random.h"

namespace arrow {
namespace compute {

constexpr auto kSeed = 0x94378165;

struct CastArgs {
  // size of memory tested (per iteration) in bytes
  const int64_t size;

  // proportion of nulls in generated arrays
  double null_proportion;

  explicit CastArgs(benchmark::State& state) : size(state.range(0)), state_(state) {
    if (state.range(1) == 0) {
      this->null_proportion = 0.0;
    } else {
      this->null_proportion = std::min(1., 1. / static_cast<double>(state.range(1)));
    }
  }

  ~CastArgs() {
    state_.counters["size"] = static_cast<double>(size);
    state_.counters["null_percent"] = null_proportion * 100;
    state_.SetItemsProcessed(state_.iterations() * size);
  }

 private:
  benchmark::State& state_;
};

template <typename InputType, typename CType = typename InputType::c_type>
static void BenchmarkIntegerCast(benchmark::State& state,
                                 std::shared_ptr<DataType> to_type,
                                 const CastOptions& options, CType min, CType max) {
  CastArgs args(state);
  random::RandomArrayGenerator rand(kSeed);
  auto array = rand.Numeric<InputType>(args.size, min, max, args.null_proportion);
  for (auto _ : state) {
    ABORT_NOT_OK(Cast(array, to_type, options).status());
  }
}

std::vector<int64_t> g_data_sizes = {kL1Size, kL2Size};

void CastSetArgs(benchmark::internal::Benchmark* bench) {
  for (int64_t size : g_data_sizes) {
    for (auto nulls : std::vector<ArgsType>({1000, 10, 2, 1, 0})) {
      bench->Args({static_cast<ArgsType>(size), nulls});
    }
  }
}

static void Int64ToInt32Safe(benchmark::State& state) {
  BenchmarkIntegerCast<Int64Type>(state, int32(), CastOptions::Safe(),
                                  std::numeric_limits<int32_t>::min(),
                                  std::numeric_limits<int32_t>::max());
}

static void Int64ToInt32Unsafe(benchmark::State& state) {
  BenchmarkIntegerCast<Int64Type>(state, int32(), CastOptions::Unsafe(),
                                  std::numeric_limits<int32_t>::min(),
                                  std::numeric_limits<int32_t>::max());
}

BENCHMARK(Int64ToInt32Safe)->Apply(CastSetArgs);
BENCHMARK(Int64ToInt32Unsafe)->Apply(CastSetArgs);

}  // namespace compute
}  // namespace arrow
