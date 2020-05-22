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

#include "arrow/compute/kernels/codegen_internal.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "arrow/type_fwd.h"
#include "arrow/util/logging.h"

namespace arrow {
namespace compute {

void ExecFail(KernelContext* ctx, const ExecBatch& batch, Datum* out) {
  ctx->SetStatus(Status::NotImplemented("This kernel is malformed"));
}

void BinaryExecFlipped(KernelContext* ctx, ArrayKernelExec exec,
                       const ExecBatch& batch, Datum* out) {
  ExecBatch flipped_batch = batch;
  std::swap(flipped_batch.values[0], flipped_batch.values[1]);
  exec(ctx, flipped_batch, out);
}

std::vector<std::shared_ptr<DataType>> g_signed_int_types;
std::vector<std::shared_ptr<DataType>> g_unsigned_int_types;
std::vector<std::shared_ptr<DataType>> g_int_types;
std::vector<std::shared_ptr<DataType>> g_floating_types;
std::vector<std::shared_ptr<DataType>> g_numeric_types;
std::vector<std::shared_ptr<DataType>> g_base_binary_types;
std::vector<std::shared_ptr<DataType>> g_temporal_types;
std::vector<std::shared_ptr<DataType>> g_primitive_types;
static std::once_flag codegen_static_initialized;

static void InitStaticData() {
  // Signed int types
  g_signed_int_types = {int8(), int16(), int32(), int64()};

  // Unsigned int types
  g_unsigned_int_types = {uint8(), uint16(), uint32(), uint64()};

  // All int types
  Extend(g_unsigned_int_types, &g_int_types);
  Extend(g_signed_int_types, &g_int_types);

  // Floating point types
  g_floating_types = {float32(), float64()};

  // Numeric types
  Extend(g_int_types, &g_numeric_types);
  Extend(g_floating_types, &g_numeric_types);

  // Temporal types
  g_temporal_types = {date32(), date64(), time32(TimeUnit::SECOND),
                      time32(TimeUnit::MILLI), time64(TimeUnit::MICRO),
                      time64(TimeUnit::NANO), timestamp(TimeUnit::SECOND),
                      timestamp(TimeUnit::MILLI), timestamp(TimeUnit::MICRO),
                      timestamp(TimeUnit::NANO)};

  // Base binary types (without FixedSizeBinary)
  g_base_binary_types = {binary(), utf8(), large_binary(), large_utf8()};

  // Non-parametric, non-nested types. This also DOES NOT include
  //
  // * Decimal
  // * Fixed Size Binary
  g_primitive_types = {null(), boolean()};
  Extend(g_numeric_types, &g_primitive_types);
  Extend(g_temporal_types, &g_primitive_types);
  Extend(g_base_binary_types, &g_primitive_types);
}

const std::vector<std::shared_ptr<DataType>>& BaseBinaryTypes() {
  std::call_once(codegen_static_initialized, InitStaticData);
  return g_base_binary_types;
}

const std::vector<std::shared_ptr<DataType>>& SignedIntTypes() {
  std::call_once(codegen_static_initialized, InitStaticData);
  return g_signed_int_types;
}

const std::vector<std::shared_ptr<DataType>>& UnsignedIntTypes() {
  std::call_once(codegen_static_initialized, InitStaticData);
  return g_unsigned_int_types;
}

const std::vector<std::shared_ptr<DataType>>& IntTypes() {
  std::call_once(codegen_static_initialized, InitStaticData);
  return g_int_types;
}

const std::vector<std::shared_ptr<DataType>>& FloatingPointTypes() {
  std::call_once(codegen_static_initialized, InitStaticData);
  return g_floating_types;
}

const std::vector<std::shared_ptr<DataType>>& NumericTypes() {
  std::call_once(codegen_static_initialized, InitStaticData);
  return g_numeric_types;
}

const std::vector<std::shared_ptr<DataType>>& TemporalTypes() {
  std::call_once(codegen_static_initialized, InitStaticData);
  return g_temporal_types;
}

const std::vector<std::shared_ptr<DataType>>& PrimitiveTypes() {
  std::call_once(codegen_static_initialized, InitStaticData);
  return g_primitive_types;
}

Result<ValueDescr> FirstType(KernelContext*, const std::vector<ValueDescr>& descrs) {
  return descrs[0];
}

}  // namespace compute
}  // namespace arrow
