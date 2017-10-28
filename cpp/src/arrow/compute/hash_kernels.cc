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

#include "arrow/compute/hash_kernels.h"

#include <sstream>

#include "arrow/builder.h"
#include "arrow/compute/context.h"
#include "arrow/compute/kernel.h"

namespace arrow {
namespace compute {

namespace {

template <typename DataType>
class UniqueKernel : public UnaryKernel {
  Status Call(FunctionContext* ctx, const Array& input,
              std::shared_ptr<ArrayData>* out) override {
    if (!unique_builder) {
      unique_builder =
          std::make_shared<UniqueBuilder<DataType>>(input.type(), ctx->memory_pool());
    }

    RETURN_NOT_OK(unique_builder->AppendArray(input));

    if (out) {
      RETURN_NOT_OK(unique_builder->FinishInternal(out));
      unique_builder.reset();
    }

    return Status::OK();
  }

 private:
  std::shared_ptr<UniqueBuilder<DataType>> unique_builder;
};
}

#define UNIQUE_FUNCTION_CASE(InType)                                    \
  case InType::type_id:                                                 \
    *kernel = std::unique_ptr<UnaryKernel>(new UniqueKernel<InType>()); \
    break

Status GetUniqueFunction(const DataType& in_type, std::unique_ptr<UnaryKernel>* kernel) {
  switch (in_type.id()) {
    // UNIQUE_FUNCTION_CASE(NullType);
    // UNIQUE_FUNCTION_CASE(BooleanType);
    UNIQUE_FUNCTION_CASE(UInt8Type);
    UNIQUE_FUNCTION_CASE(Int8Type);
    UNIQUE_FUNCTION_CASE(UInt16Type);
    UNIQUE_FUNCTION_CASE(Int16Type);
    UNIQUE_FUNCTION_CASE(UInt32Type);
    UNIQUE_FUNCTION_CASE(Int32Type);
    UNIQUE_FUNCTION_CASE(UInt64Type);
    UNIQUE_FUNCTION_CASE(Int64Type);
    UNIQUE_FUNCTION_CASE(FloatType);
    UNIQUE_FUNCTION_CASE(DoubleType);
    UNIQUE_FUNCTION_CASE(Date32Type);
    UNIQUE_FUNCTION_CASE(Date64Type);
    UNIQUE_FUNCTION_CASE(Time32Type);
    UNIQUE_FUNCTION_CASE(Time64Type);
    UNIQUE_FUNCTION_CASE(TimestampType);
    UNIQUE_FUNCTION_CASE(BinaryType);
    UNIQUE_FUNCTION_CASE(StringType);
    UNIQUE_FUNCTION_CASE(FixedSizeBinaryType);
    default:
      break;
  }

  if (*kernel == nullptr) {
    std::stringstream ss;
    ss << "No unique implemented for " << in_type.ToString();
    return Status::NotImplemented(ss.str());
  }

  return Status::OK();
}

Status Unique(FunctionContext* ctx, const Array& array, std::shared_ptr<Array>* out) {
  // Dynamic dispatch to obtain right cast function
  std::unique_ptr<UnaryKernel> func;
  RETURN_NOT_OK(GetUniqueFunction(*array.type(), &func));

  std::shared_ptr<ArrayData> out_data;
  RETURN_NOT_OK(func->Call(ctx, array, &out_data));
  *out = MakeArray(out_data);
  return Status::OK();
}

Status Unique(FunctionContext* ctx, const ChunkedArray& array,
              std::shared_ptr<Array>* out) {
  // Dynamic dispatch to obtain right cast function
  std::unique_ptr<UnaryKernel> func;
  RETURN_NOT_OK(GetUniqueFunction(*array.type(), &func));

  // Call the kernel without out_data on all but the last chunk
  for (int i = 0; i < (array.num_chunks() - 1); i++) {
    RETURN_NOT_OK(func->Call(ctx, *array.chunk(i), nullptr));
  }

  std::shared_ptr<ArrayData> out_data;
  // The array has a large chunk, call the kernel and retrieve the result.
  RETURN_NOT_OK(func->Call(ctx, *array.chunk(array.num_chunks() - 1), &out_data));
  *out = MakeArray(out_data);

  return Status::OK();
}

Status Unique(FunctionContext* context, const Column& array,
              std::shared_ptr<Array>* out) {
  return Unique(context, *array.data(), out);
}

}  // compute
}  // arrow
