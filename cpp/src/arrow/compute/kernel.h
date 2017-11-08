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

#ifndef ARROW_COMPUTE_KERNEL_H
#define ARROW_COMPUTE_KERNEL_H

#include <memory>

#include "arrow/array.h"
#include "arrow/table.h"
#include "arrow/util/macros.h"
#include "arrow/util/visibility.h"

namespace arrow {
namespace compute {

class FunctionContext;

/// \class OpKernel
/// \brief Base class for operator kernels
class ARROW_EXPORT OpKernel {
 public:
  virtual ~OpKernel() = default;
};

struct ARROW_EXPORT Scalar {
  ~Scalar() {}

  ARROW_DISALLOW_COPY_AND_ASSIGN(Scalar);
};

struct ARROW_EXPORT Datum {
  enum type { SCALAR, ARRAY, CHUNKED_ARRAY, RECORD_BATCH, TABLE };

  type kind;

  union {
    std::shared_ptr<Scalar> scalar;
    std::shared_ptr<ArrayData> array;
    std::shared_ptr<ChunkedArray> chunked_array;
    std::shared_ptr<RecordBatch> record_batch;
    std::shared_ptr<Table> table;
  };

  explicit Datum(const std::shared_ptr<Scalar>& value)
      : kind(Datum::SCALAR), scalar(value) {}

  explicit Datum(const std::shared_ptr<ArrayData>& value)
      : kind(Datum::ARRAY), array(value) {}

  explicit Datum(const std::shared_ptr<ChunkedArray>& value)
      : kind(Datum::CHUNKED_ARRAY), chunked_array(value) {}

  explicit Datum(const std::shared_ptr<RecordBatch>& value)
      : kind(Datum::RECORD_BATCH), record_batch(value) {}

  explicit Datum(const std::shared_ptr<Table>& value)
      : kind(Datum::TABLE), table(value) {}

  ~Datum() {}

  Datum(const Datum& other) noexcept : kind(other.kind) {
    switch (other.kind) {
      case Datum::SCALAR:
        this->scalar = other.scalar;
        break;
      case Datum::ARRAY:
        this->array = other.array;
        break;
      case Datum::CHUNKED_ARRAY:
        this->chunked_array = other.chunked_array;
        break;
      case Datum::RECORD_BATCH:
        this->record_batch = other.record_batch;
        break;
      case Datum::TABLE:
        this->table = other.table;
        break;
      default:
        break;
    }
  }

  bool is_arraylike() const {
    return this->kind == Datum::ARRAY || this->kind == Datum::CHUNKED_ARRAY;
  }

  std::shared_ptr<DataType> type() const {
    if (this->kind == Datum::ARRAY) {
      return this->array->type;
    } else if (this->kind == Datum::CHUNKED_ARRAY) {
      return this->chunked_array->type();
    }
    return nullptr;
  }
};

/// \class UnaryKernel
/// \brief An array-valued function of a single input argument
class ARROW_EXPORT UnaryKernel : public OpKernel {
 public:
  virtual Status Call(FunctionContext* ctx, const ArrayData& input,
                      std::vector<Datum>* out) = 0;
};

}  // namespace compute
}  // namespace arrow

#endif  // ARROW_COMPUTE_KERNEL_H
