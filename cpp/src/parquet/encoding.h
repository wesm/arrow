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

#pragma once

#include <cstdint>
#include <memory>
#include <sstream>
#include <vector>

#include "arrow/status.h"
#include "arrow/util/bit-util.h"
#include "arrow/util/checked_cast.h"

#include "parquet/exception.h"
#include "parquet/schema.h"
#include "parquet/types.h"
#include "parquet/util/memory.h"

namespace parquet {

class ColumnDescriptor;

// Untyped base for all encoders
class PARQUET_EXPORT Encoder {
 public:
  virtual ~Encoder() = default;

  virtual int64_t EstimatedDataEncodedSize() = 0;
  virtual std::shared_ptr<Buffer> FlushValues() = 0;

  Encoding::type encoding() const { return encoding_; }

 protected:
  explicit Encoder(const ColumnDescriptor* descr, Encoding::type encoding,
                   ::arrow::MemoryPool* pool);

  // For accessing type-specific metadata, like FIXED_LEN_BYTE_ARRAY
  const ColumnDescriptor* descr_;
  const Encoding::type encoding_;
  ::arrow::MemoryPool* pool_;

  /// Type length from descr
  int type_length_;
};

// Base class for value encoders. Since encoders may or not have state (e.g.,
// dictionary encoding) we use a class instance to maintain any state.
//
// TODO(wesm): Encode interface API is temporary
template <typename DType>
class TypedEncoder : public Encoder {
 public:
  typedef typename DType::c_type T;

  virtual void Put(const T* src, int num_values) = 0;
  virtual void PutSpaced(const T* src, int num_values, const uint8_t* valid_bits,
                         int64_t valid_bits_offset) {
    std::shared_ptr<ResizableBuffer> buffer;
    PARQUET_THROW_NOT_OK(
        ::arrow::AllocateResizableBuffer(pool_, num_values * sizeof(T), &buffer));
    int32_t num_valid_values = 0;
    ::arrow::internal::BitmapReader valid_bits_reader(valid_bits, valid_bits_offset,
                                                      num_values);
    T* data = reinterpret_cast<T*>(buffer->mutable_data());
    for (int32_t i = 0; i < num_values; i++) {
      if (valid_bits_reader.IsSet()) {
        data[num_valid_values++] = src[i];
      }
      valid_bits_reader.Next();
    }
    Put(data, num_valid_values);
  }

 protected:
  using Encoder::Encoder;
};

PARQUET_EXPORT
std::unique_ptr<Encoder> MakeEncoder(
    Type::type type_num, Encoding::type encoding, bool use_dictionary,
    const ColumnDescriptor* descr,
    ::arrow::MemoryPool* pool = ::arrow::default_memory_pool());

template <typename DType>
std::unique_ptr<TypedEncoder<DType>> MakeTypedEncoder(
    Encoding::type encoding, bool use_dictionary, const ColumnDescriptor* descr,
    ::arrow::MemoryPool* pool = ::arrow::default_memory_pool()) {
  std::unique_ptr<Encoder> base =
      MakeEncoder(DType::type_num, encoding, use_dictionary, descr, pool);
  return std::unique_ptr<TypedEncoder<DType>>(
      ::arrow::internal::checked_cast<TypedEncoder<DType>*>(base.release()));
}

}  // namespace parquet
