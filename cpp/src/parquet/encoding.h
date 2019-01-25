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
  virtual Encoding::type encoding() const = 0;

  virtual ::arrow::MemoryPool* memory_pool() const = 0;
};

// Base class for value encoders. Since encoders may or not have state (e.g.,
// dictionary encoding) we use a class instance to maintain any state.
//
// TODO(wesm): Encode interface API is temporary
template <typename DType>
class TypedEncoder : virtual public Encoder {
 public:
  typedef typename DType::c_type T;

  virtual void Put(const T* src, int num_values) = 0;

  virtual void PutSpaced(const T* src, int num_values, const uint8_t* valid_bits,
                         int64_t valid_bits_offset) {
    std::shared_ptr<ResizableBuffer> buffer;
    PARQUET_THROW_NOT_OK(::arrow::AllocateResizableBuffer(
        this->memory_pool(), num_values * sizeof(T), &buffer));
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
};

// Base class for dictionary encoders
class PARQUET_EXPORT DictEncoder {
 public:
  virtual ~DictEncoder() = default;

  /// Writes out any buffered indices to buffer preceded by the bit width of this data.
  /// Returns the number of bytes written.
  /// If the supplied buffer is not big enough, returns -1.
  /// buffer must be preallocated with buffer_len bytes. Use EstimatedDataEncodedSize()
  /// to size buffer.
  int WriteIndices(uint8_t* buffer, int buffer_len);

  int dict_encoded_size() { return dict_encoded_size_; }

  virtual int bit_width() const = 0;

  /// Writes out the encoded dictionary to buffer. buffer must be preallocated to
  /// dict_encoded_size() bytes.
  virtual void WriteDict(uint8_t* buffer) = 0;

  virtual int num_entries() const = 0;

 protected:
  DictEncoder();
  /// Clears all the indices (but leaves the dictionary).
  void ClearIndices() { buffered_indices_.clear(); }

  /// Indices that have not yet be written out by WriteIndices().
  std::vector<int> buffered_indices_;

  /// The number of bytes needed to encode the dictionary.
  int dict_encoded_size_;
};

PARQUET_EXPORT
std::unique_ptr<Encoder> MakeEncoder(
    Type::type type_num, Encoding::type encoding, bool use_dictionary = false,
    const ColumnDescriptor* descr = NULLPTR,
    ::arrow::MemoryPool* pool = ::arrow::default_memory_pool());

template <typename DType>
std::unique_ptr<TypedEncoder<DType>> MakeTypedEncoder(
    Encoding::type encoding, bool use_dictionary = false,
    const ColumnDescriptor* descr = NULLPTR,
    ::arrow::MemoryPool* pool = ::arrow::default_memory_pool()) {
  std::unique_ptr<Encoder> base =
      MakeEncoder(DType::type_num, encoding, use_dictionary, descr, pool);
  return std::unique_ptr<TypedEncoder<DType>>(
      ::arrow::internal::checked_cast<TypedEncoder<DType>*>(base.release()));
}

class BooleanEncoder : virtual public TypedEncoder<BooleanType> {
 public:
  using TypedEncoder<BooleanType>::Put;
  virtual void Put(const std::vector<bool>& src, int num_values) = 0;
};

using Int32Encoder = TypedEncoder<Int32Type>;
using Int64Encoder = TypedEncoder<Int64Type>;
using Int96Encoder = TypedEncoder<Int96Type>;
using FloatEncoder = TypedEncoder<FloatType>;
using DoubleEncoder = TypedEncoder<DoubleType>;
class ByteArrayEncoder : virtual public TypedEncoder<ByteArrayType> {};
class FLBAEncoder : virtual public TypedEncoder<FLBAType> {};

template <typename T>
struct TypeTraits {};

template <>
struct TypeTraits<BooleanType> {
  using Encoder = BooleanEncoder;
};

template <>
struct TypeTraits<Int32Type> {
  using Encoder = Int32Encoder;
};

template <>
struct TypeTraits<Int64Type> {
  using Encoder = Int64Encoder;
};

template <>
struct TypeTraits<Int96Type> {
  using Encoder = Int96Encoder;
};

template <>
struct TypeTraits<FloatType> {
  using Encoder = FloatEncoder;
};

template <>
struct TypeTraits<DoubleType> {
  using Encoder = DoubleEncoder;
};

template <>
struct TypeTraits<ByteArrayType> {
  using Encoder = ByteArrayEncoder;
};

template <>
struct TypeTraits<FLBAType> {
  using Encoder = FLBAEncoder;
};

}  // namespace parquet
