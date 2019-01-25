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

namespace arrow {
namespace BitUtil {

class BitWriter;

}  // namespace BitUtil
}  // namespace arrow

namespace parquet {

class ColumnDescriptor;

// Untyped base for all encoders
class Encoder {
 public:
  virtual ~Encoder() = default;

  virtual int64_t EstimatedDataEncodedSize() = 0;
  virtual std::shared_ptr<Buffer> FlushValues() = 0;

  Encoding::type encoding() const { return encoding_; }

 protected:
  explicit Encoder(const ColumnDescriptor* descr, Encoding::type encoding,
                   ::arrow::MemoryPool* pool)
      : descr_(descr), encoding_(encoding), pool_(pool) {}

  // For accessing type-specific metadata, like FIXED_LEN_BYTE_ARRAY
  const ColumnDescriptor* descr_;
  const Encoding::type encoding_;
  ::arrow::MemoryPool* pool_;
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
    auto status =
        ::arrow::AllocateResizableBuffer(pool_, num_values * sizeof(T), &buffer);
    if (!status.ok()) {
      std::ostringstream ss;
      ss << "AllocateResizableBuffer failed in Encoder.PutSpaced in " << __FILE__
         << ", on line " << __LINE__;
      throw ParquetException(ss.str());
    }
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

namespace detail {

template <typename DType>
class PlainEncoder : public TypedEncoder<DType> {
 public:
  using T = typename DType::c_type;

  explicit PlainEncoder(const ColumnDescriptor* descr,
                        ::arrow::MemoryPool* pool = ::arrow::default_memory_pool());

  int64_t EstimatedDataEncodedSize() override;
  std::shared_ptr<Buffer> FlushValues() override;

  void Put(const T* buffer, int num_values) override;

 protected:
  std::unique_ptr<InMemoryOutputStream> values_sink_;
};

}  // namespace detail

class PlainInt32Encoder : public detail::PlainEncoder<Int32Type> {
 public:
  using BASE = detail::PlainEncoder<Int32Type>;
  using BASE::PlainEncoder;
};

class PlainInt64Encoder : public detail::PlainEncoder<Int64Type> {
 public:
  using BASE = detail::PlainEncoder<Int64Type>;
  using BASE::PlainEncoder;
};

class PlainInt96Encoder : public detail::PlainEncoder<Int96Type> {
 public:
  using BASE = detail::PlainEncoder<Int96Type>;
  using BASE::PlainEncoder;
};

class PlainFloatEncoder : public detail::PlainEncoder<FloatType> {
 public:
  using BASE = detail::PlainEncoder<FloatType>;
  using BASE::PlainEncoder;
};

class PlainDoubleEncoder : public detail::PlainEncoder<DoubleType> {
 public:
  using BASE = detail::PlainEncoder<DoubleType>;
  using BASE::PlainEncoder;
};

class PlainByteArrayEncoder : public detail::PlainEncoder<ByteArrayType> {
 public:
  using BASE = detail::PlainEncoder<ByteArrayType>;
  using BASE::PlainEncoder;
};

class PlainFLBAEncoder : public detail::PlainEncoder<FLBAType> {
 public:
  using BASE = detail::PlainEncoder<FLBAType>;
  using BASE::PlainEncoder;
};

class PlainBooleanEncoder : public TypedEncoder<BooleanType> {
 public:
  explicit PlainBooleanEncoder(
      const ColumnDescriptor* descr,
      ::arrow::MemoryPool* pool = ::arrow::default_memory_pool());

  int64_t EstimatedDataEncodedSize() override;
  std::shared_ptr<Buffer> FlushValues() override;

  void Put(const bool* src, int num_values) override;
  void Put(const std::vector<bool>& src, int num_values);

 private:
  int bits_available_;
  std::unique_ptr<::arrow::BitUtil::BitWriter> bit_writer_;
  std::shared_ptr<ResizableBuffer> bits_buffer_;
  std::unique_ptr<InMemoryOutputStream> values_sink_;

  template <typename SequenceType>
  void PutImpl(const SequenceType& src, int num_values);
};

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

template <typename T>
struct EncoderTraits {};

template <>
struct EncoderTraits<BooleanType> {
  using PlainEncoder = PlainBooleanEncoder;
};

template <>
struct EncoderTraits<Int32Type> {
  using PlainEncoder = PlainInt32Encoder;
};

template <>
struct EncoderTraits<Int64Type> {
  using PlainEncoder = PlainInt64Encoder;
};

template <>
struct EncoderTraits<Int96Type> {
  using PlainEncoder = PlainInt96Encoder;
};

template <>
struct EncoderTraits<FloatType> {
  using PlainEncoder = PlainFloatEncoder;
};

template <>
struct EncoderTraits<DoubleType> {
  using PlainEncoder = PlainDoubleEncoder;
};

template <>
struct EncoderTraits<ByteArrayType> {
  using PlainEncoder = PlainByteArrayEncoder;
};

template <>
struct EncoderTraits<FLBAType> {
  using PlainEncoder = PlainFLBAEncoder;
};

}  // namespace parquet
