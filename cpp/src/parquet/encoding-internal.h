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

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "arrow/util/bit-stream-utils.h"
#include "arrow/util/bit-util.h"
#include "arrow/util/hashing.h"
#include "arrow/util/macros.h"

#include "parquet/encoding.h"
#include "parquet/exception.h"
#include "parquet/schema.h"
#include "parquet/types.h"
#include "parquet/util/memory.h"

namespace parquet {

namespace BitUtil = ::arrow::BitUtil;

class ColumnDescriptor;

// ----------------------------------------------------------------------
// Plain encoder implementation

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

using PlainInt32Encoder = PlainEncoder<Int32Type>;
using PlainInt64Encoder = PlainEncoder<Int64Type>;
using PlainInt96Encoder = PlainEncoder<Int96Type>;
using PlainFloatEncoder = PlainEncoder<FloatType>;
using PlainDoubleEncoder = PlainEncoder<DoubleType>;

class PlainByteArrayEncoder : public PlainEncoder<ByteArrayType> {
 public:
  using BASE = PlainEncoder<ByteArrayType>;
  using BASE::PlainEncoder;
};

class PlainFLBAEncoder : public PlainEncoder<FLBAType> {
 public:
  using BASE = PlainEncoder<FLBAType>;
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

// ----------------------------------------------------------------------
// Dictionary encoder

template <typename DType>
struct DictEncoderTraits {
  using c_type = typename DType::c_type;
  using MemoTableType = ::arrow::internal::ScalarMemoTable<c_type>;
};

template <>
struct DictEncoderTraits<ByteArrayType> {
  using MemoTableType = ::arrow::internal::BinaryMemoTable;
};

template <>
struct DictEncoderTraits<FLBAType> {
  using MemoTableType = ::arrow::internal::BinaryMemoTable;
};

// Base class for dictionary encoders
class DictEncoder {
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

 protected:
  DictEncoder();
  /// Clears all the indices (but leaves the dictionary).
  void ClearIndices() { buffered_indices_.clear(); }

  /// Indices that have not yet be written out by WriteIndices().
  std::vector<int> buffered_indices_;

  /// The number of bytes needed to encode the dictionary.
  int dict_encoded_size_;
};

/// See the dictionary encoding section of https://github.com/Parquet/parquet-format.
/// The encoding supports streaming encoding. Values are encoded as they are added while
/// the dictionary is being constructed. At any time, the buffered values can be
/// written out with the current dictionary size. More values can then be added to
/// the encoder, including new dictionary entries.
template <typename DType>
class DictEncoderImpl : public TypedEncoder<DType>, public DictEncoder {
  using MemoTableType = typename DictEncoderTraits<DType>::MemoTableType;

 public:
  typedef typename DType::c_type T;

  explicit DictEncoderImpl(
      const ColumnDescriptor* desc,
      ::arrow::MemoryPool* allocator = ::arrow::default_memory_pool());

  ~DictEncoderImpl() override { DCHECK(buffered_indices_.empty()); }

  void set_type_length(int type_length) { this->type_length_ = type_length; }

  /// Returns a conservative estimate of the number of bytes needed to encode the buffered
  /// indices. Used to size the buffer passed to WriteIndices().
  int64_t EstimatedDataEncodedSize() override;

  /// The minimum bit width required to encode the currently buffered indices.
  int bit_width() const override;

  /// Encode value. Note that this does not actually write any data, just
  /// buffers the value's index to be written later.
  inline void Put(const T& value);
  void Put(const T* values, int num_values) override;

  std::shared_ptr<Buffer> FlushValues() override;

  void PutSpaced(const T* src, int num_values, const uint8_t* valid_bits,
                 int64_t valid_bits_offset) override;

  /// Writes out the encoded dictionary to buffer. buffer must be preallocated to
  /// dict_encoded_size() bytes.
  void WriteDict(uint8_t* buffer);

  /// The number of entries in the dictionary.
  int num_entries() const { return memo_table_.size(); }

 private:
  MemoTableType memo_table_;
};

using DictInt32Encoder = DictEncoderImpl<Int32Type>;
using DictInt64Encoder = DictEncoderImpl<Int64Type>;
using DictInt96Encoder = DictEncoderImpl<Int96Type>;
using DictFloatEncoder = DictEncoderImpl<FloatType>;
using DictDoubleEncoder = DictEncoderImpl<DoubleType>;

class DictByteArrayEncoder : public DictEncoderImpl<ByteArrayType> {
 public:
  using DictEncoderImpl<ByteArrayType>::DictEncoderImpl;
};

class DictFLBAEncoder : public DictEncoderImpl<FLBAType> {
 public:
  using DictEncoderImpl<FLBAType>::DictEncoderImpl;
};

template <typename DType>
inline void DictEncoderImpl<DType>::Put(const T& v) {
  // Put() implementation for primitive types
  auto on_found = [](int32_t memo_index) {};
  auto on_not_found = [this](int32_t memo_index) {
    dict_encoded_size_ += static_cast<int>(sizeof(T));
  };

  auto memo_index = memo_table_.GetOrInsert(v, on_found, on_not_found);
  buffered_indices_.push_back(memo_index);
}

template <>
inline void DictEncoderImpl<ByteArrayType>::Put(const ByteArray& v) {
  static const uint8_t empty[] = {0};

  auto on_found = [](int32_t memo_index) {};
  auto on_not_found = [&](int32_t memo_index) {
    dict_encoded_size_ += static_cast<int>(v.len + sizeof(uint32_t));
  };

  DCHECK(v.ptr != nullptr || v.len == 0);
  const void* ptr = (v.ptr != nullptr) ? v.ptr : empty;
  auto memo_index =
      memo_table_.GetOrInsert(ptr, static_cast<int32_t>(v.len), on_found, on_not_found);
  buffered_indices_.push_back(memo_index);
}

template <>
inline void DictEncoderImpl<FLBAType>::Put(const FixedLenByteArray& v) {
  static const uint8_t empty[] = {0};

  auto on_found = [](int32_t memo_index) {};
  auto on_not_found = [this](int32_t memo_index) { dict_encoded_size_ += type_length_; };

  DCHECK(v.ptr != nullptr || type_length_ == 0);
  const void* ptr = (v.ptr != nullptr) ? v.ptr : empty;
  auto memo_index = memo_table_.GetOrInsert(ptr, type_length_, on_found, on_not_found);
  buffered_indices_.push_back(memo_index);
}

// ----------------------------------------------------------------------
// Mapping from data type to the appropriate encoder classes

template <typename T>
struct EncoderTraits {};

template <>
struct EncoderTraits<BooleanType> {
  using PlainEncoder = PlainBooleanEncoder;

  // XXX(wesm): This is instantiated but never used in TypedColumnWriter<T>,
  // try to remove later
  using DictEncoder = DictEncoderImpl<BooleanType>;
};

template <>
struct EncoderTraits<Int32Type> {
  using PlainEncoder = PlainInt32Encoder;
  using DictEncoder = DictInt32Encoder;
};

template <>
struct EncoderTraits<Int64Type> {
  using PlainEncoder = PlainInt64Encoder;
  using DictEncoder = DictInt64Encoder;
};

template <>
struct EncoderTraits<Int96Type> {
  using PlainEncoder = PlainInt96Encoder;
  using DictEncoder = DictInt96Encoder;
};

template <>
struct EncoderTraits<FloatType> {
  using PlainEncoder = PlainFloatEncoder;
  using DictEncoder = DictFloatEncoder;
};

template <>
struct EncoderTraits<DoubleType> {
  using PlainEncoder = PlainDoubleEncoder;
  using DictEncoder = DictDoubleEncoder;
};

template <>
struct EncoderTraits<ByteArrayType> {
  using PlainEncoder = PlainByteArrayEncoder;
  using DictEncoder = DictByteArrayEncoder;
};

template <>
struct EncoderTraits<FLBAType> {
  using PlainEncoder = PlainFLBAEncoder;
  using DictEncoder = DictFLBAEncoder;
};

}  // namespace parquet
