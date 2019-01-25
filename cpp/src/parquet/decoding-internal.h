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
#include <memory>
#include <sstream>
#include <vector>

#include "arrow/status.h"
#include "arrow/util/bit-stream-utils.h"
#include "arrow/util/bit-util.h"
#include "arrow/util/checked_cast.h"
#include "arrow/util/rle-encoding.h"

#include "parquet/decoding.h"
#include "parquet/exception.h"
#include "parquet/schema.h"
#include "parquet/types.h"
#include "parquet/util/memory.h"

namespace arrow {
namespace BitUtil {

class BitReader;

}  // namespace BitUtil
}  // namespace arrow

namespace parquet {

class ColumnDescriptor;

template <typename DType>
class PlainDecoder : public TypedDecoder<DType> {
 public:
  using T = typename DType::c_type;
  explicit PlainDecoder(const ColumnDescriptor* descr);

  int Decode(T* buffer, int max_values) override;

 protected:
  using TypedDecoder<DType>::TypedDecoder;
};

class PlainBooleanDecoder : public TypedDecoder<BooleanType> {
 public:
  explicit PlainBooleanDecoder(const ColumnDescriptor* descr);
  void SetData(int num_values, const uint8_t* data, int len) override;

  // Two flavors of bool decoding
  int Decode(uint8_t* buffer, int max_values);
  int Decode(bool* buffer, int max_values) override;

 private:
  std::unique_ptr<::arrow::BitUtil::BitReader> bit_reader_;
};

using PlainInt32Decoder = PlainDecoder<Int32Type>;
using PlainInt64Decoder = PlainDecoder<Int64Type>;
using PlainInt96Decoder = PlainDecoder<Int96Type>;
using PlainFloatDecoder = PlainDecoder<FloatType>;
using PlainDoubleDecoder = PlainDecoder<DoubleType>;

class PlainByteArrayDecoder : public PlainDecoder<ByteArrayType> {
 public:
  using Base = PlainDecoder<ByteArrayType>;
  using Base::PlainDecoder;
};

class PlainFLBADecoder : public PlainDecoder<FLBAType> {
 public:
  using Base = PlainDecoder<FLBAType>;
  using Base::PlainDecoder;
};

// ----------------------------------------------------------------------
// Dictionary encoding and decoding

template <typename Type>
class DictDecoder : public TypedDecoder<Type> {
 public:
  typedef typename Type::c_type T;

  // Initializes the dictionary with values from 'dictionary'. The data in
  // dictionary is not guaranteed to persist in memory after this call so the
  // dictionary decoder needs to copy the data out if necessary.
  explicit DictDecoder(const ColumnDescriptor* descr,
                       ::arrow::MemoryPool* pool = ::arrow::default_memory_pool())
      : TypedDecoder<Type>(descr, Encoding::RLE_DICTIONARY),
        dictionary_(0, pool),
        byte_array_data_(AllocateBuffer(pool, 0)) {}

  // Perform type-specific initiatialization
  void SetDict(TypedDecoder<Type>* dictionary);

  void SetData(int num_values, const uint8_t* data, int len) override {
    num_values_ = num_values;
    if (len == 0) return;
    uint8_t bit_width = *data;
    ++data;
    --len;
    idx_decoder_ = ::arrow::util::RleDecoder(data, len, bit_width);
  }

  int Decode(T* buffer, int max_values) override {
    max_values = std::min(max_values, num_values_);
    int decoded_values =
        idx_decoder_.GetBatchWithDict(dictionary_.data(), buffer, max_values);
    if (decoded_values != max_values) {
      ParquetException::EofException();
    }
    num_values_ -= max_values;
    return max_values;
  }

  int DecodeSpaced(T* buffer, int num_values, int null_count, const uint8_t* valid_bits,
                   int64_t valid_bits_offset) override {
    int decoded_values =
        idx_decoder_.GetBatchWithDictSpaced(dictionary_.data(), buffer, num_values,
                                            null_count, valid_bits, valid_bits_offset);
    if (decoded_values != num_values) {
      ParquetException::EofException();
    }
    return decoded_values;
  }

 private:
  using TypedDecoder<Type>::num_values_;

  // Only one is set.
  Vector<T> dictionary_;

  // Data that contains the byte array data (byte_array_dictionary_ just has the
  // pointers).
  std::shared_ptr<ResizableBuffer> byte_array_data_;

  ::arrow::util::RleDecoder idx_decoder_;
};

template <typename Type>
inline void DictDecoder<Type>::SetDict(TypedDecoder<Type>* dictionary) {
  int num_dictionary_values = dictionary->values_left();
  dictionary_.Resize(num_dictionary_values);
  dictionary->Decode(dictionary_.data(), num_dictionary_values);
}

template <>
inline void DictDecoder<BooleanType>::SetDict(TypedDecoder<BooleanType>* dictionary) {
  ParquetException::NYI("Dictionary encoding is not implemented for boolean values");
}

template <>
inline void DictDecoder<ByteArrayType>::SetDict(TypedDecoder<ByteArrayType>* dictionary) {
  int num_dictionary_values = dictionary->values_left();
  dictionary_.Resize(num_dictionary_values);
  dictionary->Decode(&dictionary_[0], num_dictionary_values);

  int total_size = 0;
  for (int i = 0; i < num_dictionary_values; ++i) {
    total_size += dictionary_[i].len;
  }
  if (total_size > 0) {
    PARQUET_THROW_NOT_OK(byte_array_data_->Resize(total_size, false));
  }

  int offset = 0;
  uint8_t* bytes_data = byte_array_data_->mutable_data();
  for (int i = 0; i < num_dictionary_values; ++i) {
    memcpy(bytes_data + offset, dictionary_[i].ptr, dictionary_[i].len);
    dictionary_[i].ptr = bytes_data + offset;
    offset += dictionary_[i].len;
  }
}

template <>
inline void DictDecoder<FLBAType>::SetDict(TypedDecoder<FLBAType>* dictionary) {
  int num_dictionary_values = dictionary->values_left();
  dictionary_.Resize(num_dictionary_values);
  dictionary->Decode(&dictionary_[0], num_dictionary_values);

  int fixed_len = descr_->type_length();
  int total_size = num_dictionary_values * fixed_len;

  PARQUET_THROW_NOT_OK(byte_array_data_->Resize(total_size, false));
  uint8_t* bytes_data = byte_array_data_->mutable_data();
  for (int32_t i = 0, offset = 0; i < num_dictionary_values; ++i, offset += fixed_len) {
    memcpy(bytes_data + offset, dictionary_[i].ptr, fixed_len);
    dictionary_[i].ptr = bytes_data + offset;
  }
}

using DictBooleanDecoder = DictDecoder<BooleanType>;
using DictInt32Decoder = DictDecoder<Int32Type>;
using DictInt64Decoder = DictDecoder<Int64Type>;
using DictInt96Decoder = DictDecoder<Int96Type>;
using DictFloatDecoder = DictDecoder<FloatType>;
using DictDoubleDecoder = DictDecoder<DoubleType>;

class DictByteArrayDecoder : public DictDecoder<ByteArrayType> {
 public:
  using BASE = DictDecoder<ByteArrayType>;
  using BASE::DictDecoder;
};

class DictFLBADecoder : public DictDecoder<FLBAType> {
 public:
  using BASE = DictDecoder<FLBAType>;
  using BASE::DictDecoder;
};

// ----------------------------------------------------------------------
// DeltaBitPackDecoder

template <typename DType>
class DeltaBitPackDecoder : public TypedDecoder<DType> {
 public:
  typedef typename DType::c_type T;

  explicit DeltaBitPackDecoder(const ColumnDescriptor* descr,
                               ::arrow::MemoryPool* pool = ::arrow::default_memory_pool())
      : TypedDecoder<DType>(descr, Encoding::DELTA_BINARY_PACKED), pool_(pool) {
    if (DType::type_num != Type::INT32 && DType::type_num != Type::INT64) {
      throw ParquetException("Delta bit pack encoding should only be for integer data.");
    }
  }

  virtual void SetData(int num_values, const uint8_t* data, int len) {
    num_values_ = num_values;
    decoder_ = ::arrow::BitUtil::BitReader(data, len);
    values_current_block_ = 0;
    values_current_mini_block_ = 0;
  }

  virtual int Decode(T* buffer, int max_values) {
    return GetInternal(buffer, max_values);
  }

 private:
  using TypedDecoder<DType>::num_values_;

  void InitBlock() {
    int32_t block_size;
    if (!decoder_.GetVlqInt(&block_size)) ParquetException::EofException();
    if (!decoder_.GetVlqInt(&num_mini_blocks_)) ParquetException::EofException();
    if (!decoder_.GetVlqInt(&values_current_block_)) {
      ParquetException::EofException();
    }
    if (!decoder_.GetZigZagVlqInt(&last_value_)) ParquetException::EofException();

    delta_bit_widths_ = AllocateBuffer(pool_, num_mini_blocks_);
    uint8_t* bit_width_data = delta_bit_widths_->mutable_data();

    if (!decoder_.GetZigZagVlqInt(&min_delta_)) ParquetException::EofException();
    for (int i = 0; i < num_mini_blocks_; ++i) {
      if (!decoder_.GetAligned<uint8_t>(1, bit_width_data + i)) {
        ParquetException::EofException();
      }
    }
    values_per_mini_block_ = block_size / num_mini_blocks_;
    mini_block_idx_ = 0;
    delta_bit_width_ = bit_width_data[0];
    values_current_mini_block_ = values_per_mini_block_;
  }

  template <typename T>
  int GetInternal(T* buffer, int max_values) {
    max_values = std::min(max_values, num_values_);
    const uint8_t* bit_width_data = delta_bit_widths_->data();
    for (int i = 0; i < max_values; ++i) {
      if (ARROW_PREDICT_FALSE(values_current_mini_block_ == 0)) {
        ++mini_block_idx_;
        if (mini_block_idx_ < static_cast<size_t>(delta_bit_widths_->size())) {
          delta_bit_width_ = bit_width_data[mini_block_idx_];
          values_current_mini_block_ = values_per_mini_block_;
        } else {
          InitBlock();
          buffer[i] = last_value_;
          continue;
        }
      }

      // TODO: the key to this algorithm is to decode the entire miniblock at once.
      int64_t delta;
      if (!decoder_.GetValue(delta_bit_width_, &delta)) ParquetException::EofException();
      delta += min_delta_;
      last_value_ += static_cast<int32_t>(delta);
      buffer[i] = last_value_;
      --values_current_mini_block_;
    }
    num_values_ -= max_values;
    return max_values;
  }

  ::arrow::MemoryPool* pool_;
  ::arrow::BitUtil::BitReader decoder_;
  int32_t values_current_block_;
  int32_t num_mini_blocks_;
  uint64_t values_per_mini_block_;
  uint64_t values_current_mini_block_;

  int32_t min_delta_;
  size_t mini_block_idx_;
  std::shared_ptr<ResizableBuffer> delta_bit_widths_;
  int delta_bit_width_;

  int32_t last_value_;
};

// ----------------------------------------------------------------------
// DELTA_LENGTH_BYTE_ARRAY

class DeltaLengthByteArrayDecoder : public TypedDecoder<ByteArrayType> {
 public:
  explicit DeltaLengthByteArrayDecoder(
      const ColumnDescriptor* descr,
      ::arrow::MemoryPool* pool = ::arrow::default_memory_pool())
      : TypedDecoder<ByteArrayType>(descr, Encoding::DELTA_LENGTH_BYTE_ARRAY),
        len_decoder_(nullptr, pool) {}

  virtual void SetData(int num_values, const uint8_t* data, int len) {
    num_values_ = num_values;
    if (len == 0) return;
    int total_lengths_len = *reinterpret_cast<const int*>(data);
    data += 4;
    len_decoder_.SetData(num_values, data, total_lengths_len);
    data_ = data + total_lengths_len;
    len_ = len - 4 - total_lengths_len;
  }

  virtual int Decode(ByteArray* buffer, int max_values) {
    max_values = std::min(max_values, num_values_);
    std::vector<int> lengths(max_values);
    len_decoder_.Decode(lengths.data(), max_values);
    for (int i = 0; i < max_values; ++i) {
      buffer[i].len = lengths[i];
      buffer[i].ptr = data_;
      data_ += lengths[i];
      len_ -= lengths[i];
    }
    num_values_ -= max_values;
    return max_values;
  }

 private:
  using TypedDecoder<ByteArrayType>::num_values_;
  using TypedDecoder<ByteArrayType>::data_;
  using TypedDecoder<ByteArrayType>::len_;
  DeltaBitPackDecoder<Int32Type> len_decoder_;
};

// ----------------------------------------------------------------------
// DELTA_BYTE_ARRAY

class DeltaByteArrayDecoder : public TypedDecoder<ByteArrayType> {
 public:
  explicit DeltaByteArrayDecoder(
      const ColumnDescriptor* descr,
      ::arrow::MemoryPool* pool = ::arrow::default_memory_pool())
      : TypedDecoder<ByteArrayType>(descr, Encoding::DELTA_BYTE_ARRAY),
        prefix_len_decoder_(nullptr, pool),
        suffix_decoder_(nullptr, pool),
        last_value_(0, nullptr) {}

  virtual void SetData(int num_values, const uint8_t* data, int len) {
    num_values_ = num_values;
    if (len == 0) return;
    int prefix_len_length = *reinterpret_cast<const int*>(data);
    data += 4;
    len -= 4;
    prefix_len_decoder_.SetData(num_values, data, prefix_len_length);
    data += prefix_len_length;
    len -= prefix_len_length;
    suffix_decoder_.SetData(num_values, data, len);
  }

  // TODO: this doesn't work and requires memory management. We need to allocate
  // new strings to store the results.
  virtual int Decode(ByteArray* buffer, int max_values) {
    max_values = std::min(max_values, num_values_);
    for (int i = 0; i < max_values; ++i) {
      int prefix_len = 0;
      prefix_len_decoder_.Decode(&prefix_len, 1);
      ByteArray suffix = {0, nullptr};
      suffix_decoder_.Decode(&suffix, 1);
      buffer[i].len = prefix_len + suffix.len;

      uint8_t* result = reinterpret_cast<uint8_t*>(malloc(buffer[i].len));
      memcpy(result, last_value_.ptr, prefix_len);
      memcpy(result + prefix_len, suffix.ptr, suffix.len);

      buffer[i].ptr = result;
      last_value_ = buffer[i];
    }
    num_values_ -= max_values;
    return max_values;
  }

 private:
  using TypedDecoder<ByteArrayType>::num_values_;

  DeltaBitPackDecoder<Int32Type> prefix_len_decoder_;
  DeltaLengthByteArrayDecoder suffix_decoder_;
  ByteArray last_value_;
};

// ----------------------------------------------------------------------
// Static decoder traits

template <typename T>
struct DecoderTraits {};

template <>
struct DecoderTraits<BooleanType> {
  using PlainDecoder = PlainBooleanDecoder;
  using DictDecoder = DictDecoder<BooleanType>;
};

template <>
struct DecoderTraits<Int32Type> {
  using PlainDecoder = PlainInt32Decoder;
  using DictDecoder = DictInt32Decoder;
};

template <>
struct DecoderTraits<Int64Type> {
  using PlainDecoder = PlainInt64Decoder;
  using DictDecoder = DictInt64Decoder;
};

template <>
struct DecoderTraits<Int96Type> {
  using PlainDecoder = PlainInt96Decoder;
  using DictDecoder = DictInt96Decoder;
};

template <>
struct DecoderTraits<FloatType> {
  using PlainDecoder = PlainFloatDecoder;
  using DictDecoder = DictFloatDecoder;
};

template <>
struct DecoderTraits<DoubleType> {
  using PlainDecoder = PlainDoubleDecoder;
  using DictDecoder = DictDoubleDecoder;
};

template <>
struct DecoderTraits<ByteArrayType> {
  using PlainDecoder = PlainByteArrayDecoder;
  using DictDecoder = DictByteArrayDecoder;
};

template <>
struct DecoderTraits<FLBAType> {
  using PlainDecoder = PlainFLBADecoder;
  using DictDecoder = DictFLBADecoder;
};

}  // namespace parquet
