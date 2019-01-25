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

class BitReader;

}  // namespace BitUtil
}  // namespace arrow

namespace parquet {

class ColumnDescriptor;

// ----------------------------------------------------------------------
// Value decoding

class PARQUET_EXPORT Decoder {
 public:
  virtual ~Decoder() = default;

  // Sets the data for a new page. This will be called multiple times on the same
  // decoder and should reset all internal state.
  virtual void SetData(int num_values, const uint8_t* data, int len);

  // Returns the number of values left (for the last call to SetData()). This is
  // the number of values left in this page.
  int values_left() const { return num_values_; }

  Encoding::type encoding() const { return encoding_; }

 protected:
  explicit Decoder(const ColumnDescriptor* descr, Encoding::type encoding)
      : descr_(descr), encoding_(encoding), num_values_(0), data_(NULLPTR), len_(0) {}

  // For accessing type-specific metadata, like FIXED_LEN_BYTE_ARRAY
  const ColumnDescriptor* descr_;

  const Encoding::type encoding_;
  int num_values_;
  const uint8_t* data_;
  int len_;
  int type_length_;
};

template <typename DType>
class TypedDecoder : public Decoder {
 public:
  using T = typename DType::c_type;

  using Decoder::SetData;

  // Subclasses should override the ones they support. In each of these functions,
  // the decoder would decode put to 'max_values', storing the result in 'buffer'.
  // The function returns the number of values decoded, which should be max_values
  // except for end of the current data page.
  virtual int Decode(T* buffer, int max_values) = 0;

  // Decode the values in this data page but leave spaces for null entries.
  //
  // num_values is the size of the def_levels and buffer arrays including the number of
  // null values.
  virtual int DecodeSpaced(T* buffer, int num_values, int null_count,
                           const uint8_t* valid_bits, int64_t valid_bits_offset) {
    int values_to_read = num_values - null_count;
    int values_read = Decode(buffer, values_to_read);
    if (values_read != values_to_read) {
      throw ParquetException("Number of values / definition_levels read did not match");
    }

    // Depending on the number of nulls, some of the value slots in buffer may
    // be uninitialized, and this will cause valgrind warnings / potentially UB
    memset(static_cast<void*>(buffer + values_read), 0,
           (num_values - values_read) * sizeof(T));

    // Add spacing for null entries. As we have filled the buffer from the front,
    // we need to add the spacing from the back.
    int values_to_move = values_read;
    for (int i = num_values - 1; i >= 0; i--) {
      if (::arrow::BitUtil::GetBit(valid_bits, valid_bits_offset + i)) {
        buffer[i] = buffer[--values_to_move];
      }
    }
    return num_values;
  }

 protected:
  using Decoder::Decoder;
};

PARQUET_EXPORT
std::unique_ptr<Decoder> MakeDecoder(Type::type type_num, Encoding::type encoding,
                                     const ColumnDescriptor* descr);

template <typename DType>
std::unique_ptr<TypedDecoder<DType>> MakeTypedDecoder(Encoding::type encoding,
                                                      const ColumnDescriptor* descr) {
  std::unique_ptr<Decoder> base = MakeDecoder(DType::type_num, encoding, descr);
  return std::unique_ptr<TypedDecoder<DType>>(
      ::arrow::internal::checked_cast<TypedDecoder<DType>*>(base.release()));
}

}  // namespace parquet
