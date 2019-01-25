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

#include "parquet/decoding.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "arrow/util/bit-stream-utils.h"
#include "arrow/util/bit-util.h"
#include "arrow/util/hashing.h"
#include "arrow/util/logging.h"
#include "arrow/util/macros.h"
#include "arrow/util/rle-encoding.h"

#include "parquet/decoding-internal.h"
#include "parquet/schema.h"
#include "parquet/types.h"
#include "parquet/util/memory.h"

namespace parquet {

namespace BitUtil = ::arrow::BitUtil;

class ColumnDescriptor;

void Decoder::SetData(int num_values, const uint8_t* data, int len) {
  num_values_ = num_values;
  data_ = data;
  len_ = len;
}

// ----------------------------------------------------------------------
// Encoding::PLAIN decoder implementation

template class PARQUET_TEMPLATE_EXPORT PlainDecoder<Int32Type>;
template class PARQUET_TEMPLATE_EXPORT PlainDecoder<Int64Type>;
template class PARQUET_TEMPLATE_EXPORT PlainDecoder<Int96Type>;
template class PARQUET_TEMPLATE_EXPORT PlainDecoder<FloatType>;
template class PARQUET_TEMPLATE_EXPORT PlainDecoder<DoubleType>;
template class PARQUET_TEMPLATE_EXPORT PlainDecoder<ByteArrayType>;
template class PARQUET_TEMPLATE_EXPORT PlainDecoder<FLBAType>;

PlainBooleanDecoder::PlainBooleanDecoder(const ColumnDescriptor* descr)
    : TypedDecoder<BooleanType>(descr, Encoding::PLAIN) {}

void PlainBooleanDecoder::SetData(int num_values, const uint8_t* data, int len) {
  num_values_ = num_values;
  bit_reader_.reset(new BitUtil::BitReader(data, len));
}

int PlainBooleanDecoder::Decode(uint8_t* buffer, int max_values) {
  max_values = std::min(max_values, num_values_);
  bool val;
  ::arrow::internal::BitmapWriter bit_writer(buffer, 0, max_values);
  for (int i = 0; i < max_values; ++i) {
    if (!bit_reader_->GetValue(1, &val)) {
      ParquetException::EofException();
    }
    if (val) {
      bit_writer.Set();
    }
    bit_writer.Next();
  }
  bit_writer.Finish();
  num_values_ -= max_values;
  return max_values;
}

int PlainBooleanDecoder::Decode(bool* buffer, int max_values) {
  max_values = std::min(max_values, num_values_);
  if (bit_reader_->GetBatch(1, buffer, max_values) != max_values) {
    ParquetException::EofException();
  }
  num_values_ -= max_values;
  return max_values;
}

template class PARQUET_TEMPLATE_EXPORT DictDecoder<BooleanType>;
template class PARQUET_TEMPLATE_EXPORT DictDecoder<Int32Type>;
template class PARQUET_TEMPLATE_EXPORT DictDecoder<Int64Type>;
template class PARQUET_TEMPLATE_EXPORT DictDecoder<Int96Type>;
template class PARQUET_TEMPLATE_EXPORT DictDecoder<FloatType>;
template class PARQUET_TEMPLATE_EXPORT DictDecoder<DoubleType>;
template class PARQUET_TEMPLATE_EXPORT DictDecoder<ByteArrayType>;
template class PARQUET_TEMPLATE_EXPORT DictDecoder<FLBAType>;

// ----------------------------------------------------------------------

std::unique_ptr<Decoder> MakeDecoder(Type::type type_num, Encoding::type encoding,
                                     const ColumnDescriptor* descr) {
  if (encoding == Encoding::PLAIN) {
    switch (type_num) {
      case Type::BOOLEAN:
        return std::unique_ptr<Decoder>(new PlainBooleanDecoder(descr));
      case Type::INT32:
        return std::unique_ptr<Decoder>(new PlainInt32Decoder(descr));
      case Type::INT64:
        return std::unique_ptr<Decoder>(new PlainInt64Decoder(descr));
      case Type::INT96:
        return std::unique_ptr<Decoder>(new PlainInt96Decoder(descr));
      case Type::FLOAT:
        return std::unique_ptr<Decoder>(new PlainFloatDecoder(descr));
      case Type::DOUBLE:
        return std::unique_ptr<Decoder>(new PlainDoubleDecoder(descr));
      case Type::BYTE_ARRAY:
        return std::unique_ptr<Decoder>(new PlainByteArrayDecoder(descr));
      case Type::FIXED_LEN_BYTE_ARRAY:
        return std::unique_ptr<Decoder>(new PlainFLBADecoder(descr));
      default:
        break;
    }
  } else {
    ParquetException::NYI("Selected encoding is not supported");
  }
  DCHECK(false) << "Should not be able to reach this code";
  return nullptr;
}

}  // namespace parquet
