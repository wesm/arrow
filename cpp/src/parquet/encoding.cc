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

#include "parquet/encoding.h"

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

#include "parquet/encoding-internal.h"
#include "parquet/exception.h"
#include "parquet/schema.h"
#include "parquet/types.h"
#include "parquet/util/memory.h"

namespace parquet {

namespace BitUtil = ::arrow::BitUtil;

class ColumnDescriptor;

Encoder::Encoder(const ColumnDescriptor* descr, Encoding::type encoding,
                 ::arrow::MemoryPool* pool)
    : descr_(descr),
      encoding_(encoding),
      pool_(pool),
      type_length_(descr ? descr->type_length() : -1) {}

// ----------------------------------------------------------------------
// Encoding::PLAIN encoder implementation

PlainBooleanEncoder::PlainBooleanEncoder(const ColumnDescriptor* descr,
                                         ::arrow::MemoryPool* pool)
    : TypedEncoder<BooleanType>(descr, Encoding::PLAIN, pool),
      bits_available_(kInMemoryDefaultCapacity * 8),
      bits_buffer_(AllocateBuffer(pool, kInMemoryDefaultCapacity)),
      values_sink_(new InMemoryOutputStream(pool)) {
  bit_writer_.reset(new BitUtil::BitWriter(bits_buffer_->mutable_data(),
                                           static_cast<int>(bits_buffer_->size())));
}

int64_t PlainBooleanEncoder::EstimatedDataEncodedSize() {
  return values_sink_->Tell() + bit_writer_->bytes_written();
}

std::shared_ptr<Buffer> PlainBooleanEncoder::FlushValues() {
  if (bits_available_ > 0) {
    bit_writer_->Flush();
    values_sink_->Write(bit_writer_->buffer(), bit_writer_->bytes_written());
    bit_writer_->Clear();
    bits_available_ = static_cast<int>(bits_buffer_->size()) * 8;
  }

  std::shared_ptr<Buffer> buffer = values_sink_->GetBuffer();
  values_sink_.reset(new InMemoryOutputStream(this->pool_));
  return buffer;
}

void PlainBooleanEncoder::Put(const bool* src, int num_values) {
  PutImpl(src, num_values);
}

void PlainBooleanEncoder::Put(const std::vector<bool>& src, int num_values) {
  PutImpl(src, num_values);
}

template class PARQUET_TEMPLATE_EXPORT PlainEncoder<Int32Type>;
template class PARQUET_TEMPLATE_EXPORT PlainEncoder<Int64Type>;
template class PARQUET_TEMPLATE_EXPORT PlainEncoder<Int96Type>;
template class PARQUET_TEMPLATE_EXPORT PlainEncoder<FloatType>;
template class PARQUET_TEMPLATE_EXPORT PlainEncoder<DoubleType>;
template class PARQUET_TEMPLATE_EXPORT PlainEncoder<ByteArrayType>;
template class PARQUET_TEMPLATE_EXPORT PlainEncoder<FLBAType>;

// ----------------------------------------------------------------------
// DictEncoder implementations

DictEncoder::DictEncoder() : dict_encoded_size_(0) {}

int DictEncoder::WriteIndices(uint8_t* buffer, int buffer_len) {
  // Write bit width in first byte
  *buffer = static_cast<uint8_t>(bit_width());
  ++buffer;
  --buffer_len;

  ::arrow::util::RleEncoder encoder(buffer, buffer_len, bit_width());
  for (int index : buffered_indices_) {
    if (!encoder.Put(index)) return -1;
  }
  encoder.Flush();

  ClearIndices();
  return 1 + encoder.len();
}

template class PARQUET_TEMPLATE_EXPORT DictEncoderImpl<BooleanType>;
template class PARQUET_TEMPLATE_EXPORT DictEncoderImpl<Int32Type>;
template class PARQUET_TEMPLATE_EXPORT DictEncoderImpl<Int64Type>;
template class PARQUET_TEMPLATE_EXPORT DictEncoderImpl<Int96Type>;
template class PARQUET_TEMPLATE_EXPORT DictEncoderImpl<FloatType>;
template class PARQUET_TEMPLATE_EXPORT DictEncoderImpl<DoubleType>;
template class PARQUET_TEMPLATE_EXPORT DictEncoderImpl<ByteArrayType>;
template class PARQUET_TEMPLATE_EXPORT DictEncoderImpl<FLBAType>;

// ----------------------------------------------------------------------
// Encoder and decoder factory functions

std::unique_ptr<Encoder> MakeEncoder(Type::type type_num, Encoding::type encoding,
                                     bool use_dictionary, const ColumnDescriptor* descr,
                                     ::arrow::MemoryPool* pool) {
  if (use_dictionary) {
    switch (type_num) {
      case Type::INT32:
        return std::unique_ptr<Encoder>(new DictInt32Encoder(descr, pool));
      case Type::INT64:
        return std::unique_ptr<Encoder>(new DictInt64Encoder(descr, pool));
      case Type::INT96:
        return std::unique_ptr<Encoder>(new DictInt96Encoder(descr, pool));
      case Type::FLOAT:
        return std::unique_ptr<Encoder>(new DictFloatEncoder(descr, pool));
      case Type::DOUBLE:
        return std::unique_ptr<Encoder>(new DictDoubleEncoder(descr, pool));
      case Type::BYTE_ARRAY:
        return std::unique_ptr<Encoder>(new DictByteArrayEncoder(descr, pool));
      case Type::FIXED_LEN_BYTE_ARRAY:
        return std::unique_ptr<Encoder>(new DictFLBAEncoder(descr, pool));
      default:
        DCHECK(false) << "Encoder not implemented";
        break;
    }
  } else if (encoding == Encoding::PLAIN) {
    switch (type_num) {
      case Type::BOOLEAN:
        return std::unique_ptr<Encoder>(new PlainBooleanEncoder(descr, pool));
      case Type::INT32:
        return std::unique_ptr<Encoder>(new PlainInt32Encoder(descr, pool));
      case Type::INT64:
        return std::unique_ptr<Encoder>(new PlainInt64Encoder(descr, pool));
      case Type::INT96:
        return std::unique_ptr<Encoder>(new PlainInt96Encoder(descr, pool));
      case Type::FLOAT:
        return std::unique_ptr<Encoder>(new PlainFloatEncoder(descr, pool));
      case Type::DOUBLE:
        return std::unique_ptr<Encoder>(new PlainDoubleEncoder(descr, pool));
      case Type::BYTE_ARRAY:
        return std::unique_ptr<Encoder>(new PlainByteArrayEncoder(descr, pool));
      case Type::FIXED_LEN_BYTE_ARRAY:
        return std::unique_ptr<Encoder>(new PlainFLBAEncoder(descr, pool));
      default:
        DCHECK(false) << "Encoder not implemented";
        break;
    }
  } else {
    ParquetException::NYI("Selected encoding is not supported");
  }
  DCHECK(false) << "Should not be able to reach this code";
  return nullptr;
}

}  // namespace parquet
