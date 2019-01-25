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

#include "parquet/exception.h"
#include "parquet/schema.h"
#include "parquet/types.h"
#include "parquet/util/memory.h"

namespace parquet {

namespace BitUtil = ::arrow::BitUtil;

class ColumnDescriptor;

class EncoderImpl : virtual public Encoder {
 public:
  EncoderImpl(const ColumnDescriptor* descr, Encoding::type encoding,
              ::arrow::MemoryPool* pool)
      : descr_(descr),
        encoding_(encoding),
        pool_(pool),
        type_length_(descr ? descr->type_length() : -1) {}

  Encoding::type encoding() const override { return encoding_; }

  ::arrow::MemoryPool* memory_pool() const override { return pool_; }

 protected:
  // For accessing type-specific metadata, like FIXED_LEN_BYTE_ARRAY
  const ColumnDescriptor* descr_;
  const Encoding::type encoding_;
  ::arrow::MemoryPool* pool_;

  /// Type length from descr
  int type_length_;
};

// ----------------------------------------------------------------------
// Plain encoder implementation

template <typename DType>
class PlainEncoder : public EncoderImpl, virtual public TypedEncoder<DType> {
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

template <typename DType>
PlainEncoder<DType>::PlainEncoder(const ColumnDescriptor* descr,
                                  ::arrow::MemoryPool* pool)
    : EncoderImpl(descr, Encoding::PLAIN, pool) {
  values_sink_.reset(new InMemoryOutputStream(pool));
}
template <typename DType>
int64_t PlainEncoder<DType>::EstimatedDataEncodedSize() {
  return values_sink_->Tell();
}

template <typename DType>
std::shared_ptr<Buffer> PlainEncoder<DType>::FlushValues() {
  std::shared_ptr<Buffer> buffer = values_sink_->GetBuffer();
  values_sink_.reset(new InMemoryOutputStream(this->pool_));
  return buffer;
}

template <typename DType>
void PlainEncoder<DType>::Put(const T* buffer, int num_values) {
  values_sink_->Write(reinterpret_cast<const uint8_t*>(buffer), num_values * sizeof(T));
}

template <>
inline void PlainEncoder<ByteArrayType>::Put(const ByteArray* src, int num_values) {
  for (int i = 0; i < num_values; ++i) {
    // Write the result to the output stream
    values_sink_->Write(reinterpret_cast<const uint8_t*>(&src[i].len), sizeof(uint32_t));
    if (src[i].len > 0) {
      DCHECK(nullptr != src[i].ptr) << "Value ptr cannot be NULL";
    }
    values_sink_->Write(reinterpret_cast<const uint8_t*>(src[i].ptr), src[i].len);
  }
}

template <>
inline void PlainEncoder<FLBAType>::Put(const FixedLenByteArray* src, int num_values) {
  for (int i = 0; i < num_values; ++i) {
    // Write the result to the output stream
    if (descr_->type_length() > 0) {
      DCHECK(nullptr != src[i].ptr) << "Value ptr cannot be NULL";
    }
    values_sink_->Write(reinterpret_cast<const uint8_t*>(src[i].ptr),
                        descr_->type_length());
  }
}

class PlainByteArrayEncoder : public PlainEncoder<ByteArrayType>,
                              virtual public ByteArrayEncoder {
 public:
  using BASE = PlainEncoder<ByteArrayType>;
  using BASE::PlainEncoder;
};

class PlainFLBAEncoder : public PlainEncoder<FLBAType>, virtual public FLBAEncoder {
 public:
  using BASE = PlainEncoder<FLBAType>;
  using BASE::PlainEncoder;
};

class PlainBooleanEncoder : public EncoderImpl,
                            virtual public TypedEncoder<BooleanType>,
                            virtual public BooleanEncoder {
 public:
  explicit PlainBooleanEncoder(
      const ColumnDescriptor* descr,
      ::arrow::MemoryPool* pool = ::arrow::default_memory_pool());

  int64_t EstimatedDataEncodedSize() override;
  std::shared_ptr<Buffer> FlushValues() override;

  void Put(const bool* src, int num_values) override;
  void Put(const std::vector<bool>& src, int num_values) override;

 private:
  int bits_available_;
  std::unique_ptr<::arrow::BitUtil::BitWriter> bit_writer_;
  std::shared_ptr<ResizableBuffer> bits_buffer_;
  std::unique_ptr<InMemoryOutputStream> values_sink_;

  template <typename SequenceType>
  void PutImpl(const SequenceType& src, int num_values);
};

template <typename SequenceType>
void PlainBooleanEncoder::PutImpl(const SequenceType& src, int num_values) {
  int bit_offset = 0;
  if (bits_available_ > 0) {
    int bits_to_write = std::min(bits_available_, num_values);
    for (int i = 0; i < bits_to_write; i++) {
      bit_writer_->PutValue(src[i], 1);
    }
    bits_available_ -= bits_to_write;
    bit_offset = bits_to_write;

    if (bits_available_ == 0) {
      bit_writer_->Flush();
      values_sink_->Write(bit_writer_->buffer(), bit_writer_->bytes_written());
      bit_writer_->Clear();
    }
  }

  int bits_remaining = num_values - bit_offset;
  while (bit_offset < num_values) {
    bits_available_ = static_cast<int>(bits_buffer_->size()) * 8;

    int bits_to_write = std::min(bits_available_, bits_remaining);
    for (int i = bit_offset; i < bit_offset + bits_to_write; i++) {
      bit_writer_->PutValue(src[i], 1);
    }
    bit_offset += bits_to_write;
    bits_available_ -= bits_to_write;
    bits_remaining -= bits_to_write;

    if (bits_available_ == 0) {
      bit_writer_->Flush();
      values_sink_->Write(bit_writer_->buffer(), bit_writer_->bytes_written());
      bit_writer_->Clear();
    }
  }
}

PlainBooleanEncoder::PlainBooleanEncoder(const ColumnDescriptor* descr,
                                         ::arrow::MemoryPool* pool)
    : EncoderImpl(descr, Encoding::PLAIN, pool),
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

// ----------------------------------------------------------------------
// DictEncoder implementations

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

/// See the dictionary encoding section of https://github.com/Parquet/parquet-format.
/// The encoding supports streaming encoding. Values are encoded as they are added while
/// the dictionary is being constructed. At any time, the buffered values can be
/// written out with the current dictionary size. More values can then be added to
/// the encoder, including new dictionary entries.
template <typename DType>
class DictEncoderImpl : public EncoderImpl,
                        public DictEncoder,
                        virtual public TypedEncoder<DType> {
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
  void WriteDict(uint8_t* buffer) override;

  /// The number of entries in the dictionary.
  int num_entries() const override { return memo_table_.size(); }

 private:
  MemoTableType memo_table_;
};

// Initially 1024 elements
static constexpr int32_t INITIAL_HASH_TABLE_SIZE = 1 << 10;

template <typename DType>
DictEncoderImpl<DType>::DictEncoderImpl(const ColumnDescriptor* desc,
                                        ::arrow::MemoryPool* pool)
    : EncoderImpl(desc, Encoding::PLAIN_DICTIONARY, pool),
      memo_table_(INITIAL_HASH_TABLE_SIZE) {}

template <typename DType>
int64_t DictEncoderImpl<DType>::EstimatedDataEncodedSize() {
  // Note: because of the way RleEncoder::CheckBufferFull() is called, we have to
  // reserve
  // an extra "RleEncoder::MinBufferSize" bytes. These extra bytes won't be used
  // but not reserving them would cause the encoder to fail.
  return 1 +
         ::arrow::util::RleEncoder::MaxBufferSize(
             bit_width(), static_cast<int>(buffered_indices_.size())) +
         ::arrow::util::RleEncoder::MinBufferSize(bit_width());
}

template <typename DType>
int DictEncoderImpl<DType>::bit_width() const {
  if (ARROW_PREDICT_FALSE(num_entries() == 0)) return 0;
  if (ARROW_PREDICT_FALSE(num_entries() == 1)) return 1;
  return BitUtil::Log2(num_entries());
}

template <typename DType>
std::shared_ptr<Buffer> DictEncoderImpl<DType>::FlushValues() {
  std::shared_ptr<ResizableBuffer> buffer =
      AllocateBuffer(this->pool_, EstimatedDataEncodedSize());
  int result_size =
      WriteIndices(buffer->mutable_data(), static_cast<int>(EstimatedDataEncodedSize()));
  PARQUET_THROW_NOT_OK(buffer->Resize(result_size, false));
  return buffer;
}

template <typename DType>
void DictEncoderImpl<DType>::Put(const T* src, int num_values) {
  for (int32_t i = 0; i < num_values; i++) {
    Put(src[i]);
  }
}

template <typename DType>
void DictEncoderImpl<DType>::PutSpaced(const T* src, int num_values,
                                       const uint8_t* valid_bits,
                                       int64_t valid_bits_offset) {
  ::arrow::internal::BitmapReader valid_bits_reader(valid_bits, valid_bits_offset,
                                                    num_values);
  for (int32_t i = 0; i < num_values; i++) {
    if (valid_bits_reader.IsSet()) {
      Put(src[i]);
    }
    valid_bits_reader.Next();
  }
}

template <typename DType>
void DictEncoderImpl<DType>::WriteDict(uint8_t* buffer) {
  // For primitive types, only a memcpy
  DCHECK_EQ(static_cast<size_t>(dict_encoded_size_), sizeof(T) * memo_table_.size());
  memo_table_.CopyValues(0 /* start_pos */, reinterpret_cast<T*>(buffer));
}

// ByteArray and FLBA already have the dictionary encoded in their data heaps
template <>
void DictEncoderImpl<ByteArrayType>::WriteDict(uint8_t* buffer) {
  memo_table_.VisitValues(0, [&](const ::arrow::util::string_view& v) {
    uint32_t len = static_cast<uint32_t>(v.length());
    memcpy(buffer, &len, sizeof(uint32_t));
    buffer += sizeof(uint32_t);
    memcpy(buffer, v.data(), v.length());
    buffer += v.length();
  });
}

template <>
void DictEncoderImpl<FLBAType>::WriteDict(uint8_t* buffer) {
  memo_table_.VisitValues(0, [&](const ::arrow::util::string_view& v) {
    DCHECK_EQ(v.length(), static_cast<size_t>(type_length_));
    memcpy(buffer, v.data(), type_length_);
    buffer += type_length_;
  });
}

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

class DictByteArrayEncoder : public DictEncoderImpl<ByteArrayType>,
                             virtual public ByteArrayEncoder {
 public:
  using BASE = DictEncoderImpl<ByteArrayType>;
  using BASE::DictEncoderImpl;
};

class DictFLBAEncoder : public DictEncoderImpl<FLBAType>, virtual public FLBAEncoder {
 public:
  using BASE = DictEncoderImpl<FLBAType>;
  using BASE::DictEncoderImpl;
};

// ----------------------------------------------------------------------
// Encoder and decoder factory functions

std::unique_ptr<Encoder> MakeEncoder(Type::type type_num, Encoding::type encoding,
                                     bool use_dictionary, const ColumnDescriptor* descr,
                                     ::arrow::MemoryPool* pool) {
  if (use_dictionary) {
    switch (type_num) {
      case Type::INT32:
        return std::unique_ptr<Encoder>(new DictEncoderImpl<Int32Type>(descr, pool));
      case Type::INT64:
        return std::unique_ptr<Encoder>(new DictEncoderImpl<Int64Type>(descr, pool));
      case Type::INT96:
        return std::unique_ptr<Encoder>(new DictEncoderImpl<Int96Type>(descr, pool));
      case Type::FLOAT:
        return std::unique_ptr<Encoder>(new DictEncoderImpl<FloatType>(descr, pool));
      case Type::DOUBLE:
        return std::unique_ptr<Encoder>(new DictEncoderImpl<DoubleType>(descr, pool));
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
        return std::unique_ptr<Encoder>(new PlainEncoder<Int32Type>(descr, pool));
      case Type::INT64:
        return std::unique_ptr<Encoder>(new PlainEncoder<Int64Type>(descr, pool));
      case Type::INT96:
        return std::unique_ptr<Encoder>(new PlainEncoder<Int96Type>(descr, pool));
      case Type::FLOAT:
        return std::unique_ptr<Encoder>(new PlainEncoder<FloatType>(descr, pool));
      case Type::DOUBLE:
        return std::unique_ptr<Encoder>(new PlainEncoder<DoubleType>(descr, pool));
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
