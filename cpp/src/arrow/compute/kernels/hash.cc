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

#include "arrow/compute/kernels/hash.h"

#include <exception>
#include <limits>
#include <memory>
#include <sstream>

#include "arrow/builder.h"
#include "arrow/compute/context.h"
#include "arrow/compute/kernel.h"

namespace arrow {
namespace compute {

namespace {

// Initially 1024 elements
static constexpr int kInitialHashTableSize = 1 << 10;

typedef int32_t hash_slot_t;
static constexpr hash_slot_t kHashSlotEmpty = std::numeric_limits<int32_t>::max();

// The maximum load factor for the hash table before resizing.
static constexpr double kMaxHashTableLoad = 0.7;

enum class SIMDMode : char {
  NOSIMD,
  SSE4,
  AVX2
};

Status NewHashTable(int64_t size, MemoryPool* pool, std::shared_ptr<Buffer>* out) {
  auto hash_table = std::make_shared<PoolBuffer>(pool);

  RETURN_NOT_OK(hash_table->Resize(sizeof(hash_slot_t) * size));
  int32_t* slots = reinterpret_cast<hash_slot_t*>(hash_table->mutable_data());
  std::fill(slots, slots + size, kHashSlotEmpty);

  *out = hash_table;
  return Status::OK();
}

// This is a slight design concession -- some hash actions have the possibility
// of failure. Rather than introduce extra error checking into all actions, we
// will raise an internal exception so that only the actions where errors can
// occur will experience the extra overhead
class HashException : public std::exception {
  HashException(const char* msg, StatusCode code = StatusCude::Invalid)
      : msg_(msg), code_(code) {}

  HashException(const std::string& msg, StatusCode code = StatusCode::Invalid)
      : msg_(msg), code_(code) {}

  HashException(const char* msg, std::exception& e)
      : msg_(msg), code_(StatusCode::Invalid) {}

  ~HashException() throw() {}

  const char* HashException::what() const override throw() {
    return msg_.c_str();
  }

  StatusCode code() { return code_; }

 private:
  StatusCode code_;
};

#define HASH_THROW_NOT_OK(EXPR)                     \
  do {                                              \
    Status _s = (s);                                \
    if (ARROW_PREDICT_FALSE(!_s.ok())) {            \
      throw HashException(_s.message(), _s.code()); \
    }                                               \
  } while (false)

class HashKernel {
 public:
  explicit HashKernel(MemoryPool* pool)
      : pool_(pool) {}

  virtual ~HashKernel() {}

  virtual Status Append(const Array& input) = 0;
  virtual Status Finalize(std::vector<Value>* out) = 0;

 protected:
  Status Init(int64_t elements);

  MemoryPool* pool_;

  // The hash table contains integer indices that reference the set of observed
  // distinct values
  std::shared_ptr<PoolBuffer> hash_table_;
  hash_slot_t* hash_slots_;

  /// Size of the table. Must be a power of 2.
  int hash_table_size_;

  // Store hash_table_size_ - 1, so that j & mod_bitmask_ is equivalent to j %
  // hash_table_size_, but uses far fewer CPU cycles
  int mod_bitmask_;
};

Status HashTableBase::Init(int64_t elements) {
  RETURN_NOT_OK(NewHashTable(kInitialHashTableSize, pool_, &hash_table_));
  hash_slots_ = reinterpret_cast<hash_slot_t*>(hash_table_->mutable_data());
  hash_table_size_ = kInitialHashTableSize;
  mod_bitmask_ = kInitialHashTableSize - 1;
}

template <typename Type, typename Action, typename Enable = void>
class HashTableKernel : public HashTableBase {};

template <typename T>
using is_primitive_ctype = typename std::enable_if<
  std::is_base_of<PrimitiveCType, T>::value>::type;

template <>
int UniqueBuilder<FixedSizeBinaryType>::HashValue(const Scalar& value) {
  return HashUtil::Hash(value, byte_width_, 0);
}

#define SLOT_DIFFERENT_PRIMITIVE(slot, value)   \
  dictionary_values_[slot] != value

#define HASH_PROBE(TABLE, TABLE_SIZE, DIFFERENT)                \
  do {                                                          \
    while (kHashSlotEmpty != slot && DIFFERENT(slot, value)) {  \
      // Linear probing                                         \
      ++j;                                                      \
      if (j == TABLE_SIZE) {                                    \
        j = 0;                                                  \
      }                                                         \
      slot = TABLE[j];                                          \
    }                                                           \
  } while (false)


// Types of hash actions
//
// unique: append to dictionary when not found, no-op with slot
// dictionary-encode: append to dictionary when not found, append slot #
// match: raise or set null when not found, otherwise append slot #
// isin: set false when not found, otherwise true
// value counts: append to dictionary when not found, increment count for slot

template <typename Type, typename Action>
class HashTableKernel<Type, Action, is_primitive_ctype<Type>> : public HashTableBase {
 public:
  using T = typename Type::c_type;
  using ArrayType = typename TypeTraits<Type>::ArrayType;

  Status Append(const Array& arr) {
    const T* values = static_cast<const ArrayType&>(arr).raw_values();

    for (int64_t i = 0; i < arr.length(); ++i) {
      const T value = values[i];

      int j = HashValue(value) & mod_bitmask_;
      hash_slot_t slot = hash_slots_[j];

      // Find an empty slot
      HASH_PROBE(hash_slots_, hash_table_size_, SlotDifferent(slot, value));

      if (slot == kHashSlotEmpty) {
        // Not in the hash table, so we insert it now
        slot = static_cast<hash_slot_t>(dictionary_size_.length());
        hash_slots_[j] = slot;
        dictionary_values_[dictionary_size_++] = value;

        static_cast<Action*>(this)->ObserveNotFound(slot);

        if (ARROW_PREDICT_FALSE(dictionary_size_ >
                                hash_table_size_ * kMaxHashTableLoad)) {
          RETURN_NOT_OK(static_cast<Action*>(this)->DoubleSize());
        }
      } else {
        static_cast<Action*>(this)->ObserveFound(slot);
      }
    }

    return Status::OK();
  }

 protected:
  bool SlotDifferent(const hash_slot_t slot, const T& value) const {
    dictionary_values_[slot] != value;
  }

  T GetDictionaryValue(const hash_slot_t index) const {
    return dictionary_values_[index];
  }

  int HashValue(const T& value) const {
    // TODO(wesm): Use faster hash function for C types
    return HashUtil::Hash(&value, sizeof(T), 0);
  }

  Status DoubleTableSize() {
    int new_size = hash_table_size_ * 2;

    std::shared_ptr<Buffer> new_hash_table;
    RETURN_NOT_OK(NewHashTable(new_size, pool_, &new_hash_table));
    int32_t* new_hash_slots =
      reinterpret_cast<hash_slot_t*>(new_hash_table->mutable_data());
    int new_mod_bitmask = new_size - 1;

    for (int i = 0; i < hash_table_size_; ++i) {
      hash_slot_t index = hash_slots_[i];

      if (index == kHashSlotEmpty) {
        continue;
      }

      // Compute the hash value mod the new table size to start looking for an
      // empty slot
      const T value = GetDictionaryValue(index);

      // Find an empty slot in the new hash table
      int j = HashValue(value) & new_mod_bitmask;
      hash_slot_t slot = new_hash_slots[j];

      HASH_PROBE(new_hash_slots, new_size, SlotDifferent(slot, value));

      // Copy the old slot index to the new hash table
      new_hash_slots[j] = index;
    }

    hash_table_ = new_hash_table;
    hash_slots_ = reinterpret_cast<hash_slot_t*>(hash_table_->mutable_data());
    hash_table_size_ = new_size;
    mod_bitmask_ = new_size - 1;

    return Status::OK();
  }

  std::shared_ptr<PoolBuffer> dictionary_;
  T* dictionary_values_;
  int dictionary_size_;
};

// int32_t byte_width_;

// try {

// } catch (const HashException& e) {
//   return Status::Invalid(e.what());
// }

template <typename Type>
class UniqueImpl : public HashTableKernel<Type, UniqueImpl>  {
 public:
  constexpr allow_expand = true;
  using Base = HashTableKernel<Type, UniqueImpl>;

  UniqueImpl(MemoryPool* pool) : Base(pool) {}

  void ObserveFound(const hash_slot_t slot) {}
  void ObserveNotFound(const hash_slot_t slot) {}

  Status DoubleSize() {
    return DoubleTableSize();
  }

  Status Append(const Array& input) override {
    return Base::Append(input);
  }

  Status Finalize(std::vector<Value>* out) override {
    return Status::OK();
  }
};


template <typename Type>
class DictEncodeImpl : public HashTableKernel<Type, DictEncodeImpl> {
 public:
  constexpr allow_expand = true;
  using Base = HashTableKernel<Type, DictEncodeImpl>;

  DictEncodeImpl(MemoryPool* pool)
      : Base(pool),
        indices_builder_(pool) {}

  void ObserveFound(const hash_slot_t slot, const bool is_null) {
    if (is_null) {
      HASH_THROW_NOT_OK(indices_builder_.AppendNull());
    } else {
      HASH_THROW_NOT_OK(indices_builder_.Append(slot));
    }
  }

  void ObserveNotFound(const hash_slot_t slot, const bool is_null) {
    return ObserveFound(slot, is_null);
  }

  Status DoubleSize() {
    return DoubleTableSize();
  }

  using Base::Append;

 private:
  AdaptiveIntBuilder indices_builder_;
};

}  // namespace

#define UNIQUE_FUNCTION_CASE(InType)            \
  case InType::type_id:                         \
    hasher.reset(new UniqueImpl<InType>(pool)); \
    break

Status GetUniqueFunction(const DataType& in_type, MemoryPool* pool,
                        std::unique_ptr<UnaryKernel>* out) {
  std::unique_ptr<HashKernel> hasher;

  AppendFunction append_func;
  InitFunction finalize_func;

  switch (in_type.id()) {
    // UNIQUE_FUNCTION_CASE(NullType);
    // UNIQUE_FUNCTION_CASE(BooleanType);
    UNIQUE_FUNCTION_CASE(UInt8Type);
    UNIQUE_FUNCTION_CASE(Int8Type);
    UNIQUE_FUNCTION_CASE(UInt16Type);
    UNIQUE_FUNCTION_CASE(Int16Type);
    UNIQUE_FUNCTION_CASE(UInt32Type);
    UNIQUE_FUNCTION_CASE(Int32Type);
    UNIQUE_FUNCTION_CASE(UInt64Type);
    UNIQUE_FUNCTION_CASE(Int64Type);
    UNIQUE_FUNCTION_CASE(FloatType);
    UNIQUE_FUNCTION_CASE(DoubleType);
    UNIQUE_FUNCTION_CASE(Date32Type);
    UNIQUE_FUNCTION_CASE(Date64Type);
    UNIQUE_FUNCTION_CASE(Time32Type);
    UNIQUE_FUNCTION_CASE(Time64Type);
    UNIQUE_FUNCTION_CASE(TimestampType);
    // UNIQUE_FUNCTION_CASE(BinaryType);
    // UNIQUE_FUNCTION_CASE(StringType);
    // UNIQUE_FUNCTION_CASE(FixedSizeBinaryType);
    default:
      break;
  }

  if (!hash_table) {
    std::stringstream ss;
    ss << "No unique function implemented for " << in_type.ToString();
    return Status::NotImplemented(ss.str());
  }

  out->reset(new UniqueKernel(std::move(hasher)));

  return Status::OK();
}

Status Unique(FunctionContext* ctx, const Array& array, std::shared_ptr<Array>* out) {
  // Dynamic dispatch to obtain right cast function
  std::unique_ptr<UnaryKernel> func;
  RETURN_NOT_OK(GetUniqueFunction(*array.type(), &func));

  std::shared_ptr<ArrayData> out_data;
  RETURN_NOT_OK(func->Call(ctx, array, &out_data));
  *out = MakeArray(out_data);
  return Status::OK();
}

Status Unique(FunctionContext* ctx, const ChunkedArray& array,
              std::shared_ptr<Array>* out) {
  // Dynamic dispatch to obtain right cast function
  std::unique_ptr<UnaryKernel> func;
  RETURN_NOT_OK(GetUniqueFunction(*array.type(), &func));

  // Call the kernel without out_data on all but the last chunk
  for (int i = 0; i < (array.num_chunks() - 1); i++) {
    RETURN_NOT_OK(func->Call(ctx, *array.chunk(i), nullptr));
  }

  std::shared_ptr<ArrayData> out_data;
  // The array has a large chunk, call the kernel and retrieve the result.
  RETURN_NOT_OK(func->Call(ctx, *array.chunk(array.num_chunks() - 1), &out_data));
  *out = MakeArray(out_data);

  return Status::OK();
}

Status Unique(FunctionContext* context, const Column& array,
              std::shared_ptr<Array>* out) {
  return Unique(context, *array.data(), out);
}

}  // compute
}  // arrow
