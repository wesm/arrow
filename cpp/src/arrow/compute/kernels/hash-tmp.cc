// TODO(ARROW-1176): Use Tensorflow's StringPiece instead of this here.
struct WrappedBinary {
  WrappedBinary(const uint8_t* ptr, int32_t length) : ptr_(ptr), length_(length) {}

  const uint8_t* ptr_;
  int32_t length_;
};

template <typename T>
struct DictionaryScalar {
  using type = typename T::c_type;
};

template <>
struct DictionaryScalar<BinaryType> {
  using type = WrappedBinary;
};

template <>
struct DictionaryScalar<StringType> {
  using type = WrappedBinary;
};

template <>
struct DictionaryScalar<FixedSizeBinaryType> {
  using type = uint8_t const*;
};

}  // namespace internal

typename TypeTraits<T>::BuilderType dict_builder_;
}
;

class ARROW_EXPORT BinaryUniqueBuilder : public UniqueBuilder<BinaryType> {
 public:
  using UniqueBuilder::Append;
  using UniqueBuilder::UniqueBuilder;

  Status Append(const uint8_t* value, int32_t length) {
    return Append(internal::WrappedBinary(value, length));
  }

  Status Append(const uint8_t* value, int32_t length, int32_t* index) {
    return Append(internal::WrappedBinary(value, length), index);
  }

  Status Append(const char* value, int32_t length) {
    return Append(
        internal::WrappedBinary(reinterpret_cast<const uint8_t*>(value), length));
  }

  Status Append(const char* value, int32_t length, int32_t* index) {
    return Append(
        internal::WrappedBinary(reinterpret_cast<const uint8_t*>(value), length), index);
  }

  Status Append(const std::string& value) {
    return Append(internal::WrappedBinary(reinterpret_cast<const uint8_t*>(value.c_str()),
                                          static_cast<int32_t>(value.size())));
  }

  Status Append(const std::string& value, int32_t* index) {
    return Append(internal::WrappedBinary(reinterpret_cast<const uint8_t*>(value.c_str()),
                                          static_cast<int32_t>(value.size())),
                  index);
  }
};

/// \brief Unique array builder with convenience methods for strings
class ARROW_EXPORT StringUniqueBuilder : public UniqueBuilder<StringType> {
 public:
  using UniqueBuilder::Append;
  using UniqueBuilder::UniqueBuilder;

  Status Append(const uint8_t* value, int32_t length) {
    return Append(internal::WrappedBinary(value, length));
  }

  Status Append(const uint8_t* value, int32_t length, int32_t* index) {
    return Append(internal::WrappedBinary(value, length), index);
  }

  Status Append(const char* value, int32_t length) {
    return Append(
        internal::WrappedBinary(reinterpret_cast<const uint8_t*>(value), length));
  }

  Status Append(const char* value, int32_t length, int32_t* index) {
    return Append(
        internal::WrappedBinary(reinterpret_cast<const uint8_t*>(value), length), index);
  }

  Status Append(const std::string& value) {
    return Append(internal::WrappedBinary(reinterpret_cast<const uint8_t*>(value.c_str()),
                                          static_cast<int32_t>(value.size())));
  }

  Status Append(const std::string& value, int32_t* index) {
    return Append(internal::WrappedBinary(reinterpret_cast<const uint8_t*>(value.c_str()),
                                          static_cast<int32_t>(value.size())),
                  index);
  }
};

template <>
class ARROW_EXPORT DictionaryBuilder<NullType> : public ArrayBuilder {
 public:
  ~DictionaryBuilder();

  DictionaryBuilder(const std::shared_ptr<DataType>& type, MemoryPool* pool);
  explicit DictionaryBuilder(MemoryPool* pool);

  /// \brief Append a scalar null value
  Status AppendNull();

  /// \brief Append a whole dense array to the builder
  Status AppendArray(const Array& array);

  Status Init(int64_t elements) override;
  Status Resize(int64_t capacity) override;
  Status FinishInternal(std::shared_ptr<ArrayData>* out) override;

 protected:
  AdaptiveIntBuilder values_builder_;
};

class ARROW_EXPORT BinaryDictionaryBuilder : public DictionaryBuilder<BinaryType> {
 public:
  using DictionaryBuilder::Append;
  using DictionaryBuilder::DictionaryBuilder;

  Status Append(const uint8_t* value, int32_t length) {
    return Append(internal::WrappedBinary(value, length));
  }

  Status Append(const char* value, int32_t length) {
    return Append(
        internal::WrappedBinary(reinterpret_cast<const uint8_t*>(value), length));
  }

  Status Append(const std::string& value) {
    return Append(internal::WrappedBinary(reinterpret_cast<const uint8_t*>(value.c_str()),
                                          static_cast<int32_t>(value.size())));
  }
};

/// \brief Dictionary array builder with convenience methods for strings
class ARROW_EXPORT StringDictionaryBuilder : public DictionaryBuilder<StringType> {
 public:
  using DictionaryBuilder::Append;
  using DictionaryBuilder::DictionaryBuilder;

  Status Append(const uint8_t* value, int32_t length) {
    return Append(internal::WrappedBinary(value, length));
  }

  Status Append(const char* value, int32_t length) {
    return Append(
        internal::WrappedBinary(reinterpret_cast<const uint8_t*>(value), length));
  }

  Status Append(const std::string& value) {
    return Append(internal::WrappedBinary(reinterpret_cast<const uint8_t*>(value.c_str()),
                                          static_cast<int32_t>(value.size())));
  }
};

// ----------------------------------------------------------------------
// UniqueBuilder

template <>
UniqueBuilder<FixedSizeBinaryType>::UniqueBuilder(const std::shared_ptr<DataType>& type,
                                                  MemoryPool* pool)
    : ArrayBuilder(type, pool),
      hash_table_(new PoolBuffer(pool)),
      hash_slots_(nullptr),
      dict_builder_(type, pool),
      byte_width_(static_cast<const FixedSizeBinaryType&>(*type).byte_width()) {
  if (!::arrow::CpuInfo::initialized()) {
    ::arrow::CpuInfo::Init();
  }
}

template <typename T>
Status UniqueBuilder<T>::DoubleTableSize() {
  int new_size = hash_table_size_ * 2;
  auto new_hash_table = std::make_shared<PoolBuffer>(pool_);

  RETURN_NOT_OK(new_hash_table->Resize(sizeof(hash_slot_t) * new_size));
  int32_t* new_hash_slots = reinterpret_cast<int32_t*>(new_hash_table->mutable_data());
  std::fill(new_hash_slots, new_hash_slots + new_size, kHashSlotEmpty);
  int new_mod_bitmask = new_size - 1;

  for (int i = 0; i < hash_table_size_; ++i) {
    hash_slot_t index = hash_slots_[i];

    if (index == kHashSlotEmpty) {
      continue;
    }

    // Compute the hash value mod the new table size to start looking for an
    // empty slot
    Scalar value = GetDictionaryValue(static_cast<int64_t>(index));

    // Find an empty slot in the new hash table
    int j = HashValue(value) & new_mod_bitmask;
    hash_slot_t slot = new_hash_slots[j];

    while (kHashSlotEmpty != slot && SlotDifferent(slot, value)) {
      ++j;
      if (j == new_size) {
        j = 0;
      }
      slot = new_hash_slots[j];
    }

    // Copy the old slot index to the new hash table
    new_hash_slots[j] = index;
  }

  hash_table_ = new_hash_table;
  hash_slots_ = reinterpret_cast<int32_t*>(hash_table_->mutable_data());
  hash_table_size_ = new_size;
  mod_bitmask_ = new_size - 1;

  return Status::OK();
}

template <typename T>
typename UniqueBuilder<T>::Scalar UniqueBuilder<T>::GetDictionaryValue(int64_t index) {
  const Scalar* data = reinterpret_cast<const Scalar*>(dict_builder_.data()->data());
  return data[index];
}

template <>
const uint8_t* UniqueBuilder<FixedSizeBinaryType>::GetDictionaryValue(int64_t index) {
  return dict_builder_.GetValue(index);
}

template <typename T>
int UniqueBuilder<T>::HashValue(const Scalar& value) {
  return HashUtil::Hash(&value, sizeof(Scalar), 0);
}

template <>
int UniqueBuilder<FixedSizeBinaryType>::HashValue(const Scalar& value) {
  return HashUtil::Hash(value, byte_width_, 0);
}

template <typename T>
bool UniqueBuilder<T>::SlotDifferent(hash_slot_t index, const Scalar& value) {
  const Scalar other = GetDictionaryValue(static_cast<int64_t>(index));
  return other != value;
}

template <>
bool UniqueBuilder<FixedSizeBinaryType>::SlotDifferent(hash_slot_t index,
                                                       const Scalar& value) {
  int32_t width = static_cast<const FixedSizeBinaryType&>(*type_).byte_width();
  const Scalar other = GetDictionaryValue(static_cast<int64_t>(index));
  return memcmp(other, value, width) != 0;
}

template <typename T>
Status UniqueBuilder<T>::AppendDictionary(const Scalar& value) {
  return dict_builder_.Append(value);
}

#define BINARY_UNIQUE_SPECIALIZATIONS(Type)                                         \
  template <>                                                                       \
  Status UniqueBuilder<Type>::AppendArray(const Array& array) {                     \
    const BinaryArray& binary_array = static_cast<const BinaryArray&>(array);       \
    WrappedBinary value(nullptr, 0);                                                \
    for (int64_t i = 0; i < array.length(); i++) {                                  \
      if (!array.IsNull(i)) {                                                       \
        value.ptr_ = binary_array.GetValue(i, &value.length_);                      \
        RETURN_NOT_OK(Append(value));                                               \
      }                                                                             \
    }                                                                               \
    return Status::OK();                                                            \
  }                                                                                 \
                                                                                    \
  template <>                                                                       \
  WrappedBinary UniqueBuilder<Type>::GetDictionaryValue(int64_t index) {            \
    int32_t v_len;                                                                  \
    const uint8_t* v = dict_builder_.GetValue(static_cast<int64_t>(index), &v_len); \
    return WrappedBinary(v, v_len);                                                 \
  }                                                                                 \
                                                                                    \
  template <>                                                                       \
  Status UniqueBuilder<Type>::AppendDictionary(const WrappedBinary& value) {        \
    return dict_builder_.Append(value.ptr_, value.length_);                         \
  }                                                                                 \
                                                                                    \
  template <>                                                                       \
  int UniqueBuilder<Type>::HashValue(const WrappedBinary& value) {                  \
    return HashUtil::Hash(value.ptr_, value.length_, 0);                            \
  }                                                                                 \
                                                                                    \
  template <>                                                                       \
  bool UniqueBuilder<Type>::SlotDifferent(hash_slot_t index,                        \
                                          const WrappedBinary& value) {             \
    int32_t other_length;                                                           \
    const uint8_t* other_value =                                                    \
        dict_builder_.GetValue(static_cast<int64_t>(index), &other_length);         \
    return !(other_length == value.length_ &&                                       \
             0 == memcmp(other_value, value.ptr_, value.length_));                  \
  }

BINARY_UNIQUE_SPECIALIZATIONS(StringType);
BINARY_UNIQUE_SPECIALIZATIONS(BinaryType);

template <typename T>
Status UniqueBuilder<T>::FinishInternal(std::shared_ptr<ArrayData>* out) {
  return dict_builder_.FinishInternal(out);
}

template class UniqueBuilder<UInt8Type>;
template class UniqueBuilder<UInt16Type>;
template class UniqueBuilder<UInt32Type>;
template class UniqueBuilder<UInt64Type>;
template class UniqueBuilder<Int8Type>;
template class UniqueBuilder<Int16Type>;
template class UniqueBuilder<Int32Type>;
template class UniqueBuilder<Int64Type>;
template class UniqueBuilder<Date32Type>;
template class UniqueBuilder<Date64Type>;
template class UniqueBuilder<Time32Type>;
template class UniqueBuilder<Time64Type>;
template class UniqueBuilder<TimestampType>;
template class UniqueBuilder<FloatType>;
template class UniqueBuilder<DoubleType>;
template class UniqueBuilder<FixedSizeBinaryType>;
template class UniqueBuilder<BinaryType>;
template class UniqueBuilder<StringType>;

// ----------------------------------------------------------------------
// DictionaryBuilder

template <typename T>
DictionaryBuilder<T>::DictionaryBuilder(const std::shared_ptr<DataType>& type,
                                        MemoryPool* pool)
    : ArrayBuilder(type, pool), unique_builder_(type, pool), values_builder_(pool) {}

DictionaryBuilder<NullType>::DictionaryBuilder(const std::shared_ptr<DataType>& type,
                                               MemoryPool* pool)
    : ArrayBuilder(type, pool), values_builder_(pool) {}

DictionaryBuilder<NullType>::~DictionaryBuilder() {}

template <>
DictionaryBuilder<FixedSizeBinaryType>::DictionaryBuilder(
    const std::shared_ptr<DataType>& type, MemoryPool* pool)
    : ArrayBuilder(type, pool), unique_builder_(type, pool), values_builder_(pool) {}

template <typename T>
Status DictionaryBuilder<T>::Init(int64_t elements) {
  RETURN_NOT_OK(ArrayBuilder::Init(elements));
  RETURN_NOT_OK(unique_builder_.Init(elements));
  return values_builder_.Init(elements);
}

Status DictionaryBuilder<NullType>::Init(int64_t elements) {
  RETURN_NOT_OK(ArrayBuilder::Init(elements));
  return values_builder_.Init(elements);
}

template <typename T>
Status DictionaryBuilder<T>::Resize(int64_t capacity) {
  if (capacity < kMinBuilderCapacity) {
    capacity = kMinBuilderCapacity;
  }

  if (capacity_ == 0) {
    return Init(capacity);
  } else {
    RETURN_NOT_OK(unique_builder_.Resize(capacity));
    return ArrayBuilder::Resize(capacity);
  }
}

Status DictionaryBuilder<NullType>::Resize(int64_t capacity) {
  if (capacity < kMinBuilderCapacity) {
    capacity = kMinBuilderCapacity;
  }

  if (capacity_ == 0) {
    return Init(capacity);
  } else {
    return ArrayBuilder::Resize(capacity);
  }
}

template <typename T>
Status DictionaryBuilder<T>::FinishInternal(std::shared_ptr<ArrayData>* out) {
  std::shared_ptr<Array> dictionary;
  RETURN_NOT_OK(unique_builder_.Finish(&dictionary));

  RETURN_NOT_OK(values_builder_.FinishInternal(out));
  (*out)->type = std::make_shared<DictionaryType>((*out)->type, dictionary);
  return Status::OK();
}

Status DictionaryBuilder<NullType>::FinishInternal(std::shared_ptr<ArrayData>* out) {
  std::shared_ptr<Array> dictionary = std::make_shared<NullArray>(0);

  RETURN_NOT_OK(values_builder_.FinishInternal(out));
  (*out)->type = std::make_shared<DictionaryType>((*out)->type, dictionary);
  return Status::OK();
}

template <typename T>
Status DictionaryBuilder<T>::Append(const Scalar& value) {
  RETURN_NOT_OK(Reserve(1));
  int32_t index;
  RETURN_NOT_OK(unique_builder_.Append(value, &index));
  return values_builder_.Append(index);
}

template <typename T>
Status DictionaryBuilder<T>::AppendArray(const Array& array) {
  const auto& numeric_array = static_cast<const NumericArray<T>&>(array);
  for (int64_t i = 0; i < array.length(); i++) {
    if (array.IsNull(i)) {
      RETURN_NOT_OK(AppendNull());
    } else {
      RETURN_NOT_OK(Append(numeric_array.Value(i)));
    }
  }
  return Status::OK();
}

Status DictionaryBuilder<NullType>::AppendArray(const Array& array) {
  for (int64_t i = 0; i < array.length(); i++) {
    RETURN_NOT_OK(AppendNull());
  }
  return Status::OK();
}

template <>
Status DictionaryBuilder<FixedSizeBinaryType>::AppendArray(const Array& array) {
  if (!type_->Equals(*array.type())) {
    return Status::Invalid("Cannot append FixedSizeBinary array with non-matching type");
  }

  const auto& numeric_array = static_cast<const FixedSizeBinaryArray&>(array);
  for (int64_t i = 0; i < array.length(); i++) {
    if (array.IsNull(i)) {
      RETURN_NOT_OK(AppendNull());
    } else {
      RETURN_NOT_OK(Append(numeric_array.Value(i)));
    }
  }
  return Status::OK();
}

template <typename T>
Status DictionaryBuilder<T>::AppendNull() {
  return values_builder_.AppendNull();
}

Status DictionaryBuilder<NullType>::AppendNull() { return values_builder_.AppendNull(); }

#define BINARY_DICTIONARY_SPECIALIZATIONS(Type)                               \
  template <>                                                                 \
  Status DictionaryBuilder<Type>::AppendArray(const Array& array) {           \
    const BinaryArray& binary_array = static_cast<const BinaryArray&>(array); \
    WrappedBinary value(nullptr, 0);                                          \
    for (int64_t i = 0; i < array.length(); i++) {                            \
      if (array.IsNull(i)) {                                                  \
        RETURN_NOT_OK(AppendNull());                                          \
      } else {                                                                \
        value.ptr_ = binary_array.GetValue(i, &value.length_);                \
        RETURN_NOT_OK(Append(value));                                         \
      }                                                                       \
    }                                                                         \
    return Status::OK();                                                      \
  }

BINARY_DICTIONARY_SPECIALIZATIONS(StringType);
BINARY_DICTIONARY_SPECIALIZATIONS(BinaryType);
