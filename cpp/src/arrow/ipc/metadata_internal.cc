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

#include "arrow/ipc/metadata_internal.h"

#include <cstdint>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <utility>

#include <flatbuffers/flatbuffers.h>

#include "arrow/array.h"
#include "arrow/extension_type.h"
#include "arrow/io/interfaces.h"
#include "arrow/ipc/message.h"
#include "arrow/ipc/util.h"
#include "arrow/sparse_tensor.h"
#include "arrow/status.h"
#include "arrow/tensor.h"
#include "arrow/type.h"
#include "arrow/util/checked_cast.h"
#include "arrow/util/key_value_metadata.h"
#include "arrow/util/logging.h"
#include "arrow/util/ubsan.h"
#include "arrow/visitor_inline.h"

#include "generated/File_generated.h"  // IWYU pragma: keep
#include "generated/Message_generated.h"
#include "generated/SparseTensor_generated.h"  // IWYU pragma: keep
#include "generated/Tensor_generated.h"        // IWYU pragma: keep

namespace arrow {

namespace flatbuf = org::apache::arrow::flatbuf;
using internal::checked_cast;

namespace ipc {
namespace internal {

using FBB = flatbuffers::FlatBufferBuilder;
using DictionaryOffset = flatbuffers::Offset<flatbuf::DictionaryEncoding>;
using FieldOffset = flatbuffers::Offset<flatbuf::Field>;
using RecordBatchOffset = flatbuffers::Offset<flatbuf::RecordBatch>;
using SparseTensorOffset = flatbuffers::Offset<flatbuf::SparseTensor>;
using Offset = flatbuffers::Offset<void>;
using FBString = flatbuffers::Offset<flatbuffers::String>;

MetadataVersion GetMetadataVersion(flatbuf::MetadataVersion version) {
  switch (version) {
    case flatbuf::MetadataVersion::V1:
      // Arrow 0.1
      return MetadataVersion::V1;
    case flatbuf::MetadataVersion::V2:
      // Arrow 0.2
      return MetadataVersion::V2;
    case flatbuf::MetadataVersion::V3:
      // Arrow 0.3 to 0.7.1
      return MetadataVersion::V4;
    case flatbuf::MetadataVersion::V4:
      // Arrow >= 0.8
      return MetadataVersion::V4;
    // Add cases as other versions become available
    default:
      return MetadataVersion::V4;
  }
}

namespace {

Status IntFromFlatbuffer(const flatbuf::Int* int_data, std::shared_ptr<DataType>* out) {
  if (int_data->bitWidth() > 64) {
    return Status::NotImplemented("Integers with more than 64 bits not implemented");
  }
  if (int_data->bitWidth() < 8) {
    return Status::NotImplemented("Integers with less than 8 bits not implemented");
  }

  switch (int_data->bitWidth()) {
    case 8:
      *out = int_data->is_signed() ? int8() : uint8();
      break;
    case 16:
      *out = int_data->is_signed() ? int16() : uint16();
      break;
    case 32:
      *out = int_data->is_signed() ? int32() : uint32();
      break;
    case 64:
      *out = int_data->is_signed() ? int64() : uint64();
      break;
    default:
      return Status::NotImplemented("Integers not in cstdint are not implemented");
  }
  return Status::OK();
}

Status FloatFromFlatbuffer(const flatbuf::FloatingPoint* float_data,
                           std::shared_ptr<DataType>* out) {
  if (float_data->precision() == flatbuf::Precision::HALF) {
    *out = float16();
  } else if (float_data->precision() == flatbuf::Precision::SINGLE) {
    *out = float32();
  } else {
    *out = float64();
  }
  return Status::OK();
}

// Forward declaration
Status FieldToFlatbuffer(FBB& fbb, const std::shared_ptr<Field>& field,
                         DictionaryMemo* dictionary_memo, FieldOffset* offset);

Offset IntToFlatbuffer(FBB& fbb, int bitWidth, bool is_signed) {
  return flatbuf::CreateInt(fbb, bitWidth, is_signed).Union();
}

Offset FloatToFlatbuffer(FBB& fbb, flatbuf::Precision precision) {
  return flatbuf::CreateFloatingPoint(fbb, precision).Union();
}

Status AppendChildFields(FBB& fbb, const DataType& type,
                         std::vector<FieldOffset>* out_children,
                         DictionaryMemo* dictionary_memo) {
  FieldOffset field;
  for (int i = 0; i < type.num_children(); ++i) {
    RETURN_NOT_OK(FieldToFlatbuffer(fbb, type.child(i), dictionary_memo, &field));
    out_children->push_back(field);
  }
  return Status::OK();
}

// ----------------------------------------------------------------------
// Union implementation

Status UnionFromFlatbuffer(const flatbuf::Union* union_data,
                           const std::vector<std::shared_ptr<Field>>& children,
                           std::shared_ptr<DataType>* out) {
  UnionMode::type mode =
      (union_data->mode() == flatbuf::UnionMode::Sparse ? UnionMode::SPARSE
                                                        : UnionMode::DENSE);

  std::vector<int8_t> type_codes;

  const flatbuffers::Vector<int32_t>* fb_type_ids = union_data->typeIds();
  if (fb_type_ids == nullptr) {
    for (int8_t i = 0; i < static_cast<int8_t>(children.size()); ++i) {
      type_codes.push_back(i);
    }
  } else {
    for (int32_t id : (*fb_type_ids)) {
      const auto type_code = static_cast<int8_t>(id);
      if (id != type_code) {
        return Status::Invalid("union type id out of bounds");
      }
      type_codes.push_back(type_code);
    }
  }

  return UnionType::Make(children, type_codes, mode).Value(out);
}

#define INT_TO_FB_CASE(BIT_WIDTH, IS_SIGNED)            \
  *out_type = flatbuf::Type::Int;                       \
  *offset = IntToFlatbuffer(fbb, BIT_WIDTH, IS_SIGNED); \
  break;

static inline flatbuf::TimeUnit ToFlatbufferUnit(TimeUnit::type unit) {
  switch (unit) {
    case TimeUnit::SECOND:
      return flatbuf::TimeUnit::SECOND;
    case TimeUnit::MILLI:
      return flatbuf::TimeUnit::MILLISECOND;
    case TimeUnit::MICRO:
      return flatbuf::TimeUnit::MICROSECOND;
    case TimeUnit::NANO:
      return flatbuf::TimeUnit::NANOSECOND;
    default:
      break;
  }
  return flatbuf::TimeUnit::MIN;
}

static inline TimeUnit::type FromFlatbufferUnit(flatbuf::TimeUnit unit) {
  switch (unit) {
    case flatbuf::TimeUnit::SECOND:
      return TimeUnit::SECOND;
    case flatbuf::TimeUnit::MILLISECOND:
      return TimeUnit::MILLI;
    case flatbuf::TimeUnit::MICROSECOND:
      return TimeUnit::MICRO;
    case flatbuf::TimeUnit::NANOSECOND:
      return TimeUnit::NANO;
    default:
      break;
  }
  // cannot reach
  return TimeUnit::SECOND;
}

Status ConcreteTypeFromFlatbuffer(flatbuf::Type type, const void* type_data,
                                  const std::vector<std::shared_ptr<Field>>& children,
                                  std::shared_ptr<DataType>* out) {
  switch (type) {
    case flatbuf::Type::NONE:
      return Status::Invalid("Type metadata cannot be none");
    case flatbuf::Type::Null:
      *out = null();
      return Status::OK();
    case flatbuf::Type::Int:
      return IntFromFlatbuffer(static_cast<const flatbuf::Int*>(type_data), out);
    case flatbuf::Type::FloatingPoint:
      return FloatFromFlatbuffer(static_cast<const flatbuf::FloatingPoint*>(type_data),
                                 out);
    case flatbuf::Type::Binary:
      *out = binary();
      return Status::OK();
    case flatbuf::Type::LargeBinary:
      *out = large_binary();
      return Status::OK();
    case flatbuf::Type::FixedSizeBinary: {
      auto fw_binary = static_cast<const flatbuf::FixedSizeBinary*>(type_data);
      *out = fixed_size_binary(fw_binary->byteWidth());
      return Status::OK();
    }
    case flatbuf::Type::Utf8:
      *out = utf8();
      return Status::OK();
    case flatbuf::Type::LargeUtf8:
      *out = large_utf8();
      return Status::OK();
    case flatbuf::Type::Bool:
      *out = boolean();
      return Status::OK();
    case flatbuf::Type::Decimal: {
      auto dec_type = static_cast<const flatbuf::Decimal*>(type_data);
      return Decimal128Type::Make(dec_type->precision(), dec_type->scale(), out);
    }
    case flatbuf::Type::Date: {
      auto date_type = static_cast<const flatbuf::Date*>(type_data);
      if (date_type->unit() == flatbuf::DateUnit::DAY) {
        *out = date32();
      } else {
        *out = date64();
      }
      return Status::OK();
    }
    case flatbuf::Type::Time: {
      auto time_type = static_cast<const flatbuf::Time*>(type_data);
      TimeUnit::type unit = FromFlatbufferUnit(time_type->unit());
      int32_t bit_width = time_type->bitWidth();
      switch (unit) {
        case TimeUnit::SECOND:
        case TimeUnit::MILLI:
          if (bit_width != 32) {
            return Status::Invalid("Time is 32 bits for second/milli unit");
          }
          *out = time32(unit);
          break;
        default:
          if (bit_width != 64) {
            return Status::Invalid("Time is 64 bits for micro/nano unit");
          }
          *out = time64(unit);
          break;
      }
      return Status::OK();
    }
    case flatbuf::Type::Timestamp: {
      auto ts_type = static_cast<const flatbuf::Timestamp*>(type_data);
      TimeUnit::type unit = FromFlatbufferUnit(ts_type->unit());
      *out = timestamp(unit, StringFromFlatbuffers(ts_type->timezone()));
      return Status::OK();
    }
    case flatbuf::Type::Duration: {
      auto duration = static_cast<const flatbuf::Duration*>(type_data);
      TimeUnit::type unit = FromFlatbufferUnit(duration->unit());
      *out = arrow::duration(unit);
      return Status::OK();
    }

    case flatbuf::Type::Interval: {
      auto i_type = static_cast<const flatbuf::Interval*>(type_data);
      switch (i_type->unit()) {
        case flatbuf::IntervalUnit::YEAR_MONTH: {
          *out = month_interval();
          return Status::OK();
        }
        case flatbuf::IntervalUnit::DAY_TIME: {
          *out = day_time_interval();
          return Status::OK();
        }
      }
      return Status::NotImplemented("Unrecognized interval type.");
    }

    case flatbuf::Type::List:
      if (children.size() != 1) {
        return Status::Invalid("List must have exactly 1 child field");
      }
      *out = std::make_shared<ListType>(children[0]);
      return Status::OK();
    case flatbuf::Type::LargeList:
      if (children.size() != 1) {
        return Status::Invalid("LargeList must have exactly 1 child field");
      }
      *out = std::make_shared<LargeListType>(children[0]);
      return Status::OK();
    case flatbuf::Type::Map:
      if (children.size() != 1) {
        return Status::Invalid("Map must have exactly 1 child field");
      }
      if (children[0]->nullable() || children[0]->type()->id() != Type::STRUCT ||
          children[0]->type()->num_children() != 2) {
        return Status::Invalid("Map's key-item pairs must be non-nullable structs");
      }
      if (children[0]->type()->child(0)->nullable()) {
        return Status::Invalid("Map's keys must be non-nullable");
      } else {
        auto map = static_cast<const flatbuf::Map*>(type_data);
        *out = std::make_shared<MapType>(children[0]->type()->child(0)->type(),
                                         children[0]->type()->child(1)->type(),
                                         map->keysSorted());
      }
      return Status::OK();
    case flatbuf::Type::FixedSizeList:
      if (children.size() != 1) {
        return Status::Invalid("FixedSizeList must have exactly 1 child field");
      } else {
        auto fs_list = static_cast<const flatbuf::FixedSizeList*>(type_data);
        *out = std::make_shared<FixedSizeListType>(children[0], fs_list->listSize());
      }
      return Status::OK();
    case flatbuf::Type::Struct_:
      *out = std::make_shared<StructType>(children);
      return Status::OK();
    case flatbuf::Type::Union:
      return UnionFromFlatbuffer(static_cast<const flatbuf::Union*>(type_data), children,
                                 out);
    default:
      return Status::Invalid("Unrecognized type:" +
                             std::to_string(static_cast<int>(type)));
  }
}

static Status TypeFromFlatbuffer(const flatbuf::Field* field,
                                 const std::vector<std::shared_ptr<Field>>& children,
                                 const KeyValueMetadata* field_metadata,
                                 std::shared_ptr<DataType>* out) {
  auto type_data = field->type();
  CHECK_FLATBUFFERS_NOT_NULL(type_data, "Field.type");
  RETURN_NOT_OK(ConcreteTypeFromFlatbuffer(field->type_type(), type_data, children, out));

  // Look for extension metadata in custom_metadata field
  // TODO(wesm): Should this be part of the Field Flatbuffers table?
  if (field_metadata != nullptr) {
    int name_index = field_metadata->FindKey(kExtensionTypeKeyName);
    if (name_index == -1) {
      return Status::OK();
    }
    std::string type_name = field_metadata->value(name_index);
    int data_index = field_metadata->FindKey(kExtensionMetadataKeyName);
    std::string type_data = data_index == -1 ? "" : field_metadata->value(data_index);

    std::shared_ptr<ExtensionType> type = GetExtensionType(type_name);
    if (type == nullptr) {
      // TODO(wesm): Extension type is unknown; we do not raise here and simply
      // return the raw data
      return Status::OK();
    }
    RETURN_NOT_OK(type->Deserialize(*out, type_data, out));
  }
  return Status::OK();
}

Status TensorTypeToFlatbuffer(FBB& fbb, const DataType& type, flatbuf::Type* out_type,
                              Offset* offset) {
  switch (type.id()) {
    case Type::UINT8:
      INT_TO_FB_CASE(8, false);
    case Type::INT8:
      INT_TO_FB_CASE(8, true);
    case Type::UINT16:
      INT_TO_FB_CASE(16, false);
    case Type::INT16:
      INT_TO_FB_CASE(16, true);
    case Type::UINT32:
      INT_TO_FB_CASE(32, false);
    case Type::INT32:
      INT_TO_FB_CASE(32, true);
    case Type::UINT64:
      INT_TO_FB_CASE(64, false);
    case Type::INT64:
      INT_TO_FB_CASE(64, true);
    case Type::HALF_FLOAT:
      *out_type = flatbuf::Type::FloatingPoint;
      *offset = FloatToFlatbuffer(fbb, flatbuf::Precision::HALF);
      break;
    case Type::FLOAT:
      *out_type = flatbuf::Type::FloatingPoint;
      *offset = FloatToFlatbuffer(fbb, flatbuf::Precision::SINGLE);
      break;
    case Type::DOUBLE:
      *out_type = flatbuf::Type::FloatingPoint;
      *offset = FloatToFlatbuffer(fbb, flatbuf::Precision::DOUBLE);
      break;
    default:
      *out_type = flatbuf::Type::NONE;  // Make clang-tidy happy
      return Status::NotImplemented("Unable to convert type: ", type.ToString());
  }
  return Status::OK();
}

Status GetDictionaryEncoding(FBB& fbb, const std::shared_ptr<Field>& field,
                             DictionaryMemo* memo, DictionaryOffset* out) {
  int64_t dictionary_id = -1;
  RETURN_NOT_OK(memo->GetOrAssignId(field, &dictionary_id));

  const auto& type = checked_cast<const DictionaryType&>(*field->type());

  // We assume that the dictionary index type (as an integer) has already been
  // validated elsewhere, and can safely assume we are dealing with signed
  // integers
  const auto& fw_index_type = checked_cast<const FixedWidthType&>(*type.index_type());

  auto index_type_offset = flatbuf::CreateInt(fbb, fw_index_type.bit_width(), true);

  // TODO(wesm): ordered dictionaries
  *out = flatbuf::CreateDictionaryEncoding(fbb, dictionary_id, index_type_offset,
                                           type.ordered());
  return Status::OK();
}

KeyValueOffset AppendKeyValue(FBB& fbb, const std::string& key,
                              const std::string& value) {
  return flatbuf::CreateKeyValue(fbb, fbb.CreateString(key), fbb.CreateString(value));
}

void AppendKeyValueMetadata(FBB& fbb, const KeyValueMetadata& metadata,
                            std::vector<KeyValueOffset>* key_values) {
  key_values->reserve(metadata.size());
  for (int i = 0; i < metadata.size(); ++i) {
    key_values->push_back(AppendKeyValue(fbb, metadata.key(i), metadata.value(i)));
  }
}

class FieldToFlatbufferVisitor {
 public:
  FieldToFlatbufferVisitor(FBB& fbb, DictionaryMemo* dictionary_memo)
      : fbb_(fbb), dictionary_memo_(dictionary_memo) {}

  Status VisitType(const DataType& type) { return VisitTypeInline(type, this); }

  Status Visit(const NullType& type) {
    fb_type_ = flatbuf::Type::Null;
    type_offset_ = flatbuf::CreateNull(fbb_).Union();
    return Status::OK();
  }

  Status Visit(const BooleanType& type) {
    fb_type_ = flatbuf::Type::Bool;
    type_offset_ = flatbuf::CreateBool(fbb_).Union();
    return Status::OK();
  }

  template <int BitWidth, bool IsSigned, typename T>
  Status Visit(const T& type) {
    fb_type_ = flatbuf::Type::Int;
    type_offset_ = IntToFlatbuffer(fbb_, BitWidth, IsSigned);
    return Status::OK();
  }

  template <typename T>
  enable_if_integer<T, Status> Visit(const T& type) {
    constexpr bool is_signed = is_signed_integer_type<T>::value;
    return Visit<sizeof(typename T::c_type) * 8, is_signed>(type);
  }

  Status Visit(const HalfFloatType& type) {
    fb_type_ = flatbuf::Type::FloatingPoint;
    type_offset_ = FloatToFlatbuffer(fbb_, flatbuf::Precision::HALF);
    return Status::OK();
  }

  Status Visit(const FloatType& type) {
    fb_type_ = flatbuf::Type::FloatingPoint;
    type_offset_ = FloatToFlatbuffer(fbb_, flatbuf::Precision::SINGLE);
    return Status::OK();
  }

  Status Visit(const DoubleType& type) {
    fb_type_ = flatbuf::Type::FloatingPoint;
    type_offset_ = FloatToFlatbuffer(fbb_, flatbuf::Precision::DOUBLE);
    return Status::OK();
  }

  Status Visit(const FixedSizeBinaryType& type) {
    const auto& fw_type = checked_cast<const FixedSizeBinaryType&>(type);
    fb_type_ = flatbuf::Type::FixedSizeBinary;
    type_offset_ = flatbuf::CreateFixedSizeBinary(fbb_, fw_type.byte_width()).Union();
    return Status::OK();
  }

  Status Visit(const BinaryType& type) {
    fb_type_ = flatbuf::Type::Binary;
    type_offset_ = flatbuf::CreateBinary(fbb_).Union();
    return Status::OK();
  }

  Status Visit(const LargeBinaryType& type) {
    fb_type_ = flatbuf::Type::LargeBinary;
    type_offset_ = flatbuf::CreateLargeBinary(fbb_).Union();
    return Status::OK();
  }

  Status Visit(const StringType& type) {
    fb_type_ = flatbuf::Type::Utf8;
    type_offset_ = flatbuf::CreateUtf8(fbb_).Union();
    return Status::OK();
  }

  Status Visit(const LargeStringType& type) {
    fb_type_ = flatbuf::Type::LargeUtf8;
    type_offset_ = flatbuf::CreateLargeUtf8(fbb_).Union();
    return Status::OK();
  }

  Status Visit(const Date32Type& type) {
    fb_type_ = flatbuf::Type::Date;
    type_offset_ = flatbuf::CreateDate(fbb_, flatbuf::DateUnit::DAY).Union();
    return Status::OK();
  }

  Status Visit(const Date64Type& type) {
    fb_type_ = flatbuf::Type::Date;
    type_offset_ = flatbuf::CreateDate(fbb_, flatbuf::DateUnit::MILLISECOND).Union();
    return Status::OK();
  }

  Status Visit(const Time32Type& type) {
    const auto& time_type = checked_cast<const Time32Type&>(type);
    fb_type_ = flatbuf::Type::Time;
    type_offset_ =
        flatbuf::CreateTime(fbb_, ToFlatbufferUnit(time_type.unit()), 32).Union();
    return Status::OK();
  }

  Status Visit(const Time64Type& type) {
    const auto& time_type = checked_cast<const Time64Type&>(type);
    fb_type_ = flatbuf::Type::Time;
    type_offset_ =
        flatbuf::CreateTime(fbb_, ToFlatbufferUnit(time_type.unit()), 64).Union();
    return Status::OK();
  }

  Status Visit(const TimestampType& type) {
    const auto& ts_type = checked_cast<const TimestampType&>(type);
    fb_type_ = flatbuf::Type::Timestamp;
    flatbuf::TimeUnit fb_unit = ToFlatbufferUnit(ts_type.unit());
    FBString fb_timezone = 0;
    if (ts_type.timezone().size() > 0) {
      fb_timezone = fbb_.CreateString(ts_type.timezone());
    }
    type_offset_ = flatbuf::CreateTimestamp(fbb_, fb_unit, fb_timezone).Union();
    return Status::OK();
  }

  Status Visit(const DurationType& type) {
    fb_type_ = flatbuf::Type::Duration;
    flatbuf::TimeUnit fb_unit = ToFlatbufferUnit(type.unit());
    type_offset_ = flatbuf::CreateDuration(fbb_, fb_unit).Union();
    return Status::OK();
  }

  Status Visit(const DayTimeIntervalType& type) {
    fb_type_ = flatbuf::Type::Interval;
    type_offset_ = flatbuf::CreateInterval(fbb_, flatbuf::IntervalUnit::DAY_TIME).Union();
    return Status::OK();
  }

  Status Visit(const MonthIntervalType& type) {
    fb_type_ = flatbuf::Type::Interval;
    type_offset_ =
        flatbuf::CreateInterval(fbb_, flatbuf::IntervalUnit::YEAR_MONTH).Union();
    return Status::OK();
  }

  Status Visit(const DecimalType& type) {
    const auto& dec_type = checked_cast<const Decimal128Type&>(type);
    fb_type_ = flatbuf::Type::Decimal;
    type_offset_ =
        flatbuf::CreateDecimal(fbb_, dec_type.precision(), dec_type.scale()).Union();
    return Status::OK();
  }

  Status Visit(const ListType& type) {
    fb_type_ = flatbuf::Type::List;
    RETURN_NOT_OK(AppendChildFields(fbb_, type, &children_, dictionary_memo_));
    type_offset_ = flatbuf::CreateList(fbb_).Union();
    return Status::OK();
  }

  Status Visit(const LargeListType& type) {
    fb_type_ = flatbuf::Type::LargeList;
    RETURN_NOT_OK(AppendChildFields(fbb_, type, &children_, dictionary_memo_));
    type_offset_ = flatbuf::CreateLargeList(fbb_).Union();
    return Status::OK();
  }

  Status Visit(const MapType& type) {
    fb_type_ = flatbuf::Type::Map;
    RETURN_NOT_OK(AppendChildFields(fbb_, type, &children_, dictionary_memo_));
    type_offset_ = flatbuf::CreateMap(fbb_, type.keys_sorted()).Union();
    return Status::OK();
  }

  Status Visit(const FixedSizeListType& type) {
    fb_type_ = flatbuf::Type::FixedSizeList;
    RETURN_NOT_OK(AppendChildFields(fbb_, type, &children_, dictionary_memo_));
    type_offset_ = flatbuf::CreateFixedSizeList(fbb_, type.list_size()).Union();
    return Status::OK();
  }

  Status Visit(const StructType& type) {
    fb_type_ = flatbuf::Type::Struct_;
    RETURN_NOT_OK(AppendChildFields(fbb_, type, &children_, dictionary_memo_));
    type_offset_ = flatbuf::CreateStruct_(fbb_).Union();
    return Status::OK();
  }

  Status Visit(const UnionType& type) {
    fb_type_ = flatbuf::Type::Union;
    RETURN_NOT_OK(AppendChildFields(fbb_, type, &children_, dictionary_memo_));

    const auto& union_type = checked_cast<const UnionType&>(type);

    flatbuf::UnionMode mode = union_type.mode() == UnionMode::SPARSE
                                  ? flatbuf::UnionMode::Sparse
                                  : flatbuf::UnionMode::Dense;

    std::vector<int32_t> type_ids;
    type_ids.reserve(union_type.type_codes().size());
    for (uint8_t code : union_type.type_codes()) {
      type_ids.push_back(code);
    }

    auto fb_type_ids =
        fbb_.CreateVector(util::MakeNonNull(type_ids.data()), type_ids.size());

    type_offset_ = flatbuf::CreateUnion(fbb_, mode, fb_type_ids).Union();
    return Status::OK();
  }

  Status Visit(const DictionaryType& type) {
    // In this library, the dictionary "type" is a logical construct. Here we
    // pass through to the value type, as we've already captured the index
    // type in the DictionaryEncoding metadata in the parent field
    return VisitType(*checked_cast<const DictionaryType&>(type).value_type());
  }

  Status Visit(const ExtensionType& type) {
    RETURN_NOT_OK(VisitType(*type.storage_type()));
    extra_type_metadata_[kExtensionTypeKeyName] = type.extension_name();
    extra_type_metadata_[kExtensionMetadataKeyName] = type.Serialize();
    return Status::OK();
  }

  Status GetResult(const std::shared_ptr<Field>& field, FieldOffset* offset) {
    auto fb_name = fbb_.CreateString(field->name());
    RETURN_NOT_OK(VisitType(*field->type()));
    auto fb_children =
        fbb_.CreateVector(util::MakeNonNull(children_.data()), children_.size());

    DictionaryOffset dictionary = 0;
    if (field->type()->id() == Type::DICTIONARY) {
      RETURN_NOT_OK(GetDictionaryEncoding(fbb_, field, dictionary_memo_, &dictionary));
    }

    auto metadata = field->metadata();

    flatbuffers::Offset<KVVector> fb_custom_metadata;
    std::vector<KeyValueOffset> key_values;
    if (metadata != nullptr) {
      AppendKeyValueMetadata(fbb_, *metadata, &key_values);
    }

    for (auto it : extra_type_metadata_) {
      key_values.push_back(AppendKeyValue(fbb_, it.first, it.second));
    }

    if (key_values.size() > 0) {
      fb_custom_metadata = fbb_.CreateVector(key_values);
    }
    *offset =
        flatbuf::CreateField(fbb_, fb_name, field->nullable(), fb_type_, type_offset_,
                             dictionary, fb_children, fb_custom_metadata);
    return Status::OK();
  }

 private:
  FBB& fbb_;
  DictionaryMemo* dictionary_memo_;
  flatbuf::Type fb_type_;
  Offset type_offset_;
  std::vector<FieldOffset> children_;
  std::unordered_map<std::string, std::string> extra_type_metadata_;
};

Status FieldToFlatbuffer(FBB& fbb, const std::shared_ptr<Field>& field,
                         DictionaryMemo* dictionary_memo, FieldOffset* offset) {
  FieldToFlatbufferVisitor field_visitor(fbb, dictionary_memo);
  return field_visitor.GetResult(field, offset);
}

Status FieldFromFlatbuffer(const flatbuf::Field* field, DictionaryMemo* dictionary_memo,
                           std::shared_ptr<Field>* out) {
  std::shared_ptr<DataType> type;

  std::shared_ptr<const KeyValueMetadata> metadata;
  RETURN_NOT_OK(internal::GetKeyValueMetadata(field->custom_metadata(), &metadata));

  // Reconstruct the data type
  auto children = field->children();
  CHECK_FLATBUFFERS_NOT_NULL(children, "Field.children");
  std::vector<std::shared_ptr<Field>> child_fields(children->size());
  for (int i = 0; i < static_cast<int>(children->size()); ++i) {
    RETURN_NOT_OK(
        FieldFromFlatbuffer(children->Get(i), dictionary_memo, &child_fields[i]));
  }
  RETURN_NOT_OK(TypeFromFlatbuffer(field, child_fields, metadata.get(), &type));

  auto field_name = StringFromFlatbuffers(field->name());

  const flatbuf::DictionaryEncoding* encoding = field->dictionary();

  if (encoding != nullptr) {
    // The field is dictionary-encoded. Construct the DictionaryType
    // based on the DictionaryEncoding metadata and record in the
    // dictionary_memo
    std::shared_ptr<DataType> index_type;
    auto int_data = encoding->indexType();
    CHECK_FLATBUFFERS_NOT_NULL(int_data, "DictionaryEncoding.indexType");
    RETURN_NOT_OK(IntFromFlatbuffer(int_data, &index_type));
    ARROW_ASSIGN_OR_RAISE(type,
                          DictionaryType::Make(index_type, type, encoding->isOrdered()));
    *out = ::arrow::field(field_name, type, field->nullable(), metadata);
    RETURN_NOT_OK(dictionary_memo->AddField(encoding->id(), *out));
  } else {
    *out = ::arrow::field(field_name, type, field->nullable(), metadata);
  }
  return Status::OK();
}

// will return the endianness of the system we are running on
// based the NUMPY_API function. See NOTICE.txt
flatbuf::Endianness endianness() {
  union {
    uint32_t i;
    char c[4];
  } bint = {0x01020304};

  return bint.c[0] == 1 ? flatbuf::Endianness::Big : flatbuf::Endianness::Little;
}

flatbuffers::Offset<KVVector> SerializeCustomMetadata(
    FBB& fbb, const std::shared_ptr<const KeyValueMetadata>& metadata) {
  std::vector<KeyValueOffset> key_values;
  if (metadata != nullptr) {
    AppendKeyValueMetadata(fbb, *metadata, &key_values);
    return fbb.CreateVector(key_values);
  } else {
    // null
    return 0;
  }
}

Status SchemaToFlatbuffer(FBB& fbb, const Schema& schema, DictionaryMemo* dictionary_memo,
                          flatbuffers::Offset<flatbuf::Schema>* out) {
  /// Fields
  std::vector<FieldOffset> field_offsets;
  for (int i = 0; i < schema.num_fields(); ++i) {
    FieldOffset offset;
    RETURN_NOT_OK(FieldToFlatbuffer(fbb, schema.field(i), dictionary_memo, &offset));
    field_offsets.push_back(offset);
  }

  auto fb_offsets = fbb.CreateVector(field_offsets);
  *out = flatbuf::CreateSchema(fbb, endianness(), fb_offsets,
                               SerializeCustomMetadata(fbb, schema.metadata()));
  return Status::OK();
}

Result<std::shared_ptr<Buffer>> WriteFBMessage(
    FBB& fbb, flatbuf::MessageHeader header_type, flatbuffers::Offset<void> header,
    int64_t body_length,
    const std::shared_ptr<const KeyValueMetadata>& custom_metadata = nullptr) {
  auto message =
      flatbuf::CreateMessage(fbb, kCurrentMetadataVersion, header_type, header,
                             body_length, SerializeCustomMetadata(fbb, custom_metadata));
  fbb.Finish(message);
  return WriteFlatbufferBuilder(fbb);
}

using FieldNodeVector =
    flatbuffers::Offset<flatbuffers::Vector<const flatbuf::FieldNode*>>;
using BufferVector = flatbuffers::Offset<flatbuffers::Vector<const flatbuf::Buffer*>>;

static Status WriteFieldNodes(FBB& fbb, const std::vector<FieldMetadata>& nodes,
                              FieldNodeVector* out) {
  std::vector<flatbuf::FieldNode> fb_nodes;
  fb_nodes.reserve(nodes.size());

  for (size_t i = 0; i < nodes.size(); ++i) {
    const FieldMetadata& node = nodes[i];
    if (node.offset != 0) {
      return Status::Invalid("Field metadata for IPC must have offset 0");
    }
    fb_nodes.emplace_back(node.length, node.null_count);
  }
  *out = fbb.CreateVectorOfStructs(util::MakeNonNull(fb_nodes.data()), fb_nodes.size());
  return Status::OK();
}

static Status WriteBuffers(FBB& fbb, const std::vector<BufferMetadata>& buffers,
                           BufferVector* out) {
  std::vector<flatbuf::Buffer> fb_buffers;
  fb_buffers.reserve(buffers.size());

  for (size_t i = 0; i < buffers.size(); ++i) {
    const BufferMetadata& buffer = buffers[i];
    fb_buffers.emplace_back(buffer.offset, buffer.length);
  }
  *out =
      fbb.CreateVectorOfStructs(util::MakeNonNull(fb_buffers.data()), fb_buffers.size());

  return Status::OK();
}

static Status MakeRecordBatch(FBB& fbb, int64_t length, int64_t body_length,
                              const std::vector<FieldMetadata>& nodes,
                              const std::vector<BufferMetadata>& buffers,
                              RecordBatchOffset* offset) {
  FieldNodeVector fb_nodes;
  BufferVector fb_buffers;

  RETURN_NOT_OK(WriteFieldNodes(fbb, nodes, &fb_nodes));
  RETURN_NOT_OK(WriteBuffers(fbb, buffers, &fb_buffers));

  *offset = flatbuf::CreateRecordBatch(fbb, length, fb_nodes, fb_buffers);
  return Status::OK();
}

Status MakeSparseTensorIndexCOO(FBB& fbb, const SparseCOOIndex& sparse_index,
                                const std::vector<BufferMetadata>& buffers,
                                flatbuf::SparseTensorIndex* fb_sparse_index_type,
                                Offset* fb_sparse_index, size_t* num_buffers) {
  *fb_sparse_index_type = flatbuf::SparseTensorIndex::SparseTensorIndexCOO;

  // We assume that the value type of indices tensor is an integer.
  const auto& index_value_type =
      checked_cast<const IntegerType&>(*sparse_index.indices()->type());
  auto indices_type_offset =
      flatbuf::CreateInt(fbb, index_value_type.bit_width(), index_value_type.is_signed());

  auto fb_strides =
      fbb.CreateVector(util::MakeNonNull(sparse_index.indices()->strides().data()),
                       sparse_index.indices()->strides().size());

  const BufferMetadata& indices_metadata = buffers[0];
  flatbuf::Buffer indices(indices_metadata.offset, indices_metadata.length);

  *fb_sparse_index =
      flatbuf::CreateSparseTensorIndexCOO(fbb, indices_type_offset, fb_strides, &indices)
          .Union();
  *num_buffers = 1;
  return Status::OK();
}

template <typename SparseIndexType>
struct SparseMatrixCompressedAxis {};

template <>
struct SparseMatrixCompressedAxis<SparseCSRIndex> {
  constexpr static const auto value = flatbuf::SparseMatrixCompressedAxis::Row;
};

template <>
struct SparseMatrixCompressedAxis<SparseCSCIndex> {
  constexpr static const auto value = flatbuf::SparseMatrixCompressedAxis::Column;
};

template <typename SparseIndexType>
Status MakeSparseMatrixIndexCSX(FBB& fbb, const SparseIndexType& sparse_index,
                                const std::vector<BufferMetadata>& buffers,
                                flatbuf::SparseTensorIndex* fb_sparse_index_type,
                                Offset* fb_sparse_index, size_t* num_buffers) {
  *fb_sparse_index_type = flatbuf::SparseTensorIndex::SparseMatrixIndexCSX;

  // We assume that the value type of indptr tensor is an integer.
  const auto& indptr_value_type =
      checked_cast<const IntegerType&>(*sparse_index.indptr()->type());
  auto indptr_type_offset = flatbuf::CreateInt(fbb, indptr_value_type.bit_width(),
                                               indptr_value_type.is_signed());

  const BufferMetadata& indptr_metadata = buffers[0];
  flatbuf::Buffer indptr(indptr_metadata.offset, indptr_metadata.length);

  // We assume that the value type of indices tensor is an integer.
  const auto& indices_value_type =
      checked_cast<const IntegerType&>(*sparse_index.indices()->type());
  auto indices_type_offset = flatbuf::CreateInt(fbb, indices_value_type.bit_width(),
                                                indices_value_type.is_signed());

  const BufferMetadata& indices_metadata = buffers[1];
  flatbuf::Buffer indices(indices_metadata.offset, indices_metadata.length);

  auto compressedAxis = SparseMatrixCompressedAxis<SparseIndexType>::value;
  *fb_sparse_index =
      flatbuf::CreateSparseMatrixIndexCSX(fbb, compressedAxis, indptr_type_offset,
                                          &indptr, indices_type_offset, &indices)
          .Union();
  *num_buffers = 2;
  return Status::OK();
}

Status MakeSparseTensorIndex(FBB& fbb, const SparseIndex& sparse_index,
                             const std::vector<BufferMetadata>& buffers,
                             flatbuf::SparseTensorIndex* fb_sparse_index_type,
                             Offset* fb_sparse_index, size_t* num_buffers) {
  switch (sparse_index.format_id()) {
    case SparseTensorFormat::COO:
      RETURN_NOT_OK(MakeSparseTensorIndexCOO(
          fbb, checked_cast<const SparseCOOIndex&>(sparse_index), buffers,
          fb_sparse_index_type, fb_sparse_index, num_buffers));
      break;

    case SparseTensorFormat::CSR:
      RETURN_NOT_OK(MakeSparseMatrixIndexCSX(
          fbb, checked_cast<const SparseCSRIndex&>(sparse_index), buffers,
          fb_sparse_index_type, fb_sparse_index, num_buffers));
      break;

    case SparseTensorFormat::CSC:
      RETURN_NOT_OK(MakeSparseMatrixIndexCSX(
          fbb, checked_cast<const SparseCSCIndex&>(sparse_index), buffers,
          fb_sparse_index_type, fb_sparse_index, num_buffers));
      break;

    default:
      std::stringstream ss;
      ss << "Unsupported sparse tensor format:: " << sparse_index.ToString() << std::endl;
      return Status::NotImplemented(ss.str());
  }

  return Status::OK();
}

Status MakeSparseTensor(FBB& fbb, const SparseTensor& sparse_tensor, int64_t body_length,
                        const std::vector<BufferMetadata>& buffers,
                        SparseTensorOffset* offset) {
  flatbuf::Type fb_type_type;
  Offset fb_type;
  RETURN_NOT_OK(
      TensorTypeToFlatbuffer(fbb, *sparse_tensor.type(), &fb_type_type, &fb_type));

  using TensorDimOffset = flatbuffers::Offset<flatbuf::TensorDim>;
  std::vector<TensorDimOffset> dims;
  for (int i = 0; i < sparse_tensor.ndim(); ++i) {
    FBString name = fbb.CreateString(sparse_tensor.dim_name(i));
    dims.push_back(flatbuf::CreateTensorDim(fbb, sparse_tensor.shape()[i], name));
  }

  auto fb_shape = fbb.CreateVector(dims);

  flatbuf::SparseTensorIndex fb_sparse_index_type;
  Offset fb_sparse_index;
  size_t num_index_buffers = 0;
  RETURN_NOT_OK(MakeSparseTensorIndex(fbb, *sparse_tensor.sparse_index(), buffers,
                                      &fb_sparse_index_type, &fb_sparse_index,
                                      &num_index_buffers));

  const BufferMetadata& data_metadata = buffers[num_index_buffers];
  flatbuf::Buffer data(data_metadata.offset, data_metadata.length);

  const int64_t non_zero_length = sparse_tensor.non_zero_length();

  *offset =
      flatbuf::CreateSparseTensor(fbb, fb_type_type, fb_type, fb_shape, non_zero_length,
                                  fb_sparse_index_type, fb_sparse_index, &data);

  return Status::OK();
}

}  // namespace

Status GetKeyValueMetadata(const KVVector* fb_metadata,
                           std::shared_ptr<const KeyValueMetadata>* out) {
  if (fb_metadata == nullptr) {
    *out = nullptr;
    return Status::OK();
  }

  auto metadata = std::make_shared<KeyValueMetadata>();

  metadata->reserve(fb_metadata->size());
  for (const auto& pair : *fb_metadata) {
    CHECK_FLATBUFFERS_NOT_NULL(pair->key(), "custom_metadata.key");
    CHECK_FLATBUFFERS_NOT_NULL(pair->value(), "custom_metadata.value");
    metadata->Append(pair->key()->str(), pair->value()->str());
  }

  *out = std::move(metadata);
  return Status::OK();
}

Status WriteSchemaMessage(const Schema& schema, DictionaryMemo* dictionary_memo,
                          std::shared_ptr<Buffer>* out) {
  FBB fbb;
  flatbuffers::Offset<flatbuf::Schema> fb_schema;
  RETURN_NOT_OK(SchemaToFlatbuffer(fbb, schema, dictionary_memo, &fb_schema));
  return WriteFBMessage(fbb, flatbuf::MessageHeader::Schema, fb_schema.Union(),
                        /*body_length=*/0)
      .Value(out);
}

Status WriteRecordBatchMessage(
    int64_t length, int64_t body_length,
    const std::shared_ptr<const KeyValueMetadata>& custom_metadata,
    const std::vector<FieldMetadata>& nodes, const std::vector<BufferMetadata>& buffers,
    std::shared_ptr<Buffer>* out) {
  FBB fbb;
  RecordBatchOffset record_batch;
  RETURN_NOT_OK(MakeRecordBatch(fbb, length, body_length, nodes, buffers, &record_batch));
  return WriteFBMessage(fbb, flatbuf::MessageHeader::RecordBatch, record_batch.Union(),
                        body_length, custom_metadata)
      .Value(out);
}

Result<std::shared_ptr<Buffer>> WriteTensorMessage(const Tensor& tensor,
                                                   int64_t buffer_start_offset) {
  using TensorDimOffset = flatbuffers::Offset<flatbuf::TensorDim>;
  using TensorOffset = flatbuffers::Offset<flatbuf::Tensor>;

  FBB fbb;

  const auto& type = checked_cast<const FixedWidthType&>(*tensor.type());
  const int elem_size = type.bit_width() / 8;

  flatbuf::Type fb_type_type;
  Offset fb_type;
  RETURN_NOT_OK(TensorTypeToFlatbuffer(fbb, *tensor.type(), &fb_type_type, &fb_type));

  std::vector<TensorDimOffset> dims;
  for (int i = 0; i < tensor.ndim(); ++i) {
    FBString name = fbb.CreateString(tensor.dim_name(i));
    dims.push_back(flatbuf::CreateTensorDim(fbb, tensor.shape()[i], name));
  }

  auto fb_shape = fbb.CreateVector(util::MakeNonNull(dims.data()), dims.size());

  flatbuffers::Offset<flatbuffers::Vector<int64_t>> fb_strides;
  fb_strides = fbb.CreateVector(util::MakeNonNull(tensor.strides().data()),
                                tensor.strides().size());
  int64_t body_length = tensor.size() * elem_size;
  flatbuf::Buffer buffer(buffer_start_offset, body_length);

  TensorOffset fb_tensor =
      flatbuf::CreateTensor(fbb, fb_type_type, fb_type, fb_shape, fb_strides, &buffer);

  return WriteFBMessage(fbb, flatbuf::MessageHeader::Tensor, fb_tensor.Union(),
                        body_length);
}

Result<std::shared_ptr<Buffer>> WriteSparseTensorMessage(
    const SparseTensor& sparse_tensor, int64_t body_length,
    const std::vector<BufferMetadata>& buffers) {
  FBB fbb;
  SparseTensorOffset fb_sparse_tensor;
  RETURN_NOT_OK(
      MakeSparseTensor(fbb, sparse_tensor, body_length, buffers, &fb_sparse_tensor));
  return WriteFBMessage(fbb, flatbuf::MessageHeader::SparseTensor,
                        fb_sparse_tensor.Union(), body_length);
}

Status WriteDictionaryMessage(
    int64_t id, int64_t length, int64_t body_length,
    const std::shared_ptr<const KeyValueMetadata>& custom_metadata,
    const std::vector<FieldMetadata>& nodes, const std::vector<BufferMetadata>& buffers,
    std::shared_ptr<Buffer>* out) {
  FBB fbb;
  RecordBatchOffset record_batch;
  RETURN_NOT_OK(MakeRecordBatch(fbb, length, body_length, nodes, buffers, &record_batch));
  auto dictionary_batch = flatbuf::CreateDictionaryBatch(fbb, id, record_batch).Union();
  return WriteFBMessage(fbb, flatbuf::MessageHeader::DictionaryBatch, dictionary_batch,
                        body_length, custom_metadata)
      .Value(out);
}

static flatbuffers::Offset<flatbuffers::Vector<const flatbuf::Block*>>
FileBlocksToFlatbuffer(FBB& fbb, const std::vector<FileBlock>& blocks) {
  std::vector<flatbuf::Block> fb_blocks;

  for (const FileBlock& block : blocks) {
    fb_blocks.emplace_back(block.offset, block.metadata_length, block.body_length);
  }

  return fbb.CreateVectorOfStructs(util::MakeNonNull(fb_blocks.data()), fb_blocks.size());
}

Status WriteFileFooter(const Schema& schema, const std::vector<FileBlock>& dictionaries,
                       const std::vector<FileBlock>& record_batches,
                       io::OutputStream* out) {
  FBB fbb;

  flatbuffers::Offset<flatbuf::Schema> fb_schema;
  DictionaryMemo dictionary_memo;  // unused
  RETURN_NOT_OK(SchemaToFlatbuffer(fbb, schema, &dictionary_memo, &fb_schema));

#ifndef NDEBUG
  for (size_t i = 0; i < dictionaries.size(); ++i) {
    DCHECK(BitUtil::IsMultipleOf8(dictionaries[i].offset)) << i;
    DCHECK(BitUtil::IsMultipleOf8(dictionaries[i].metadata_length)) << i;
    DCHECK(BitUtil::IsMultipleOf8(dictionaries[i].body_length)) << i;
  }

  for (size_t i = 0; i < record_batches.size(); ++i) {
    DCHECK(BitUtil::IsMultipleOf8(record_batches[i].offset)) << i;
    DCHECK(BitUtil::IsMultipleOf8(record_batches[i].metadata_length)) << i;
    DCHECK(BitUtil::IsMultipleOf8(record_batches[i].body_length)) << i;
  }
#endif

  auto fb_dictionaries = FileBlocksToFlatbuffer(fbb, dictionaries);
  auto fb_record_batches = FileBlocksToFlatbuffer(fbb, record_batches);

  auto footer = flatbuf::CreateFooter(fbb, kCurrentMetadataVersion, fb_schema,
                                      fb_dictionaries, fb_record_batches);
  fbb.Finish(footer);

  int32_t size = fbb.GetSize();

  return out->Write(fbb.GetBufferPointer(), size);
}

// ----------------------------------------------------------------------

Status GetSchema(const void* opaque_schema, DictionaryMemo* dictionary_memo,
                 std::shared_ptr<Schema>* out) {
  auto schema = static_cast<const flatbuf::Schema*>(opaque_schema);
  CHECK_FLATBUFFERS_NOT_NULL(schema, "schema");
  CHECK_FLATBUFFERS_NOT_NULL(schema->fields(), "Schema.fields");
  int num_fields = static_cast<int>(schema->fields()->size());

  std::vector<std::shared_ptr<Field>> fields(num_fields);
  for (int i = 0; i < num_fields; ++i) {
    const flatbuf::Field* field = schema->fields()->Get(i);
    // XXX I don't think this check is necessary (AP)
    CHECK_FLATBUFFERS_NOT_NULL(field, "DictionaryEncoding.indexType");
    RETURN_NOT_OK(FieldFromFlatbuffer(field, dictionary_memo, &fields[i]));
  }

  std::shared_ptr<const KeyValueMetadata> metadata;
  RETURN_NOT_OK(internal::GetKeyValueMetadata(schema->custom_metadata(), &metadata));
  *out = ::arrow::schema(std::move(fields), metadata);
  return Status::OK();
}

Status GetTensorMetadata(const Buffer& metadata, std::shared_ptr<DataType>* type,
                         std::vector<int64_t>* shape, std::vector<int64_t>* strides,
                         std::vector<std::string>* dim_names) {
  const flatbuf::Message* message;
  RETURN_NOT_OK(internal::VerifyMessage(metadata.data(), metadata.size(), &message));
  auto tensor = message->header_as_Tensor();
  if (tensor == nullptr) {
    return Status::IOError("Header-type of flatbuffer-encoded Message is not Tensor.");
  }

  int ndim = static_cast<int>(tensor->shape()->size());

  for (int i = 0; i < ndim; ++i) {
    auto dim = tensor->shape()->Get(i);

    shape->push_back(dim->size());
    dim_names->push_back(StringFromFlatbuffers(dim->name()));
  }

  if (tensor->strides() && tensor->strides()->size() > 0) {
    for (int i = 0; i < ndim; ++i) {
      strides->push_back(tensor->strides()->Get(i));
    }
  }

  auto type_data = tensor->type();  // Required
  return ConcreteTypeFromFlatbuffer(tensor->type_type(), type_data, {}, type);
}

Status GetSparseCOOIndexMetadata(const flatbuf::SparseTensorIndexCOO* sparse_index,
                                 std::shared_ptr<DataType>* indices_type) {
  return IntFromFlatbuffer(sparse_index->indicesType(), indices_type);
}

Status GetSparseCSXIndexMetadata(const flatbuf::SparseMatrixIndexCSX* sparse_index,
                                 std::shared_ptr<DataType>* indptr_type,
                                 std::shared_ptr<DataType>* indices_type) {
  RETURN_NOT_OK(IntFromFlatbuffer(sparse_index->indptrType(), indptr_type));
  RETURN_NOT_OK(IntFromFlatbuffer(sparse_index->indicesType(), indices_type));
  return Status::OK();
}

Status GetSparseTensorMetadata(const Buffer& metadata, std::shared_ptr<DataType>* type,
                               std::vector<int64_t>* shape,
                               std::vector<std::string>* dim_names,
                               int64_t* non_zero_length,
                               SparseTensorFormat::type* sparse_tensor_format_id) {
  const flatbuf::Message* message;
  RETURN_NOT_OK(internal::VerifyMessage(metadata.data(), metadata.size(), &message));
  auto sparse_tensor = message->header_as_SparseTensor();
  if (sparse_tensor == nullptr) {
    return Status::IOError(
        "Header-type of flatbuffer-encoded Message is not SparseTensor.");
  }
  int ndim = static_cast<int>(sparse_tensor->shape()->size());

  if (shape || dim_names) {
    for (int i = 0; i < ndim; ++i) {
      auto dim = sparse_tensor->shape()->Get(i);

      if (shape) {
        shape->push_back(dim->size());
      }

      if (dim_names) {
        dim_names->push_back(StringFromFlatbuffers(dim->name()));
      }
    }
  }

  if (non_zero_length) {
    *non_zero_length = sparse_tensor->non_zero_length();
  }

  if (sparse_tensor_format_id) {
    switch (sparse_tensor->sparseIndex_type()) {
      case flatbuf::SparseTensorIndex::SparseTensorIndexCOO:
        *sparse_tensor_format_id = SparseTensorFormat::COO;
        break;

      case flatbuf::SparseTensorIndex::SparseMatrixIndexCSX: {
        auto cs = sparse_tensor->sparseIndex_as_SparseMatrixIndexCSX();
        switch (cs->compressedAxis()) {
          case flatbuf::SparseMatrixCompressedAxis::Row:
            *sparse_tensor_format_id = SparseTensorFormat::CSR;
            break;

          case flatbuf::SparseMatrixCompressedAxis::Column:
            *sparse_tensor_format_id = SparseTensorFormat::CSC;
            break;

          default:
            return Status::Invalid("Invalid value of SparseMatrixCompressedAxis");
        }
      } break;

      default:
        return Status::Invalid("Unrecognized sparse index type");
    }
  }

  auto type_data = sparse_tensor->type();  // Required
  if (type) {
    return ConcreteTypeFromFlatbuffer(sparse_tensor->type_type(), type_data, {}, type);
  } else {
    return Status::OK();
  }
}

}  // namespace internal
}  // namespace ipc
}  // namespace arrow
