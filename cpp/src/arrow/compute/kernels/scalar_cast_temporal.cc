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

// Implementation of casting to (or between) temporal types

#include <utility>
#include <vector>

#include "arrow/compute/kernels/common.h"
#include "arrow/compute/kernels/scalar_cast_internal.h"
#include "arrow/util/time.h"
#include "arrow/util/value_parsing.h"

namespace arrow {

using internal::BitmapReader;
using internal::ParseTimestampContext;
using internal::ParseValue;

namespace compute {
namespace internal {

constexpr int64_t kMillisecondsInDay = 86400000;

// ----------------------------------------------------------------------
// From one timestamp to another

template <typename in_type, typename out_type>
void ShiftTime(KernelContext* ctx, const util::DivideOrMultiply factor_op,
               const int64_t factor, const ArrayData& input, ArrayData* output) {
  const CastOptions& options = checked_cast<const CastState&>(*ctx->state()).options;
  const in_type* in_data = input.GetValues<in_type>(1);
  auto out_data = output->GetMutableValues<out_type>(1);

  if (factor == 1) {
    for (int64_t i = 0; i < input.length; i++) {
      out_data[i] = static_cast<out_type>(in_data[i]);
    }
  } else if (factor_op == util::MULTIPLY) {
    if (options.allow_time_overflow) {
      for (int64_t i = 0; i < input.length; i++) {
        out_data[i] = static_cast<out_type>(in_data[i] * factor);
      }
    } else {
#define RAISE_OVERFLOW_CAST(VAL)                                                  \
  ctx->SetStatus(Status::Invalid("Casting from ", input.type->ToString(), " to ", \
                                 output->type->ToString(), " would result in ",   \
                                 "out of bounds timestamp: ", VAL));

      int64_t max_val = std::numeric_limits<int64_t>::max() / factor;
      int64_t min_val = std::numeric_limits<int64_t>::min() / factor;
      if (input.null_count != 0) {
        BitmapReader bit_reader(input.buffers[0]->data(), input.offset, input.length);
        for (int64_t i = 0; i < input.length; i++) {
          if (bit_reader.IsSet() && (in_data[i] < min_val || in_data[i] > max_val)) {
            RAISE_OVERFLOW_CAST(in_data[i]);
            break;
          }
          out_data[i] = static_cast<out_type>(in_data[i] * factor);
          bit_reader.Next();
        }
      } else {
        for (int64_t i = 0; i < input.length; i++) {
          if (in_data[i] < min_val || in_data[i] > max_val) {
            RAISE_OVERFLOW_CAST(in_data[i]);
            break;
          }
          out_data[i] = static_cast<out_type>(in_data[i] * factor);
        }
      }

#undef RAISE_OVERFLOW_CAST
    }
  } else {
    if (options.allow_time_truncate) {
      for (int64_t i = 0; i < input.length; i++) {
        out_data[i] = static_cast<out_type>(in_data[i] / factor);
      }
    } else {
#define RAISE_INVALID_CAST(VAL)                                                   \
  ctx->SetStatus(Status::Invalid("Casting from ", input.type->ToString(), " to ", \
                                 output->type->ToString(), " would lose data: ", VAL));

      if (input.null_count != 0) {
        BitmapReader bit_reader(input.buffers[0]->data(), input.offset, input.length);
        for (int64_t i = 0; i < input.length; i++) {
          out_data[i] = static_cast<out_type>(in_data[i] / factor);
          if (bit_reader.IsSet() && (out_data[i] * factor != in_data[i])) {
            RAISE_INVALID_CAST(in_data[i]);
            break;
          }
          bit_reader.Next();
        }
      } else {
        for (int64_t i = 0; i < input.length; i++) {
          out_data[i] = static_cast<out_type>(in_data[i] / factor);
          if (out_data[i] * factor != in_data[i]) {
            RAISE_INVALID_CAST(in_data[i]);
            break;
          }
        }
      }

#undef RAISE_INVALID_CAST
    }
  }
}

// <TimestampType, TimestampType> and <DurationType, DurationType>
template <typename O, typename I>
struct CastFunctor<
    O, I,
    enable_if_t<(is_timestamp_type<O>::value && is_timestamp_type<I>::value) ||
                (is_duration_type<O>::value && is_duration_type<I>::value)>> {
  static void Exec(KernelContext* ctx, const ExecBatch& batch, Datum* out) {
    const ArrayData& input = *batch[0].array();
    ArrayData* output = out->mutable_array();

    // If units are the same, zero copy, otherwise convert
    const auto& in_type = checked_cast<const I&>(*batch[0].type());
    const auto& out_type = checked_cast<const O&>(*output->type);

    DCHECK_NE(in_type.unit(), out_type.unit()) << "Do not cast equal types";

    auto conversion = util::kTimestampConversionTable[static_cast<int>(in_type.unit())]
                                                     [static_cast<int>(out_type.unit())];
    ShiftTime<int64_t, int64_t>(ctx, conversion.first, conversion.second, input, output);
  }
};

template <>
struct CastFunctor<Date32Type, TimestampType> {
  static void Exec(KernelContext* ctx, const ExecBatch& batch, Datum* out) {
    const ArrayData& input = *batch[0].array();
    ArrayData* output = out->mutable_array();

    const auto& in_type = checked_cast<const TimestampType&>(*input.type);

    static const int64_t kTimestampToDateFactors[4] = {
        86400LL,                             // SECOND
        86400LL * 1000LL,                    // MILLI
        86400LL * 1000LL * 1000LL,           // MICRO
        86400LL * 1000LL * 1000LL * 1000LL,  // NANO
    };

    const int64_t factor = kTimestampToDateFactors[static_cast<int>(in_type.unit())];
    ShiftTime<int64_t, int32_t>(ctx, util::DIVIDE, factor, input, output);
  }
};

template <>
struct CastFunctor<Date64Type, TimestampType> {
  static void Exec(KernelContext* ctx, const ExecBatch& batch, Datum* out) {
    const CastOptions& options = checked_cast<const CastState&>(*ctx->state()).options;
    const ArrayData& input = *batch[0].array();
    ArrayData* output = out->mutable_array();
    const auto& in_type = checked_cast<const TimestampType&>(*input.type);

    auto conversion = util::kTimestampConversionTable[static_cast<int>(in_type.unit())]
                                                     [static_cast<int>(TimeUnit::MILLI)];
    ShiftTime<int64_t, int64_t>(ctx, conversion.first, conversion.second, input, output);
    if (!ctx->status().ok()) {
      return;
    }

    // Ensure that intraday milliseconds have been zeroed out
    auto out_data = output->GetMutableValues<int64_t>(1);

    if (input.null_count != 0) {
      BitmapReader bit_reader(input.buffers[0]->data(), input.offset, input.length);

      for (int64_t i = 0; i < input.length; ++i) {
        const int64_t remainder = out_data[i] % kMillisecondsInDay;
        if (ARROW_PREDICT_FALSE(!options.allow_time_truncate && bit_reader.IsSet() &&
                                remainder > 0)) {
          ctx->SetStatus(
              Status::Invalid("Timestamp value had non-zero intraday milliseconds"));
          break;
        }
        out_data[i] -= remainder;
        bit_reader.Next();
      }
    } else {
      for (int64_t i = 0; i < input.length; ++i) {
        const int64_t remainder = out_data[i] % kMillisecondsInDay;
        if (ARROW_PREDICT_FALSE(!options.allow_time_truncate && remainder > 0)) {
          ctx->SetStatus(
              Status::Invalid("Timestamp value had non-zero intraday milliseconds"));
          break;
        }
        out_data[i] -= remainder;
      }
    }
  }
};

// ----------------------------------------------------------------------
// From one time32 or time64 to another

template <typename O, typename I>
struct CastFunctor<O, I, enable_if_t<is_time_type<I>::value && is_time_type<O>::value>> {
  using in_t = typename I::c_type;
  using out_t = typename O::c_type;

  static void Exec(KernelContext* ctx, const ExecBatch& batch, Datum* out) {
    const ArrayData& input = *batch[0].array();
    ArrayData* output = out->mutable_array();

    // If units are the same, zero copy, otherwise convert
    const auto& in_type = checked_cast<const I&>(*input.type);
    const auto& out_type = checked_cast<const O&>(*output->type);
    DCHECK_NE(in_type.unit(), out_type.unit()) << "Do not cast equal types";
    auto conversion = util::kTimestampConversionTable[static_cast<int>(in_type.unit())]
                                                     [static_cast<int>(out_type.unit())];

    ShiftTime<in_t, out_t>(ctx, conversion.first, conversion.second, input, output);
  }
};

// ----------------------------------------------------------------------
// Between date32 and date64

template <>
struct CastFunctor<Date64Type, Date32Type> {
  static void Exec(KernelContext* ctx, const ExecBatch& batch, Datum* out) {
    ShiftTime<int32_t, int64_t>(ctx, util::MULTIPLY, kMillisecondsInDay,
                                *batch[0].array(), out->mutable_array());
  }
};

template <>
struct CastFunctor<Date32Type, Date64Type> {
  static void Exec(KernelContext* ctx, const ExecBatch& batch, Datum* out) {
    ShiftTime<int64_t, int32_t>(ctx, util::DIVIDE, kMillisecondsInDay, *batch[0].array(),
                                out->mutable_array());
  }
};

// ----------------------------------------------------------------------
// String to Timestamp

struct ParseTimestamp {
  explicit ParseTimestamp(TimeUnit::type unit) : unit(unit) {}

  template <typename OUT, typename ARG0>
  OUT Call(KernelContext* ctx, ARG0 val) {
    ParseTimestampContext parse_ctx{this->unit};
    OUT result;
    if (ARROW_PREDICT_FALSE(
            !ParseValue<TimestampType>(val.data(), val.size(), &result, &parse_ctx))) {
      ctx->SetStatus(Status::Invalid("Failed to parse string: ", val));
    }
    return result;
  }

  TimeUnit::type unit;
};

template <typename I>
struct CastFunctor<TimestampType, I, enable_if_t<is_base_binary_type<I>::value>> {
  static void Exec(KernelContext* ctx, const ExecBatch& batch, Datum* out) {
    const CastOptions& options = checked_cast<const CastState&>(*ctx->state()).options;
    const auto& out_type = checked_cast<const TimestampType&>(*options.to_type);
    codegen::ScalarUnaryNotNullStateful<TimestampType, I, ParseTimestamp> kernel(
        ParseTimestamp(out_type.unit()));
    return kernel.Exec(ctx, batch, out);
  }
};

/// You will see some of these kernels with
///
/// ResolveOutputFromOptions
///
/// for their output type resolution. This is somewhat of an eyesore but the
/// easiest initial way to get the requested cast type including the TimeUnit
/// to the kernel (which is needed to compute the output) was through
/// CastOptions

static OutputType kOutputTargetType(ResolveOutputFromOptions);

template <typename Type>
void AddCrossUnitCast(CastFunction* func) {
  ScalarKernel kernel;
  kernel.exec = CastFunctor<Type, Type>::Exec;
  kernel.signature = KernelSignature::Make({InputType(Type::type_id)}, kOutputTargetType);
  DCHECK_OK(func->AddKernel(Type::type_id, std::move(kernel)));
}

std::shared_ptr<CastFunction> GetDate32Cast() {
  auto func = std::make_shared<CastFunction>("cast_date32", Type::DATE32);
  auto out_ty = date32();
  AddCommonCasts<Date32Type>(out_ty, func.get());

  // int64 -> date64
  AddZeroCopyCast(int32(), date32(), func.get());

  // date64 -> date32
  AddSimpleCast<Date64Type, Date32Type>(date64(), date32(), func.get());

  // timestamp -> date32
  AddSimpleCast<TimestampType, Date32Type>(InputType(Type::TIMESTAMP), date32(),
                                           func.get());
  return func;
}

std::shared_ptr<CastFunction> GetDate64Cast() {
  auto func = std::make_shared<CastFunction>("cast_date64", Type::DATE64);
  auto out_ty = date64();
  AddCommonCasts<Date64Type>(out_ty, func.get());

  // int64 -> date64
  AddZeroCopyCast(int64(), date64(), func.get());

  // date32 -> date64
  AddSimpleCast<Date32Type, Date64Type>(date32(), date64(), func.get());

  // timestamp -> date64
  AddSimpleCast<TimestampType, Date64Type>(InputType(Type::TIMESTAMP), date64(),
                                           func.get());

  return func;
}

std::shared_ptr<CastFunction> GetDurationCast() {
  auto func = std::make_shared<CastFunction>("cast_duration", Type::DURATION);
  AddCommonCasts<DurationType>(kOutputTargetType, func.get());

  auto seconds = duration(TimeUnit::SECOND);
  auto millis = duration(TimeUnit::MILLI);
  auto micros = duration(TimeUnit::MICRO);
  auto nanos = duration(TimeUnit::NANO);

  // Same integer representation
  AddZeroCopyCast(/*in_type=*/int64(), /*out_type=*/seconds, func.get());
  AddZeroCopyCast(int64(), millis, func.get());
  AddZeroCopyCast(int64(), micros, func.get());
  AddZeroCopyCast(int64(), nanos, func.get());

  // Between durations
  AddCrossUnitCast<DurationType>(func.get());

  return func;
}

std::shared_ptr<CastFunction> GetTime32Cast() {
  auto func = std::make_shared<CastFunction>("cast_time32", Type::TIME32);
  AddCommonCasts<Date32Type>(kOutputTargetType, func.get());

  auto seconds = time32(TimeUnit::SECOND);
  auto millis = time32(TimeUnit::MILLI);

  // Zero copy when the unit is the same or same integer representation
  AddZeroCopyCast(int32(), seconds, func.get());
  AddZeroCopyCast(int32(), millis, func.get());

  // time64 -> time32
  AddSimpleCast<Time64Type, Time32Type>(InputType(Type::TIME64), kOutputTargetType,
                                        func.get());

  // time32 -> time32
  AddCrossUnitCast<Time32Type>(func.get());

  return func;
}

std::shared_ptr<CastFunction> GetTime64Cast() {
  auto func = std::make_shared<CastFunction>("cast_time64", Type::TIME64);
  AddCommonCasts<Time64Type>(kOutputTargetType, func.get());

  auto micros = time64(TimeUnit::MICRO);
  auto nanos = time64(TimeUnit::NANO);

  // Zero copy when the unit is the same or same integer representation
  AddZeroCopyCast(int64(), micros, func.get());
  AddZeroCopyCast(int64(), nanos, func.get());

  // time32 -> time64
  AddSimpleCast<Time32Type, Time64Type>(InputType(Type::TIME32), kOutputTargetType,
                                        func.get());

  // Between durations
  AddCrossUnitCast<Time64Type>(func.get());

  return func;
}

std::shared_ptr<CastFunction> GetTimestampCast() {
  auto func = std::make_shared<CastFunction>("cast_timestamp", Type::TIMESTAMP);
  AddCommonCasts<TimestampType>(kOutputTargetType, func.get());

  // Same integer representation
  AddZeroCopyCast(/*in_type=*/int64(), /*out_type=*/timestamp(TimeUnit::SECOND),
                  func.get());
  AddZeroCopyCast(int64(), timestamp(TimeUnit::MILLI), func.get());
  AddZeroCopyCast(int64(), timestamp(TimeUnit::MICRO), func.get());
  AddZeroCopyCast(int64(), timestamp(TimeUnit::NANO), func.get());

  // From date types
  // TODO: ARROW-8876, these casts are not implemented
  // AddSimpleCast<Date32Type, TimestampType>(InputType(Type::DATE32),
  //                                          kOutputTargetType, func.get());
  // AddSimpleCast<Date64Type, TimestampType>(InputType(Type::DATE64),
  //                                          kOutputTargetType, func.get());

  // From one timestamp to another
  AddCrossUnitCast<TimestampType>(func.get());

  return func;
}

std::vector<std::shared_ptr<CastFunction>> GetTemporalCasts() {
  std::vector<std::shared_ptr<CastFunction>> functions;

  functions.push_back(GetDate32Cast());
  functions.push_back(GetDate64Cast());
  functions.push_back(GetDurationCast());
  functions.push_back(GetTime32Cast());
  functions.push_back(GetTime64Cast());
  functions.push_back(GetTimestampCast());
  return functions;
}

}  // namespace internal
}  // namespace compute
}  // namespace arrow
