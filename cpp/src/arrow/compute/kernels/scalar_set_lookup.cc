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

#include <algorithm>

#include "arrow/builder.h"
#include "arrow/compute/kernels/common.h"
#include "arrow/util/hashing.h"
#include "arrow/util/optional.h"

namespace arrow {

using internal::checked_cast;

namespace compute {

template <typename T, typename R = void>
using enable_if_supports_set_lookup =
    enable_if_t<has_c_type<T>::value || is_base_binary_type<T>::value ||
                    is_fixed_size_binary_type<T>::value || is_decimal_type<T>::value,
                R>;

template <typename Type>
struct SetLookupState : public KernelState {
  explicit SetLookupState(MemoryPool* pool) : lookup_table(pool, 0) {}

  Status Init(const SetLookupOptions& options) {
    using T = typename GetValueType<Type>::T;
    this->lookup_null_count = options.value_set->null_count();
    auto insert_value = [&](util::optional<T> v) {
      if (v.has_value()) {
        int32_t unused_memo_index;
        return lookup_table.GetOrInsert(*v, &unused_memo_index);
      } else if (!options.skip_nulls) {
        lookup_table.GetOrInsertNull();
      }
      return Status::OK();
    };
    return VisitArrayDataInline<Type>(*options.value_set->data(), insert_value);
  }

  using MemoTable = typename internal::HashTraits<Type>::MemoTableType;
  MemoTable lookup_table;
  int64_t lookup_null_count;
  int64_t null_index = -1;
};

template <>
struct SetLookupState<NullType> : public KernelState {
  explicit SetLookupState(MemoryPool*) {}

  Status Init(const SetLookupOptions& options) {
    this->lookup_null_count = options.value_set->null_count();
    return Status::OK();
  }

  int64_t lookup_null_count;
};

// Constructing the type requires a type parameter
struct InitStateVisitor {
  KernelContext* ctx;
  const SetLookupOptions* options;
  std::unique_ptr<KernelState> result;

  InitStateVisitor(KernelContext* ctx, const SetLookupOptions* options)
      : ctx(ctx), options(options) {}

  template <typename Type>
  Status Init() {
    using StateType = SetLookupState<Type>;
    result.reset(new StateType(ctx->exec_context()->memory_pool()));
    return static_cast<StateType*>(result.get())->Init(*options);
  }

  Status Visit(const DataType&) { return Init<NullType>(); }

  template <typename Type>
  enable_if_supports_set_lookup<Type, Status> Visit(const Type&) {
    return Init<Type>();
  }
  Status GetResult(std::unique_ptr<KernelState>* out) {
    RETURN_NOT_OK(VisitTypeInline(*options->value_set->type(), this));
    *out = std::move(result);
    return Status::OK();
  }
};

std::unique_ptr<KernelState> InitSetLookup(KernelContext* ctx,
                                           const KernelInitArgs& args) {
  InitStateVisitor visitor{ctx, static_cast<const SetLookupOptions*>(args.options)};
  std::unique_ptr<KernelState> result;
  ctx->SetStatus(visitor.GetResult(&result));
  return result;
}

struct MatchVisitor {
  KernelContext* ctx;
  const ArrayData& data;
  Datum* out;
  Int32Builder builder;

  MatchVisitor(KernelContext* ctx, const ArrayData& data, Datum* out)
      : ctx(ctx), data(data), out(out), builder(ctx->exec_context()->memory_pool()) {}

  Status Visit(const DataType&) {
    const auto& state = checked_cast<const SetLookupState<NullType>&>(*ctx->state());
    if (data.length != 0) {
      if (state.lookup_null_count == 0) {
        RETURN_NOT_OK(this->builder.AppendNulls(data.length));
      } else {
        RETURN_NOT_OK(this->builder.Reserve(data.length));
        for (int64_t i = 0; i < data.length; ++i) {
          this->builder.UnsafeAppend(0);
        }
      }
    }
    return Status::OK();
  }

  template <typename Type>
  enable_if_supports_set_lookup<Type, Status> Visit(const Type&) {
    using T = typename GetValueType<Type>::T;

    const auto& state = checked_cast<const SetLookupState<Type>&>(*ctx->state());

    int32_t null_index = state.lookup_table.GetNull();
    RETURN_NOT_OK(this->builder.Reserve(data.length));
    auto lookup_value = [&](util::optional<T> v) {
      if (v.has_value()) {
        int32_t index = state.lookup_table.Get(*v);
        if (index != -1) {
          // matching needle; output index from value_set
          this->builder.UnsafeAppend(index);
        } else {
          // no matching needle; output null
          this->builder.UnsafeAppendNull();
        }
      } else {
        if (null_index != -1) {
          // value_set included null
          this->builder.UnsafeAppend(null_index);
        } else {
          // value_set does not include null; output null
          this->builder.UnsafeAppendNull();
        }
      }
    };
    VisitArrayDataInline<Type>(data, lookup_value);
    return Status::OK();
  }

  Status Execute() {
    Status s = VisitTypeInline(*data.type, this);
    if (!s.ok()) {
      return s;
    }
    std::shared_ptr<ArrayData> out_data;
    RETURN_NOT_OK(this->builder.FinishInternal(&out_data));
    out->value = std::move(out_data);
    return Status::OK();
  }
};

void ExecMatch(KernelContext* ctx, const ExecBatch& batch, Datum* out) {
  MatchVisitor dispatch(ctx, *batch[0].array(), out);
  ctx->SetStatus(dispatch.Execute());
}

// ----------------------------------------------------------------------

// IsIn writes the results into a preallocated binary data bitmap
struct IsInVisitor {
  KernelContext* ctx;
  const ArrayData& data;
  Datum* out;

  IsInVisitor(KernelContext* ctx, const ArrayData& data, Datum* out)
      : ctx(ctx), data(data), out(out) {}

  Status Visit(const DataType&) {
    const auto& state = checked_cast<const SetLookupState<NullType>&>(*ctx->state());
    ArrayData* output = out->mutable_array();
    if (state.lookup_null_count > 0) {
      BitUtil::SetBitsTo(output->buffers[0]->mutable_data(), output->offset,
                         output->length, true);
      BitUtil::SetBitsTo(output->buffers[1]->mutable_data(), output->offset,
                         output->length, true);
    } else {
      BitUtil::SetBitsTo(output->buffers[1]->mutable_data(), output->offset,
                         output->length, false);
    }
    return Status::OK();
  }

  template <typename Type>
  enable_if_supports_set_lookup<Type, Status> Visit(const Type&) {
    using T = typename GetValueType<Type>::T;
    const auto& state = checked_cast<const SetLookupState<Type>&>(*ctx->state());
    ArrayData* output = out->mutable_array();

    if (this->data.GetNullCount() > 0 && state.lookup_null_count > 0) {
      // If there were nulls in the value set, set the whole validity bitmap to
      // true
      output->null_count = 0;
      BitUtil::SetBitsTo(output->buffers[0]->mutable_data(), output->offset,
                         output->length, true);
    }
    internal::FirstTimeBitmapWriter writer(output->buffers[1]->mutable_data(),
                                           output->offset, output->length);
    auto lookup_value = [&](util::optional<T> v) {
      if (!v.has_value() || state.lookup_table.Get(*v) != -1) {
        writer.Set();
      } else {
        writer.Clear();
      }
      writer.Next();
    };
    VisitArrayDataInline<Type>(this->data, std::move(lookup_value));
    writer.Finish();
    return Status::OK();
  }

  Status Execute() { return VisitTypeInline(*data.type, this); }
};

void ExecIsIn(KernelContext* ctx, const ExecBatch& batch, Datum* out) {
  IsInVisitor dispatch(ctx, *batch[0].array(), out);
  ctx->SetStatus(dispatch.Execute());
}

namespace codegen {

// Unary set lookup kernels available for the following input types
//
// * Null type
// * Boolean
// * Numeric
// * Simple temporal types (date, time, timestamp)
// * Base binary types
// * Decimal

void AddBasicSetLookupKernels(ScalarKernel kernel,
                              const std::shared_ptr<DataType>& out_ty,
                              ScalarFunction* func) {
  auto AddKernels = [&](const std::vector<std::shared_ptr<DataType>>& types) {
    for (const std::shared_ptr<DataType>& ty : types) {
      kernel.signature = KernelSignature::Make({InputType::Array(ty)}, out_ty);
      DCHECK_OK(func->AddKernel(kernel));
    }
  };

  AddKernels(BaseBinaryTypes());
  AddKernels(NumericTypes());
  AddKernels(TemporalTypes());

  std::vector<Type::type> other_types = {Type::BOOL, Type::DECIMAL,
                                         Type::FIXED_SIZE_BINARY};
  for (auto ty : other_types) {
    kernel.signature = KernelSignature::Make({InputType::Array(ty)}, out_ty);
    DCHECK_OK(func->AddKernel(kernel));
  }
}

}  // namespace codegen

namespace internal {

void RegisterScalarSetLookup(FunctionRegistry* registry) {
  // IsIn always writes into preallocated memory
  {
    ScalarKernel isin_base;
    isin_base.init = InitSetLookup;
    isin_base.exec = ExecIsIn;
    auto isin = std::make_shared<ScalarFunction>("isin", /*arity=*/1);

    codegen::AddBasicSetLookupKernels(isin_base, /*output_type=*/boolean(), isin.get());

    isin_base.signature = KernelSignature::Make({InputType::Array(null())}, boolean());
    isin_base.null_handling = NullHandling::COMPUTED_PREALLOCATE;
    DCHECK_OK(isin->AddKernel(isin_base));
    DCHECK_OK(registry->AddFunction(isin));
  }

  // Match uses Int32Builder and so is responsible for all its own allocation
  {
    ScalarKernel match_base;
    match_base.init = InitSetLookup;
    match_base.exec = ExecMatch;
    match_base.null_handling = NullHandling::COMPUTED_NO_PREALLOCATE;
    match_base.mem_allocation = MemAllocation::NO_PREALLOCATE;
    auto match = std::make_shared<ScalarFunction>("match", /*arity=*/1);
    codegen::AddBasicSetLookupKernels(match_base, /*output_type=*/int32(), match.get());

    match_base.signature = KernelSignature::Make({InputType::Array(null())}, int32());
    DCHECK_OK(match->AddKernel(match_base));
    DCHECK_OK(registry->AddFunction(match));
  }
}

}  // namespace internal
}  // namespace compute
}  // namespace arrow
