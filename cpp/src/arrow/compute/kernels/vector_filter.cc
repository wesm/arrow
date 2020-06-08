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

#include "arrow/array/array_base.h"
#include "arrow/array/array_primitive.h"
#include "arrow/compute/api_vector.h"
#include "arrow/compute/kernels/common.h"
#include "arrow/compute/kernels/vector_selection_internal.h"
#include "arrow/record_batch.h"
#include "arrow/result.h"
#include "arrow/visitor_inline.h"

namespace arrow {
namespace compute {
namespace internal {

// IndexSequence which yields the indices of positions in a BooleanArray
// which are either null or true
template <FilterOptions::NullSelectionBehavior NullSelectionBehavior>
class FilterIndexSequence {
 public:
  // constexpr so we'll never instantiate bounds checking
  constexpr bool never_out_of_bounds() const { return true; }
  void set_never_out_of_bounds() {}

  constexpr FilterIndexSequence() = default;

  FilterIndexSequence(const BooleanArray& filter, int64_t out_length)
      : filter_(&filter), out_length_(out_length) {}

  std::pair<int64_t, bool> Next() {
    if (NullSelectionBehavior == FilterOptions::DROP) {
      // skip until an index is found at which the filter is true
      while (filter_->IsNull(index_) || !filter_->Value(index_)) {
        ++index_;
      }
      return std::make_pair(index_++, true);
    }

    // skip until an index is found at which the filter is either null or true
    while (filter_->IsValid(index_) && !filter_->Value(index_)) {
      ++index_;
    }
    bool is_valid = filter_->IsValid(index_);
    return std::make_pair(index_++, is_valid);
  }

  int64_t length() const { return out_length_; }

  int64_t null_count() const {
    if (NullSelectionBehavior == FilterOptions::DROP) {
      return 0;
    }
    return filter_->null_count();
  }

 private:
  const BooleanArray* filter_ = nullptr;
  int64_t index_ = 0, out_length_ = -1;
};

int64_t FilterOutputSize(FilterOptions::NullSelectionBehavior null_selection,
                         const Array& arr) {
  const auto& filter = checked_cast<const BooleanArray&>(arr);
  // TODO(bkietz) this can be optimized. Use Bitmap::VisitWords
  int64_t size = 0;
  if (null_selection == FilterOptions::EMIT_NULL) {
    for (auto i = 0; i < filter.length(); ++i) {
      if (filter.IsNull(i) || filter.Value(i)) {
        ++size;
      }
    }
  } else {
    for (auto i = 0; i < filter.length(); ++i) {
      if (filter.IsValid(i) && filter.Value(i)) {
        ++size;
      }
    }
  }
  return size;
}

struct FilterState : public KernelState {
  explicit FilterState(FilterOptions options) : options(std::move(options)) {}
  FilterOptions options;
};

std::unique_ptr<KernelState> InitFilter(KernelContext*, const KernelInitArgs& args) {
  FilterOptions options;
  if (args.options == nullptr) {
    options = FilterOptions::Defaults();
  } else {
    options = *static_cast<const FilterOptions*>(args.options);
  }
  return std::unique_ptr<KernelState>(new FilterState(std::move(options)));
}

template <typename ValueType>
struct FilterFunctor {
  using ArrayType = typename TypeTraits<ValueType>::ArrayType;

  template <FilterOptions::NullSelectionBehavior NullSelection>
  static void ExecImpl(KernelContext* ctx, const ExecBatch& batch, Datum* out) {
    using IS = FilterIndexSequence<NullSelection>;
    ArrayType values(batch[0].array());
    BooleanArray filter(batch[1].array());
    const int64_t output_size = FilterOutputSize(NullSelection, filter);
    std::shared_ptr<Array> result;
    KERNEL_RETURN_IF_ERROR(ctx, Select(ctx, values, IS(filter, output_size), &result));
    out->value = result->data();
  }

  static void Exec(KernelContext* ctx, const ExecBatch& batch, Datum* out) {
    const auto& state = checked_cast<const FilterState&>(*ctx->state());
    if (state.options.null_selection_behavior == FilterOptions::EMIT_NULL) {
      ExecImpl<FilterOptions::EMIT_NULL>(ctx, batch, out);
    } else {
      ExecImpl<FilterOptions::DROP>(ctx, batch, out);
    }
  }
};

struct FilterKernelVisitor {
  template <typename Type>
  Status Visit(const Type&) {
    this->result = FilterFunctor<Type>::Exec;
    return Status::OK();
  }

  Status Create(const DataType& type) { return VisitTypeInline(type, this); }
  ArrayKernelExec result;
};

Status GetFilterKernel(const DataType& type, ArrayKernelExec* exec) {
  FilterKernelVisitor visitor;
  RETURN_NOT_OK(visitor.Create(type));
  *exec = visitor.result;
  return Status::OK();
}

Result<std::shared_ptr<RecordBatch>> FilterRecordBatch(const RecordBatch& batch,
                                                       const Datum& filter,
                                                       const FunctionOptions* options,
                                                       ExecContext* ctx) {
  if (!filter.is_array()) {
    return Status::Invalid("Cannot filter a RecordBatch with a filter of kind ",
                           filter.kind());
  }

  const auto& filter_opts = *static_cast<const FilterOptions*>(options);
  // TODO: Rewrite this to convert to selection vector and use Take
  std::vector<std::shared_ptr<Array>> columns(batch.num_columns());
  for (int i = 0; i < batch.num_columns(); ++i) {
    ARROW_ASSIGN_OR_RAISE(Datum out,
                          Filter(batch.column(i)->data(), filter, filter_opts, ctx));
    columns[i] = out.make_array();
  }

  int64_t out_length;
  if (columns.size() == 0) {
    out_length =
        FilterOutputSize(filter_opts.null_selection_behavior, *filter.make_array());
  } else {
    out_length = columns[0]->length();
  }
  return RecordBatch::Make(batch.schema(), out_length, columns);
}

Result<std::shared_ptr<Table>> FilterTable(const Table& table, const Datum& filter,
                                           const FunctionOptions* options,
                                           ExecContext* ctx) {
  auto new_columns = table.columns();
  for (auto& column : new_columns) {
    ARROW_ASSIGN_OR_RAISE(
        Datum out_column,
        Filter(column, filter, *static_cast<const FilterOptions*>(options), ctx));
    column = out_column.chunked_array();
  }
  return Table::Make(table.schema(), std::move(new_columns));
}

class FilterMetaFunction : public MetaFunction {
 public:
  FilterMetaFunction() : MetaFunction("filter", Arity::Binary()) {}

  Result<Datum> ExecuteImpl(const std::vector<Datum>& args,
                            const FunctionOptions* options,
                            ExecContext* ctx) const override {
    if (args[0].kind() == Datum::RECORD_BATCH) {
      auto values_batch = args[0].record_batch();
      ARROW_ASSIGN_OR_RAISE(
          std::shared_ptr<RecordBatch> out_batch,
          FilterRecordBatch(*args[0].record_batch(), args[1], options, ctx));
      return Datum(out_batch);
    } else if (args[0].kind() == Datum::TABLE) {
      ARROW_ASSIGN_OR_RAISE(std::shared_ptr<Table> out_table,
                            FilterTable(*args[0].table(), args[1], options, ctx));
      return Datum(out_table);
    } else {
      return CallFunction("array_filter", args, options, ctx);
    }
  }
};

void RegisterVectorFilter(FunctionRegistry* registry) {
  VectorKernel base;
  base.init = InitFilter;

  auto filter = std::make_shared<VectorFunction>("array_filter", Arity::Binary());
  InputType filter_ty = InputType::Array(boolean());
  OutputType out_ty(FirstType);

  auto AddKernel = [&](InputType in_ty, const DataType& example_type) {
    base.signature = KernelSignature::Make({in_ty, filter_ty}, out_ty);
    DCHECK_OK(GetFilterKernel(example_type, &base.exec));
    DCHECK_OK(filter->AddKernel(base));
  };

  for (const auto& value_ty : PrimitiveTypes()) {
    AddKernel(InputType::Array(value_ty), *value_ty);
  }
  // Other types where we may only on the DataType::id
  for (const auto& value_ty : ExampleParametricTypes()) {
    AddKernel(InputType::Array(value_ty->id()), *value_ty);
  }
  DCHECK_OK(registry->AddFunction(std::move(filter)));

  // Add filter metafunction
  DCHECK_OK(registry->AddFunction(std::make_shared<FilterMetaFunction>()));
}

}  // namespace internal
}  // namespace compute
}  // namespace arrow
