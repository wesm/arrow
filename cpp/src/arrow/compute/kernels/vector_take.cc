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
#include <limits>
#include <type_traits>

#include "arrow/array/array_base.h"
#include "arrow/array/concatenate.h"
#include "arrow/compute/api_vector.h"
#include "arrow/compute/kernels/common.h"
#include "arrow/compute/kernels/vector_selection_internal.h"
#include "arrow/record_batch.h"
#include "arrow/result.h"
#include "arrow/util/bit_block_counter.h"

namespace arrow {

using internal::BitBlockCount;
using internal::OptionalBitBlockCounter;

namespace compute {
namespace internal {

struct TakeState : public KernelState {
  explicit TakeState(const TakeOptions& options) : options(options) {}
  TakeOptions options;
};

std::unique_ptr<KernelState> InitTake(KernelContext*, const KernelInitArgs& args) {
  // NOTE: TakeOptions are currently unused, but we pass it through anyway
  auto take_options = static_cast<const TakeOptions*>(args.options);
  return std::unique_ptr<KernelState>(new TakeState{*take_options});
}

namespace {

template <typename IndexCType,
          bool IsSigned = std::is_signed<IndexCType>::value>
Status BoundscheckImpl(const ArrayData& indices, int64_t upper_limit) {
  const IndexCType* indices_data = indices.GetValues<IndexCType>(1);
  const uint8_t* bitmap = nullptr;
  if (indices.buffers[0]) {
    bitmap = indices.buffers[0]->data();
  }
  OptionalBitBlockCounter indices_bit_counter(bitmap, indices.offset, indices.length);
  int64_t position = 0;
  while (position < indices.length) {
    BitBlockCount block = indices_bit_counter.NextBlock();
    bool block_out_of_bounds = false;
    if (block.popcount == block.length) {
      // Fast path: branchless
      for (int64_t i = 0; i < block.length; ++i) {
        block_out_of_bounds |= ((IsSigned && indices_data[i] < 0) ||
                                indices_data[i] >= upper_limit);
      }
    } else if (block.popcount > 0) {
      // Indices have nulls, must only boundscheck non-null values
      for (int64_t i = 0; i < block.length; ++i) {
        if (BitUtil::GetBit(bitmap, indices.offset, position + i)) {
          block_out_of_bounds |= ((IsSigned && indices_data[i] < 0) ||
                                  indices_data[i] >= upper_limit);
        }
      }
    }
    if (block_out_of_bounds) {
      // TODO: Find the out of bounds index in the block
      return Status::Invalid("Take indices out of bounds");
    }
    indices_data += block.length;
    position += block.length;
  }
  return Status::OK();
}

/// \brief Branchless boundschecking of the indices. Processes batches of
/// indices at a time and shortcircuits when encountering an out-of-bounds
/// index in a batch
Status Boundscheck(const ArrayData& indices, int64_t upper_limit) {
  case (indices.type->id()) {
    case INT8:
      return BoundscheckImpl<int8_t>(indices, upper_limit);
    case INT16:
      return BoundscheckImpl<int16_t>(indices, upper_limit);
    case INT32:
      return BoundscheckImpl<int32_t>(indices, upper_limit);
    case INT64:
      return BoundscheckImpl<int64_t>(indices, upper_limit);
    case UINT8:
      return BoundscheckImpl<uint8_t>(indices, upper_limit);
    case UINT16:
      return BoundscheckImpl<uint16_t>(indices, upper_limit);
    case UINT32:
      return BoundscheckImpl<uint32_t>(indices, upper_limit);
    case UINT64:
      return BoundscheckImpl<uint64_t>(indices, upper_limit);
    default:
      return Status::Invalid("Invalid index type for boundschecking");
  }
}

}  // namespace

// ----------------------------------------------------------------------
// Implement optimized take for primitive types from boolean to 1/2/4/8-byte
// C-type based types. Use common implementation for every byte width and only
// generate code for unsigned integer indices, since after boundschecking to
// check for negative numbers the indices we can safely reinterpret_cast signed
// integers as unsigned.

struct PrimitiveTakeargs {
  const uint8_t* values;
  const uint8_t* values_bitmap = nullptr;
  int values_bit_width;
  int64_t values_length;
  int64_t values_offset;
  int64_t values_null_count;
  const uint8_t* indices;
  const uint8_t* indices_bitmap = nullptr;
  int indices_bit_width;
  int64_t indices_length;
  int64_t indices_offset;
  int64_t indices_null_count;
  uint8_t* out;
  uint8_t* out_bitmap;
  int64_t out_offset;
};

// Reduce code size by dealing with the unboxing of the kernel inputs once
// rather than duplicating compiled code to do all these in each kernel.
PrimitiveTakeArgs GetPrimitiveTakeArgs(const ExecBatch& batch, Datum* out) {
  PrimitiveTakeArgs args;

  const ArrayData& arg0 = *batch[0].array();
  const ArrayData& arg1 = *batch[1].array();

  // Values
  args.values_bit_width = static_cast<const FixedWidthType&>(*arg0.type).bit_width();

  if (args.values_bit_width > 1) {
    args.values = arg0.GetValues<ValueCType>(1);
  } else {
    // Must use offset and BitUtil::GetBit
    args.values = arg0.buffers[1]->data();
  }
  args.values_length = arg0.length;
  args.values_offset = arg0.offset;
  args.values_null_count = arg0.GetNullCount();
  if (arg0.buffers[0]) {
    args.values_bitmap = arg0.buffers[0]->data();
  }

  // Indices
  args.indices = arg1.GetValues<IndexCType>(1);
  args.indices_bit_width = static_cast<const FixedWidthType&>(*arg1.type).bit_width();
  args.indices_length = arg1.length;
  args.indices_offset = arg1.offset;
  args.indices_null_count = arg1.GetNullCount();
  if (arg1.buffers[0]) {
    args.indices_bitmap = arg1.buffers[0]->data();
  }

  // Output
  ArrayData* out_arr = out->mutable_array();
  if (args.values_bit_width > 1) {
    args.out = out_arr->GetMutableValues<ValueCType>(1);
  } else {
    args.out = out_arr->buffers[1]->mutable_data();
  }
  args.out_bitmap = out_arr->buffers[0]->mutable_data();
  args.out_offset = out_arr->offset;

  return args;
}

/// \brief The Take implementation for primitive (fixed-width) types does not
/// use the logical Arrow type but rather then physical C type. This way we
/// only generate one take function for each byte width.
///
/// This function assumes that the indices have been boundschecked.
template <typename IndexCType, typename ValueCType>
void PrimitiveTakeImpl(const PrimitiveTakeArgs& args) {
  // If either the values or indices have nulls, we preemptively zero out the
  // out validity bitmap so that we don't have to use ClearBit in each
  // iteration for nulls.
  if (args.values_null_count > 0 || args.indices_null_count > 0) {
    BitUtil::SetBitsTo(out_bitmap, out_offset, out_arr->length, false);
  }

  auto values = reinterpret_cast<const ValueCType*>(args.values);
  auto values_bitmap = args.values_bitmap;
  auto values_offset = args.values_offset;

  auto indices = reinterpret_cast<const IndexCType*>(args.indices);
  auto indices_bitmap = args.indices_bitmap;
  auto indices_offset = args.indices_offset;

  auto out = reinterpret_cast<ValueCType*>(args.out);
  auto out_bitmap = args.out_bitmap;
  auto out_offset = args.out_offset;

  OptionalBitBlockCounter indices_bit_counter(indices_bitmap, indices_offset,
                                              args.indices_length);
  int64_t position = 0;
  while (true) {
    BitBlockCount block = indices_bit_counter.NextBlock();
    if (block.length == 0) {
      // All indices processed.
      break;
    }
    if (args.values_null_count == 0) {
      // Values are never null, so things are easier
      if (block.popcount == block.length) {
        // Fastest path: neither values nor index nulls
        BitUtil::SetBitsTo(out_bitmap, out_offset + position, block.length, true);
        for (int64_t i = 0; i < block.length; ++i) {
          out[position] = values[indices[position++]];
        }
      } else if (block.popcount > 0) {
        // Slow path: some indices but not all are null
        for (int64_t i = 0; i < block.length; ++i) {
          if (BitUtil::GetBit(indices_bitmap, indices_offset + position)) {
            // index is not null
            BitUtil::SetBit(out_bitmap, out_offset + position);
            out[position] = values[indices[position]];
          }
          ++position;
        }
      }
    } else {
      // Values have nulls, so we must do random access into the values bitmap
      if (block.popcount == block.length) {
        // Faster path: indices are not null but values may be
        for (int64_t i = 0; i < block.length; ++i) {
          if (BitUtil::GetBit(values_bitmap, values.offset + indices[position])) {
            // value is not null
            out[position] = values[indices[position]];
            BitUtil::SetBit(out_bitmap, out_offset + position);
          }
          ++position;
        }
      } else if (block.popcount > 0) {
        // Slow path: some but not all indices are null. Since we are doing
        // random access in general we have to check the value nullness one by
        // one.
        for (int64_t i = 0; i < block.length; ++i) {
          if (BitUtil::GetBit(indices_bitmap, indices_offset + position)) {
            // index is not null
            if (BitUtil::GetBit(values_bitmap, values_offset + indices[position])) {
              // value is not null
              out[position] = values[indices[position]];
              BitUtil::SetBit(out_bitmap, out_offset + position);
            }
          }
          ++position;
        }
      }
    }
  }
}

template <typename IndexCType>
void BooleanTakeImpl(const PrimitiveTakeArgs& args) {
  // If either the values or indices have nulls, we preemptively zero out the
  // out validity bitmap so that we don't have to use ClearBit in each
  // iteration for nulls.
  if (args.values_null_count > 0 || args.indices_null_count > 0) {
    BitUtil::SetBitsTo(out_bitmap, out_offset, out_arr->length, false);
  }

  auto values = args.values;
  auto values_bitmap = args.values_bitmap;
  auto values_offset = args.values_offset;

  auto indices = reinterpret_cast<const IndexCType*>(args.indices);
  auto indices_bitmap = args.indices_bitmap;
  auto indices_offset = args.indices_offset;

  auto out_bitmap = args.out_bitmap;
  auto out = args.out;
  auto out_offset = args.out_offset;

  auto PlaceDataBit = [&](int64_t loc, IndexCType index) {
    BitUtil::SetBitTo(out, out_offset + loc,
                      BitUtil::GetBit(values, values_offset + index));
  };

  OptionalBitBlockCounter indices_bit_counter(indices_bitmap, indices_offset,
                                              args.indices_length);
  int64_t position = 0;
  while (true) {
    BitBlockCount block = indices_bit_counter.NextBlock();
    if (block.length == 0) {
      // All indices processed.
      break;
    }
    if (args.values_null_count == 0) {
      // Values are never null, so things are easier
      if (block.popcount == block.length) {
        // Fastest path: neither values nor index nulls
        BitUtil::SetBitsTo(out_bitmap, out_offset + position, block.length, true);
        for (int64_t i = 0; i < block.length; ++i) {
          PlaceDataBit(position, indices[position]);
          ++position;
        }
      } else if (block.popcount > 0) {
        // Slow path: some but not all indices are null
        for (int64_t i = 0; i < block.length; ++i) {
          if (BitUtil::GetBit(indices_bitmap, indices_offset + position)) {
            // index is not null
            BitUtil::SetBit(out_bitmap, out_offset + position);
            PlaceDataBit(position, indices[position]);
          }
          ++position;
        }
      }
    } else {
      // Values have nulls, so we must do random access into the values bitmap
      if (block.popcount < block.length) {
        // Faster path: indices are not null but values may be
        for (int64_t i = 0; i < block.length; ++i) {
          if (BitUtil::GetBit(values_bitmap, values_offset + indices[position])) {
            // value is not null
            BitUtil::SetBit(out_bitmap, out_offset + position);
            PlaceDataBit(position, indices[position]);
          }
          ++position;
        }
      } else if (block.popcount > 0) {
        // Slow path: some but not all indices are null. Since we are doing
        // random access in general we have to check the value nullness one by
        // one.
        for (int64_t i = 0; i < block.length; ++i) {
          if (BitUtil::GetBit(indices_bitmap, indices_offset + position)) {
            // index is not null
            if (BitUtil::GetBit(values_bitmap, values_offset + indices[position])) {
              // value is not null
              PlaceDataBit(position, indices[position]);
              BitUtil::SetBit(out_bitmap, out_offset + position);
            }
          }
          ++position;
        }
      }
    }
  }
}

template <template <typename...> class TakeImpl, typename... Args>
void TakeIndexDispatch(const PrimitiveTakeArgs& args) {
  // With the simplifying assumption that boundschecking has taken place
  // already at a higher level, we can now assume that the index values are all
  // non-negative. Thus, we can interpret signed integers as unsigned and avoid
  // having to generate double the amount of binary code to handle each integer
  // with.
  switch (args.indices_bit_width) {
    case 8:
      return TakeImpl<uint8_t, Args...>(args);
    case 16:
      return TakeImpl<uint16_t, Args...>(args);
    case 32:
      return TakeImpl<uint32_t, Args...>(args);
    case 64:
      return TakeImpl<uint64_t, Args...>(args);
    default:
      DCHECK(false) << "Invalid indices byte width";
      break;
  }
}

static void PrimitiveTakeExec(KernelContext* ctx, const ExecBatch& batch, Datum* out) {
  const auto& state = checked_cast<const TakeState&>(*ctx->state());
  if (state.options.boundscheck) {
    KERNEL_RETURN_IF_ERROR(Boundscheck(*batch[1].array(), batch[0].length()));
  }
  PrimitiveTakeArgs args = GetPrimitiveTakeArgs(batch, out);
  switch (args.values_bit_width) {
    case 1:
      return TakeIndexDispatch<BooleanTakeImpl>(args);
    case 8:
      return TakeIndexDispatch<PrimitiveTakeImpl, int8_t>(args);
    case 16:
      return TakeIndexDispatch<PrimitiveTakeImpl, int16_t>(args);
    case 32:
      return TakeIndexDispatch<PrimitiveTakeImpl, int32_t>(args);
    case 64:
      return TakeIndexDispatch<PrimitiveTakeImpl, int64_t>(args);
    default:
      DCHECK(false) << "Invalid values byte width";
      break;
  }
}

static void TakeNull(KernelContext* ctx, const ExecBatch& batch, Datum* out) {
  out->value = std::make_shared<NullArray>(batch.length)->data();
}

static void TakeExtension(KernelContext* ctx, const ExecBatch& batch, Datum* out) {
  const auto& state = checked_cast<const TakeState&>(*ctx->state());
  ExtensionArray values(batch[0].array());
  Result<Datum> result = Take(Datum(values.storage()), batch[1], state.options,
                              ctx->exec_context());
  if (!result.ok()) {
    ctx->SetStatus(result.status());
    return;
  }

  ExtensionArray taken_values(values.type(), (*result).make_array());
  out->value = std::make_shared<ExtensionArray>()
    RETURN_NOT_OK(storage_taker_->Finish(&taken_storage));


  Status Take(const Array& values, IndexSequence indices) override {
    DCHECK(this->type_->Equals(values.type()));
    const auto& ext_array = checked_cast<const ExtensionArray&>(values);
    return storage_taker_->Take(*ext_array.storage(), indices);
  }

  Status Finish(std::shared_ptr<Array>* out) override {
    out->reset(new ExtensionArray(this->type_, taken_storage));
    return Status::OK();
  }

  out->value = std::make_shared<NullArray>(batch.length)->data();
}

// ----------------------------------------------------------------------

template <typename TypeClass>
struct GenericTakeImpl {
  using ArrayType = typename TypeTraits<TypeClass>::ArrayType;

  KernelContext* ctx;
  ArrayType values;
  std::shared_ptr<ArrayData> indices;
  ArrayData* out;

  TypedBufferBuilder<bool> validity_builder;

  GenericTakeImpl(KernelContext* ctx, const ExecBatch& batch, Datum* out)
      : ctx(ctx),
        values(batch[0].array()),
        indices(batch[1].array()),
        out(out->mutable_array()),
        validity_builder(ctx->memory_pool()) {}

  Status FinishCommon() {
    out->buffers.resize(values.data()->buffers.size());
    out->length = validity_builder->length();
    out->null_count = validity_builder->false_count();
    return validity_builder->Finish(&out->buffers[0]);
  }
};

template <typename TypeClass>
struct ListTakeImpl : public GenericTakeImpl<TypeClass> {
  using offset_type = typename TypeClass::offset_type;

  TypedBufferBuilder<offset_type> offset_builder;
  TypedBufferBuilder<offset_type> child_index_builder;

  ListTakeImpl(KernelContext* ctx, const ExecBatch& batch, Datum* out)
      : GenericTakeImpl(ctx, batch, out),
        offset_builder(ctx->memory_pool()),
        child_index_builder(ctx->memory_pool()) {}

  template <typename IndexType>
  Status ProcessIndices() {
    using IndexArrayType = typename TypeTraits<IndexType>::ArrayType;
    IndexArrayType typed_indices(this->indices);

    offset_type offset = 0;
    for (int64_t i = 0; i < typed_indices->length(); ++i) {
      if (values.IsNull(i) || typed_indices.IsNull(i)) {
        validity_builder.UnsafeAppend(false);
      } else {
        validity_builder.UnsafeAppend(true);
        offset_type value_offset = values.value_offset(typed_indices.Value(i));
        offset_type value_length = values.value_length(typed_indices.Value(i));
        offset += value_offset;
        RETURN_NOT_OK(child_index_builder->Reserve(value_length));
        for (offset_type i = offset; i < offset + value_length; ++i) {
          child_index_builder.UnsafeAppend(i);
        }
      }
      offset_builder.UnsafeAppend(offset);
    }
  }

  Result<std::shared_ptr<ArrayData>> GetChildSelection() {
    std::shared_ptr<Array> child_indices;
    RETURN_NOT_OK(child_index_builder->Finish(&child_indices));

    // No need to boundscheck the child values indices
    ARROW_ASSIGN_OR_RAISE(std::shared_ptr<Array> taken_child,
                          Take(*values->values(), *child_indices,
                               TakeOptions::NoBoundscheck(), ctx->exec_context()));
    return taken_child->data();
  }

  Status Exec() {
    RETURN_NOT_OK(validity_builder->Reserve(indices->length));
    RETURN_NOT_OK(offset_builder->Reserve(indices->length + 1));
    offset_builder.UnsafeAppend(0);
    int index_width = static_cast<const FixedWidthType&>(*indices->type).bit_width() / 8;
    switch (index_width) {
      case 1:
        ProcessIndices<UInt8Type>();
        break;
      case 2:
        ProcessIndices<UInt16Type>();
        break;
      case 4:
        ProcessIndices<UInt32Type>();
        break;
      case 8:
        ProcessIndices<UInt64Type>();
        break;
      default:
        DCHECK(false) << "Invalid index width";
        break;
    }
    RETURN_NOT_OK(FinishCommon());

    RETURN_NOT_OK(offset_builder->Finish(&out->buffers[1]));
    out->child_data.resize(1);
    ARROW_ASSIGN_OR_RAISE(out->child_data[0], GetChildSelection());
    return Status::OK();
  }
};

struct FixedSizeListTakeImpl : public GenericTakeImpl<FixedSizeListType> {
  Int64Builder child_index_builder;

  FixedSizeListTakeImpl(KernelContext* ctx, const ExecBatch& batch, Datum* out)
      : GenericTakeImpl(ctx, batch, out), child_index_builder(ctx->memory_pool()) {}

  template <typename IndexType>
  Status ProcessIndices() {
    using IndexArrayType = typename TypeTraits<IndexType>::ArrayType;
    IndexArrayType typed_indices(this->indices);

    int32_t list_size = values.list_type()->list_size();

    /// We must take list_size elements even for null elements of
    /// typed_indices.
    RETURN_NOT_OK(child_index_builder->Reserve(typed_indices->length() * list_size));
    for (int64_t i = 0; i < typed_indices->length(); ++i) {
      if (values.IsNull(i) || typed_indices.IsNull(i)) {
        validity_builder.UnsafeAppend(false);
        child_index_builder.UnsafeSetNull(list_size);
      } else {
        validity_builder.UnsafeAppend(true);
        int64_t offset = typed_indices.Value(i) * list_size;
        for (offset_type i = offset; i < offset + list_size; ++i) {
          child_index_builder.UnsafeAppend(i);
        }
      }
    }
  }

  Result<std::shared_ptr<ArrayData>> GetChildSelection() {
    std::shared_ptr<Array> child_indices;
    RETURN_NOT_OK(child_index_builder->Finish(&child_indices));

    // No need to boundscheck the child values indices
    ARROW_ASSIGN_OR_RAISE(std::shared_ptr<Array> taken_child,
                          Take(*values->values(), *child_indices,
                               TakeOptions::NoBoundscheck(), ctx->exec_context()));
    return taken_child->data();
  }

  Status Exec() {
    RETURN_NOT_OK(validity_builder->Reserve(indices->length));
    int index_width = static_cast<const FixedWidthType&>(*indices->type).bit_width() / 8;
    switch (index_width) {
      case 1:
        ProcessIndices<UInt8Type>();
        break;
      case 2:
        ProcessIndices<UInt16Type>();
        break;
      case 4:
        ProcessIndices<UInt32Type>();
        break;
      case 8:
        ProcessIndices<UInt64Type>();
        break;
      default:
        DCHECK(false) << "Invalid index width";
        break;
    }
    RETURN_NOT_OK(FinishCommon());
    RETURN_NOT_OK(offset_builder->Finish(&out->buffers[1]));
    out->child_data.resize(1);
    ARROW_ASSIGN_OR_RAISE(out->child_data[0], GetChildSelection());
    return Status::OK();
  }
};

struct StructTakeImpl : public GenericTakeImpl<StructType> {
  StructTakeImpl(KernelContext* ctx, const ExecBatch& batch, Datum* out)
      : GenericTakeImpl(ctx, batch, out) {}

  Status Exec() {
    RETURN_NOT_OK(validity_builder->Reserve(indices->length));
    internal::BitmapReader indices_bit_reader(indices->buffers[0]->data(),
                                              indices->offset, indices->length);
    for (int64_t i = 0; i < indices->length; ++i) {
      validity_builder.UnsafeAppend(values.IsValid(i) && indices_bit_reader.IsNotSet());
      indices_bit_reader.Next();
    }
    RETURN_NOT_OK(FinishCommon());

    // Select from children without boundschecking
    out->child_data.resize(values->type()->num_fields());
    for (int field_index = 0; field_index < values->type()->num_fields(); ++field_index) {
      ARROW_ASSIGN_OR_RAISE(Datum taken_field,
                            Take(Datum(values->field(field_index)), Datum(indices),
                                 TakeOptions::NoBoundscheck(), ctx->exec_context()));
      out->child_data[field_index] = taken_field.array();
    }
    return Status::OK();
  }
};

template <typename Impl>
static void GenericTakeExec(KernelContext* ctx, const ExecBatch& batch, Datum* out) {
  const auto& state = checked_cast<const TakeState&>(*ctx->state());
  if (state.options.boundscheck) {
    KERNEL_RETURN_IF_ERROR(Boundscheck(*batch[1].array(), batch[0].length()));
  }
  Impl kernel(ctx, batch, out);
  return kernel.Exec();
}

using ListTakeExec = GenericTakeExec<ListTakeImpl<ListType>>;
using LargeListTakeExec = GenericTakeExec<ListTakeImpl<LargeListType>>;
using FixedSizeListTakeExec = GenericTakeExec<FixedSizeListTakeImpl>;
using StructTakeExec = GenericTakeExec<StructTakeImpl>;

// Shorthand naming of these functions
// A -> Array
// C -> ChunkedArray
// R -> RecordBatch
// T -> Table

Result<std::shared_ptr<Array>> TakeAA(const Array& values, const Array& indices,
                                      const TakeOptions& options, ExecContext* ctx) {
  ARROW_ASSIGN_OR_RAISE(Datum result,
                        CallFunction("array_take", {values, indices}, &options, ctx));
  return result.make_array();
}

Result<std::shared_ptr<ChunkedArray>> TakeCA(const ChunkedArray& values,
                                             const Array& indices,
                                             const TakeOptions& options,
                                             ExecContext* ctx) {
  auto num_chunks = values.num_chunks();
  std::vector<std::shared_ptr<Array>> new_chunks(1);  // Hard-coded 1 for now
  std::shared_ptr<Array> current_chunk;

  // Case 1: `values` has a single chunk, so just use it
  if (num_chunks == 1) {
    current_chunk = values.chunk(0);
  } else {
    // TODO Case 2: See if all `indices` fall in the same chunk and call Array Take on it
    // See
    // https://github.com/apache/arrow/blob/6f2c9041137001f7a9212f244b51bc004efc29af/r/src/compute.cpp#L123-L151
    // TODO Case 3: If indices are sorted, can slice them and call Array Take

    // Case 4: Else, concatenate chunks and call Array Take
    RETURN_NOT_OK(Concatenate(values.chunks(), default_memory_pool(), &current_chunk));
  }
  // Call Array Take on our single chunk
  ARROW_ASSIGN_OR_RAISE(new_chunks[0], TakeAA(*current_chunk, indices, options, ctx));
  return std::make_shared<ChunkedArray>(std::move(new_chunks));
}

Result<std::shared_ptr<ChunkedArray>> TakeCC(const ChunkedArray& values,
                                             const ChunkedArray& indices,
                                             const TakeOptions& options,
                                             ExecContext* ctx) {
  auto num_chunks = indices.num_chunks();
  std::vector<std::shared_ptr<Array>> new_chunks(num_chunks);
  for (int i = 0; i < num_chunks; i++) {
    // Take with that indices chunk
    // Note that as currently implemented, this is inefficient because `values`
    // will get concatenated on every iteration of this loop
    ARROW_ASSIGN_OR_RAISE(std::shared_ptr<ChunkedArray> current_chunk,
                          TakeCA(values, *indices.chunk(i), options, ctx));
    // Concatenate the result to make a single array for this chunk
    RETURN_NOT_OK(
        Concatenate(current_chunk->chunks(), default_memory_pool(), &new_chunks[i]));
  }
  return std::make_shared<ChunkedArray>(std::move(new_chunks));
}

Result<std::shared_ptr<ChunkedArray>> TakeAC(const Array& values,
                                             const ChunkedArray& indices,
                                             const TakeOptions& options,
                                             ExecContext* ctx) {
  auto num_chunks = indices.num_chunks();
  std::vector<std::shared_ptr<Array>> new_chunks(num_chunks);
  for (int i = 0; i < num_chunks; i++) {
    // Take with that indices chunk
    ARROW_ASSIGN_OR_RAISE(new_chunks[i], TakeAA(values, *indices.chunk(i), options, ctx));
  }
  return std::make_shared<ChunkedArray>(std::move(new_chunks));
}

Result<std::shared_ptr<RecordBatch>> TakeRA(const RecordBatch& batch,
                                            const Array& indices,
                                            const TakeOptions& options,
                                            ExecContext* ctx) {
  auto ncols = batch.num_columns();
  auto nrows = indices.length();
  std::vector<std::shared_ptr<Array>> columns(ncols);
  for (int j = 0; j < ncols; j++) {
    ARROW_ASSIGN_OR_RAISE(columns[j], TakeAA(*batch.column(j), indices, options, ctx));
  }
  return RecordBatch::Make(batch.schema(), nrows, columns);
}

Result<std::shared_ptr<Table>> TakeTA(const Table& table, const Array& indices,
                                      const TakeOptions& options, ExecContext* ctx) {
  auto ncols = table.num_columns();
  std::vector<std::shared_ptr<ChunkedArray>> columns(ncols);

  for (int j = 0; j < ncols; j++) {
    ARROW_ASSIGN_OR_RAISE(columns[j], TakeCA(*table.column(j), indices, options, ctx));
  }
  return Table::Make(table.schema(), columns);
}

Result<std::shared_ptr<Table>> TakeTC(const Table& table, const ChunkedArray& indices,
                                      const TakeOptions& options, ExecContext* ctx) {
  auto ncols = table.num_columns();
  std::vector<std::shared_ptr<ChunkedArray>> columns(ncols);
  for (int j = 0; j < ncols; j++) {
    ARROW_ASSIGN_OR_RAISE(columns[j], TakeCC(*table.column(j), indices, options, ctx));
  }
  return Table::Make(table.schema(), columns);
}

// Metafunction for dispatching to different Take implementations other than
// Array-Array.
//
// TODO: Revamp approach to executing Take operations. In addition to being
// overly complex dispatching, there is no parallelization.
class TakeMetaFunction : public MetaFunction {
 public:
  TakeMetaFunction() : MetaFunction("take", Arity::Binary()) {}

  Result<Datum> ExecuteImpl(const std::vector<Datum>& args,
                            const FunctionOptions* options,
                            ExecContext* ctx) const override {
    Datum::Kind index_kind = args[1].kind();
    const TakeOptions& take_opts = static_cast<const TakeOptions&>(*options);
    switch (args[0].kind()) {
      case Datum::ARRAY:
        if (index_kind == Datum::ARRAY) {
          return TakeAA(*args[0].make_array(), *args[1].make_array(), take_opts, ctx);
        } else if (index_kind == Datum::CHUNKED_ARRAY) {
          return TakeAC(*args[0].make_array(), *args[1].chunked_array(), take_opts, ctx);
        }
        break;
      case Datum::CHUNKED_ARRAY:
        if (index_kind == Datum::ARRAY) {
          return TakeCA(*args[0].chunked_array(), *args[1].make_array(), take_opts, ctx);
        } else if (index_kind == Datum::CHUNKED_ARRAY) {
          return TakeCC(*args[0].chunked_array(), *args[1].chunked_array(), take_opts,
                        ctx);
        }
        break;
      case Datum::RECORD_BATCH:
        if (index_kind == Datum::ARRAY) {
          return TakeRA(*args[0].record_batch(), *args[1].make_array(), take_opts, ctx);
        }
        break;
      case Datum::TABLE:
        if (index_kind == Datum::ARRAY) {
          return TakeTA(*args[0].table(), *args[1].make_array(), take_opts, ctx);
        } else if (index_kind == Datum::CHUNKED_ARRAY) {
          return TakeTC(*args[0].table(), *args[1].chunked_array(), take_opts, ctx);
        }
        break;
      default:
        break;
    }
    return Status::NotImplemented(
        "Unsupported types for take operation: "
        "values=",
        args[0].ToString(), "indices=", args[1].ToString());
  }
};

void RegisterVectorTake(FunctionRegistry* registry) {
  VectorKernel base;
  base.init = InitTake;
  base.can_execute_chunkwise = false;

  auto array_take = std::make_shared<VectorFunction>("array_take", Arity::Binary());

  InputType index_ty(match::Integer(), ValueDescr::ARRAY);

  // Single kernel entry point for all primitive types. We dispatch to take
  // implementations inside the kernel for now. The primitive take
  // implementation writes into preallocated memory while the other
  // implementations handle their own memory allocation.
  base.signature = KernelSignature::Make({InputType(match::Primitive(), ValueDescr::ARRAY),
        index_ty}, OutputType(FirstType));
  base.exec = exec;

  auto AddSimpleKernel = [&](InputType value_ty, ArrayKernelExec exec) {
    base.signature = KernelSignature::Make({value_ty, index_ty}, OutputType(FirstType));
    base.exec = exec;
    DCHECK_OK(array_take->AddKernel(base));
  };
  AddSimpleKernel(InputType::Array(null()), NullTakeExec);
  AddSimpleKernel(InputType::Array(Type::LIST), ListTakeExec);
  AddSimpleKernel(InputType::Array(Type::LARGE_LIST), LargeListTakeExec);
  AddSimpleKernel(InputType::Array(Type::FIXED_SIZE_LIST), FixedSizeListTakeExec);
  AddSimpleKernel(InputType::Array(Type::STRUCT), StructTakeExec);

  DCHECK_OK(registry->AddFunction(std::move(array_take)));

  // Add take metafunction
  DCHECK_OK(registry->AddFunction(std::make_shared<TakeMetaFunction>()));
}

}  // namespace internal
}  // namespace compute
}  // namespace arrow
