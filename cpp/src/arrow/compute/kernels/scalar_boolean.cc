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

#include "arrow/compute/kernels/common.h"
#include "arrow/util/bit_util.h"

namespace arrow {
namespace compute {

namespace {

enum class ResolveNull { KLEENE_LOGIC, PROPAGATE };

enum BitmapIndex { LEFT_VALID, LEFT_DATA, RIGHT_VALID, RIGHT_DATA };

template <typename ComputeWord>
void ComputeKleene(ComputeWord&& compute_word, KernelContext* ctx, const ArrayData& left,
                   const ArrayData& right, ArrayData* out) {
  using internal::Bitmap;
  DCHECK(left.null_count != 0 || right.null_count != 0);

  Bitmap bitmaps[4];
  bitmaps[LEFT_VALID] = {left.buffers[0], left.offset, left.length};
  bitmaps[LEFT_DATA] = {left.buffers[1], left.offset, left.length};

  bitmaps[RIGHT_VALID] = {right.buffers[0], right.offset, right.length};
  bitmaps[RIGHT_DATA] = {right.buffers[1], right.offset, right.length};

  auto out_validity = out->GetMutableValues<uint64_t>(0);
  auto out_data = out->GetMutableValues<uint64_t>(1);

  int64_t i = 0;
  auto apply = [&](uint64_t left_valid, uint64_t left_data, uint64_t right_valid,
                   uint64_t right_data) {
    auto left_true = left_valid & left_data;
    auto left_false = left_valid & ~left_data;

    auto right_true = right_valid & right_data;
    auto right_false = right_valid & ~right_data;

    compute_word(left_true, left_false, right_true, right_false, &out_validity[i],
                 &out_data[i]);
    ++i;
  };

  if (right.null_count == 0 || left.null_count == 0) {
    if (left.null_count == 0) {
      // ensure only bitmaps[RIGHT_VALID].buffer might be null
      std::swap(bitmaps[LEFT_VALID], bitmaps[RIGHT_VALID]);
      std::swap(bitmaps[LEFT_DATA], bitmaps[RIGHT_DATA]);
    }
    // override bitmaps[RIGHT_VALID] to make it safe for Visit()
    bitmaps[RIGHT_VALID] = bitmaps[RIGHT_DATA];

    Bitmap::VisitWords(bitmaps, [&](std::array<uint64_t, 4> words) {
      apply(words[LEFT_VALID], words[LEFT_DATA], ~uint64_t(0), words[RIGHT_DATA]);
    });
  } else {
    Bitmap::VisitWords(bitmaps, [&](std::array<uint64_t, 4> words) {
      apply(words[LEFT_VALID], words[LEFT_DATA], words[RIGHT_VALID], words[RIGHT_DATA]);
    });
  }
}

struct Invert {
  static void Call(KernelContext* ctx, bool value) {
    ctx->SetStatus(Status::NotImplemented("NYI"));
  }

  static void Call(KernelContext* ctx, const ArrayData& in, ArrayData* out) {
    internal::InvertBitmap(in.buffers[1]->data(), in.offset, in.length,
                           out->buffers[1]->mutable_data(), out->offset);
  }
};

struct And {
  static void Call(KernelContext* ctx, const ArrayData& left, const ArrayData& right,
                   ArrayData* out) {
    internal::BitmapAnd(left.buffers[1]->data(), left.offset, right.buffers[1]->data(),
                        right.offset, right.length, out->offset,
                        out->buffers[1]->mutable_data());
  }
};

struct KleeneAnd {
  static void Call(KernelContext* ctx, const ArrayData& left, const ArrayData& right,
                   ArrayData* out) {
    if (left.GetNullCount() == 0 && right.GetNullCount() == 0) {
      return And::Call(ctx, left, right, out);
    }
    auto compute_word = [](uint64_t left_true, uint64_t left_false, uint64_t right_true,
                           uint64_t right_false, uint64_t* out_valid,
                           uint64_t* out_data) {
      *out_data = left_true & right_true;
      *out_valid = left_false | right_false | (left_true & right_true);
    };
    ComputeKleene(compute_word, ctx, left, right, out);
  }
};

struct Or {
  static void Call(KernelContext* ctx, const ArrayData& left, const ArrayData& right,
                   ArrayData* out) {
    internal::BitmapOr(left.buffers[1]->data(), left.offset, right.buffers[1]->data(),
                       right.offset, right.length, out->offset,
                       out->buffers[1]->mutable_data());
  }
};

struct KleeneOr {
  static void Call(KernelContext* ctx, const ArrayData& left, const ArrayData& right,
                   ArrayData* out) {
    if (left.GetNullCount() == 0 && right.GetNullCount() == 0) {
      return Or::Call(ctx, left, right, out);
    }
    static auto compute_word = [](uint64_t left_true, uint64_t left_false,
                                  uint64_t right_true, uint64_t right_false,
                                  uint64_t* out_valid, uint64_t* out_data) {
      *out_data = left_true | right_true;
      *out_valid = left_true | right_true | (left_false & right_false);
    };

    return ComputeKleene(compute_word, ctx, left, right, out);
  }
};

struct Xor {
  static void Call(KernelContext* ctx, const ArrayData& left, const ArrayData& right,
                   ArrayData* out) {
    internal::BitmapXor(left.buffers[1]->data(), left.offset, right.buffers[1]->data(),
                        right.offset, right.length, out->offset,
                        out->buffers[1]->mutable_data());
  }
};

void MakeFunction(std::string name, int arity, ArrayKernelExec exec,
                  FunctionRegistry* registry, bool can_write_into_slices = true) {
  auto func = std::make_shared<ScalarFunction>(name, arity);

  // Scalar arguments not yet supported
  std::vector<InputType> in_types(arity, InputType::Array(boolean()));
  ScalarKernel kernel(std::move(in_types), boolean(), exec);
  kernel.can_write_into_slices = can_write_into_slices;

  DCHECK_OK(func->AddKernel(kernel));
  DCHECK_OK(registry->AddFunction(std::move(func)));
}

}  // namespace

namespace internal {

using codegen::SimpleExec;

void RegisterScalarBoolean(FunctionRegistry* registry) {
  // These functions can write into sliced output bitmaps
  MakeFunction("invert", 1, SimpleExec::Unary<Invert>, registry);
  MakeFunction("and", 2, SimpleExec::Binary<And>, registry);
  MakeFunction("or", 2, SimpleExec::Binary<Or>, registry);
  MakeFunction("xor", 2, SimpleExec::Binary<Xor>, registry);

  // The Kleene logic kernels cannot write into sliced output bitmaps
  MakeFunction("and_kleene", 2, SimpleExec::Binary<KleeneAnd>, registry,
               /*can_write_into_slices=*/false);
  MakeFunction("or_kleene", 2, SimpleExec::Binary<KleeneOr>, registry,
               /*can_write_into_slices=*/false);
}

}  // namespace internal
}  // namespace compute
}  // namespace arrow
