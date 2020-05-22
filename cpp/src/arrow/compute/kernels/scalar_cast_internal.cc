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

#include <utility>

#include "arrow/compute/kernels/common.h"
#include "arrow/compute/kernels/scalar_cast_internal.h"

namespace arrow {
namespace compute {
namespace internal {

void CastFromExtension(KernelContext* ctx, const ExecBatch& batch, Datum* out) {
  const CastOptions& options = checked_cast<const CastState*>(ctx->state())->options;

  const DataType& in_type = *batch[0].type();
  const auto storage_type = checked_cast<const ExtensionType&>(in_type).storage_type();

  ExtensionArray extension(batch[0].array());

  Datum casted_storage;
  KERNEL_ABORT_IF_ERROR(
      ctx, Cast(*extension.storage(), out->type(), options, ctx->exec_context())
               .Value(&casted_storage));
  out->value = casted_storage.array();
}

Result<ValueDescr> ResolveOutputFromOptions(KernelContext* ctx,
                                            const std::vector<ValueDescr>& args) {
  const CastOptions& options = checked_cast<const CastState&>(*ctx->state()).options;
  return ValueDescr(options.to_type, args[0].shape);
}

void ZeroCopyCastExec(KernelContext* ctx, const ExecBatch& batch, Datum* out) {
  if (batch[0].kind() == Datum::ARRAY) {
    // Make a copy of the buffers into a destination array without carrying
    // the type
    const ArrayData& input = *batch[0].array();
    ArrayData* output = out->mutable_array();
    output->length = input.length;
    output->SetNullCount(input.null_count);
    output->buffers = input.buffers;
    output->offset = input.offset;
    output->child_data = input.child_data;
  } else {
    ctx->SetStatus(
        Status::NotImplemented("This cast not yet implemented for "
                               "scalar input"));
  }
}

void AddZeroCopyCast(InputType in_type, OutputType out_type, CastFunction* func) {
  auto sig = KernelSignature::Make({in_type}, out_type);
  ScalarKernel kernel;
  kernel.exec = ZeroCopyCastExec;
  kernel.signature = sig;
  kernel.null_handling = NullHandling::COMPUTED_NO_PREALLOCATE;
  kernel.mem_allocation = MemAllocation::NO_PREALLOCATE;
  DCHECK_OK(func->AddKernel(in_type.type_id(), std::move(kernel)));
}

}  // namespace internal
}  // namespace compute
}  // namespace arrow
