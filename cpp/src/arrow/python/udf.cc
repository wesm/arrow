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

#include "arrow/python/udf.h"
#include "arrow/compute/function.h"
#include "arrow/python/common.h"

namespace arrow {

using compute::ExecResult;
using compute::ExecSpan;

namespace py {

namespace {
Status CheckOutputType(const DataType& expected, const DataType& actual) {
  if (!expected.Equals(actual)) {
    return Status::TypeError("Expected output datatype ", expected.ToString(),
                             ", but function returned datatype ", actual.ToString());
  }
  return Status::OK();
}

struct PythonUdf {
  ScalarUdfWrapperCallback cb;
  std::shared_ptr<OwnedRefNoGIL> function;
  compute::OutputType output_type;

  // function needs to be destroyed at process exit
  // and Python may no longer be initialized.
  ~PythonUdf() {
    if (_Py_IsFinalizing()) {
      function->detach();
    }
  }

  Status operator()(compute::KernelContext* ctx, const ExecSpan& batch, ExecResult* out) {
    return SafeCallIntoPython([&]() -> Status { return Execute(ctx, batch, out); });
  }

  Status Execute(compute::KernelContext* ctx, const ExecSpan& batch, ExecResult* out) {
    const int num_args = batch.num_values();
    ScalarUdfContext udf_context{ctx->memory_pool(), batch.length};

    OwnedRef arg_tuple(PyTuple_New(num_args));
    RETURN_NOT_OK(CheckPyError());
    for (int arg_id = 0; arg_id < num_args; arg_id++) {
      if (batch[arg_id].is_scalar()) {
        std::shared_ptr<Scalar> c_data = batch[arg_id].scalar->GetSharedPtr();
        PyObject* data = wrap_scalar(c_data);
        PyTuple_SetItem(arg_tuple.obj(), arg_id, data);
      } else {
        std::shared_ptr<Array> c_data = batch[arg_id].array.ToArray();
        PyObject* data = wrap_array(c_data);
        PyTuple_SetItem(arg_tuple.obj(), arg_id, data);
        break;
      }
    }

    OwnedRef result(cb(function->obj(), udf_context, arg_tuple.obj()));
    RETURN_NOT_OK(CheckPyError());
    // unwrapping the output for expected output type
    if (is_scalar(result.obj())) {
      ARROW_ASSIGN_OR_RAISE(std::shared_ptr<Scalar> val, unwrap_scalar(result.obj()));
      RETURN_NOT_OK(CheckOutputType(*output_type.type(), *val->type));
      out->value = val;
      return Status::OK();
    } else if (is_array(result.obj())) {
      ARROW_ASSIGN_OR_RAISE(std::shared_ptr<Array> val, unwrap_array(result.obj()));
      RETURN_NOT_OK(CheckOutputType(*output_type.type(), *val->type()));
      out->value = std::move(val->data());
      return Status::OK();
    } else {
      return Status::TypeError("Unexpected output type: ", Py_TYPE(result.obj())->tp_name,
                               " (expected Scalar or Array)");
    }
    return Status::OK();
  }
};

}  // namespace

Status RegisterScalarFunction(PyObject* user_function, ScalarUdfWrapperCallback wrapper,
                              const ScalarUdfOptions& options) {
  if (!PyCallable_Check(user_function)) {
    return Status::TypeError("Expected a callable Python object.");
  }
  auto scalar_func = std::make_shared<compute::ScalarFunction>(
      options.func_name, options.arity, options.func_doc);
  Py_INCREF(user_function);
  std::vector<compute::InputType> input_types;
  for (const auto& in_dtype : options.input_types) {
    input_types.emplace_back(in_dtype);
  }
  compute::OutputType output_type(options.output_type);
  PythonUdf exec{wrapper, std::make_shared<OwnedRefNoGIL>(user_function), output_type};
  compute::ScalarKernel kernel(
      compute::KernelSignature::Make(std::move(input_types), std::move(output_type),
                                     options.arity.is_varargs),
      std::move(exec));
  kernel.mem_allocation = compute::MemAllocation::NO_PREALLOCATE;
  kernel.null_handling = compute::NullHandling::COMPUTED_NO_PREALLOCATE;
  RETURN_NOT_OK(scalar_func->AddKernel(std::move(kernel)));
  auto registry = compute::GetFunctionRegistry();
  RETURN_NOT_OK(registry->AddFunction(std::move(scalar_func)));
  return Status::OK();
}

}  // namespace py

}  // namespace arrow
