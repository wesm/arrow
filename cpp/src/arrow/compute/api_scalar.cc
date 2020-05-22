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

#include "arrow/compute/api_scalar.h"

#include <sstream>
#include <string>
#include <utility>

#include "arrow/compute/exec.h"

namespace arrow {
namespace compute {

#define SCALAR_EAGER_UNARY(NAME, REGISTRY_NAME)              \
  Result<Datum> NAME(const Datum& value, ExecContext* ctx) { \
    return CallFunction(ctx, REGISTRY_NAME, {value});        \
  }

#define SCALAR_EAGER_BINARY(NAME, REGISTRY_NAME)                                \
  Result<Datum> NAME(const Datum& left, const Datum& right, ExecContext* ctx) { \
    return CallFunction(ctx, REGISTRY_NAME, {left, right});                     \
  }

// ----------------------------------------------------------------------
// Arithmetic

SCALAR_EAGER_BINARY(Add, "add")

// ----------------------------------------------------------------------
// Set-related operations

static Result<Datum> ExecSetLookup(const std::string& func_name, const Datum& data,
                                   std::shared_ptr<Array> value_set,
                                   bool add_nulls_to_hash_table, ExecContext* ctx) {
  if (value_set->length() > 0 && !data.type()->Equals(value_set->type())) {
    std::stringstream ss;
    ss << "Array type didn't match type of values set: " << data.type()->ToString()
       << " vs " << value_set->type()->ToString();
    return Status::Invalid(ss.str());
  }
  SetLookupOptions options(std::move(value_set), !add_nulls_to_hash_table);
  return CallFunction(ctx, func_name, {data}, &options);
}

Result<Datum> IsIn(const Datum& values, std::shared_ptr<Array> value_set,
                   ExecContext* ctx) {
  return ExecSetLookup("isin", values, std::move(value_set),
                       /*add_nulls_to_hash_table=*/false, ctx);
}

Result<Datum> Match(const Datum& values, std::shared_ptr<Array> value_set,
                    ExecContext* ctx) {
  return ExecSetLookup("match", values, std::move(value_set),
                       /*add_nulls_to_hash_table=*/true, ctx);
}

// ----------------------------------------------------------------------
// Boolean functions

SCALAR_EAGER_UNARY(Invert, "invert")
SCALAR_EAGER_BINARY(And, "and")
SCALAR_EAGER_BINARY(KleeneAnd, "and_kleene")
SCALAR_EAGER_BINARY(Or, "or")
SCALAR_EAGER_BINARY(KleeneOr, "or_kleene")
SCALAR_EAGER_BINARY(Xor, "xor")

// ----------------------------------------------------------------------

Result<Datum> Compare(const Datum& left, const Datum& right, CompareOptions options,
                      ExecContext* ctx) {
  std::string func_name;
  switch (options.op) {
    case CompareOperator::EQUAL:
      func_name = "==";
      break;
    case CompareOperator::NOT_EQUAL:
      func_name = "!=";
      break;
    case CompareOperator::GREATER:
      func_name = ">";
      break;
    case CompareOperator::GREATER_EQUAL:
      func_name = ">=";
      break;
    case CompareOperator::LESS:
      func_name = "<";
      break;
    case CompareOperator::LESS_EQUAL:
      func_name = "<=";
      break;
    default:
      DCHECK(false);
      break;
  }
  return CallFunction(ctx, func_name, {left, right}, &options);
}

}  // namespace compute
}  // namespace arrow
