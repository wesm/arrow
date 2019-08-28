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

#pragma once

#include <algorithm>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "arrow/compute/kernel.h"
#include "arrow/dataset/dataset.h"
#include "arrow/dataset/visibility.h"
#include "arrow/result.h"
#include "arrow/scalar.h"

namespace arrow {
namespace dataset {

/// placeholder until ARROW-6243 is resolved
/// Represents a conjunction of several equality constraints if trivial == NULLPTR.
/// If trivial != NULLPTR, represents a comparison which always evaluates to *trivial.
class ARROW_DS_EXPORT Expression {
 public:
  Result<std::shared_ptr<Expression>> Assume(const Expression& given) const {
    if (trivial != NULLPTR) {
      // no further simplification is possible
      return Copy();
    }

    if (given.trivial != NULLPTR) {
      return Status::NotImplemented("simplification given trivial expression");
    }

    auto out = std::make_shared<Expression>();

    using Constraint = std::pair<const std::string, std::shared_ptr<Scalar>>;
    std::set_difference(values.begin(), values.end(), given.values.begin(),
                        given.values.end(), std::inserter(out->values, out->values.end()),
                        [&out](const Constraint& c, const Constraint& given) {
                          if (c.first != given.first) {
                            return c.first < given.first;
                          }
                          if (!c.second->Equals(*given.second)) {
                            // the given expression indicates this field will always equal
                            // something else, so this expression will always evaluate to
                            // false
                            out->trivial = std::make_shared<BooleanScalar>(false);
                          }
                          // drop constraints from this expression which are guaranteed by
                          // given
                          return false;
                        });

    if (out->trivial != NULLPTR) {
      out->values.clear();
    }
    return std::move(out);
  }

  Result<compute::Datum> Evaluate(compute::FunctionContext* ctx,
                                  const RecordBatch& batch) const {
    return Status::NotImplemented("evaluation of expressions against record batches");
  }

  bool IsTrivialCondition(BooleanScalar* c = NULLPTR) const {
    if (values.size() != 0) {
      return false;
    }

    if (c != NULLPTR) {
      *c = trivial == NULLPTR ? BooleanScalar(true) : *trivial;
    }
    return true;
  }

  std::shared_ptr<Expression> Copy() const { return std::make_shared<Expression>(*this); }

  std::map<std::string, std::shared_ptr<Scalar>> values;
  std::shared_ptr<BooleanScalar> trivial;
};

class ARROW_DS_EXPORT Filter {
 public:
  enum type {
    /// Simple boolean predicate consisting of comparisons and boolean
    /// logic (AND, OR, NOT) involving Schema fields
    EXPRESSION,

    ///
    GENERIC
  };

  explicit Filter(std::shared_ptr<Expression> e) : expression_(std::move(e)) {}

  const std::shared_ptr<Expression>& expression() const { return expression_; }

 private:
  std::shared_ptr<Expression> expression_;
};

inline Result<std::shared_ptr<Expression>> SelectorAssume(
    const std::shared_ptr<DataSelector>& selector,
    const std::shared_ptr<Expression>& given) {
  if (selector == nullptr) {
    return std::make_shared<Expression>();
  }

  Expression out_expr;
  for (const auto& f : selector->filters) {
    for (const auto& name_value : f->expression()->values) {
      if (!out_expr.values.insert(name_value).second) {
        return Status::Invalid("filter expressions were not disjoint");
      }
    }
  }

  if (given == nullptr) {
    return out_expr.Copy();
  }
  return out_expr.Assume(*given);
}

inline std::shared_ptr<DataSelector> ExpressionSelector(std::shared_ptr<Expression> e) {
  return std::make_shared<DataSelector>(
      DataSelector{FilterVector{std::make_shared<Filter>(std::move(e))}});
}

}  // namespace dataset
}  // namespace arrow
