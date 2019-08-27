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

#include <memory>
#include <unordered_map>

#include "arrow/dataset/visibility.h"
#include "arrow/scalar.h"

namespace arrow {
namespace dataset {

/// placeholder until ARROW-6243 is resolved
/// represents a conjunction of several equality constraints
class ARROW_DS_EXPORT Expression {
 public:
  std::unordered_map<std::string, std::shared_ptr<Scalar>> values;
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

  const std::shared_ptr<Expression>& expression() const { return expression_; }

 private:
  std::shared_ptr<Expression> expression_;
};

}  // namespace dataset
}  // namespace arrow
