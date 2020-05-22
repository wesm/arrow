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

#include "arrow/compute/registry.h"

#include <algorithm>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "arrow/compute/function.h"
#include "arrow/compute/registry_internal.h"
#include "arrow/status.h"
#include "arrow/util/logging.h"

namespace arrow {
namespace compute {

class FunctionRegistry::FunctionRegistryImpl {
 public:
  Status AddFunction(std::shared_ptr<const Function> function, bool allow_overwrite) {
    std::lock_guard<std::mutex> mutation_guard(lock_);

    const std::string& name = function->name();
    auto it = name_to_function_.find(name);
    if (it != name_to_function_.end() && !allow_overwrite) {
      return Status::KeyError("Already have a function registered with name: ", name);
    }
    name_to_function_[name] = std::move(function);
    return Status::OK();
  }

  Result<std::shared_ptr<const Function>> GetFunction(const std::string& name) const {
    auto it = name_to_function_.find(name);
    if (it == name_to_function_.end()) {
      return Status::KeyError("No function registered with name: ", name);
    }
    return it->second;
  }

  std::vector<std::string> GetFunctionNames() const {
    std::vector<std::string> results;
    for (auto it : name_to_function_) {
      results.push_back(it.first);
    }
    std::sort(results.begin(), results.end());
    return results;
  }

  int num_functions() const { return static_cast<int>(name_to_function_.size()); }

 private:
  std::mutex lock_;
  std::unordered_map<std::string, std::shared_ptr<const Function>> name_to_function_;
};

std::unique_ptr<FunctionRegistry> FunctionRegistry::Make() {
  return std::unique_ptr<FunctionRegistry>(new FunctionRegistry());
}

FunctionRegistry::FunctionRegistry() { impl_.reset(new FunctionRegistryImpl()); }

FunctionRegistry::~FunctionRegistry() {}

Status FunctionRegistry::AddFunction(std::shared_ptr<const Function> function,
                                     bool allow_overwrite) {
  return impl_->AddFunction(std::move(function), allow_overwrite);
}

Result<std::shared_ptr<const Function>> FunctionRegistry::GetFunction(
    const std::string& name) const {
  return impl_->GetFunction(name);
}

std::vector<std::string> FunctionRegistry::GetFunctionNames() const {
  return impl_->GetFunctionNames();
}

int FunctionRegistry::num_functions() const { return impl_->num_functions(); }

static std::unique_ptr<FunctionRegistry> g_registry;
static std::once_flag func_registry_initialized;

namespace internal {

static void CreateBuiltInRegistry() {
  g_registry = FunctionRegistry::Make();

  // Scalar functions
  RegisterScalarArithmetic(g_registry.get());
  RegisterScalarBoolean(g_registry.get());
  RegisterScalarComparison(g_registry.get());
  RegisterScalarSetLookup(g_registry.get());
  RegisterScalarCasts(g_registry.get());

  // Aggregate functions
  RegisterScalarAggregateBasic(g_registry.get());

  // Vector functions
  RegisterVectorFilter(g_registry.get());
  RegisterVectorHash(g_registry.get());
  RegisterVectorSort(g_registry.get());
  RegisterVectorTake(g_registry.get());
}

}  // namespace internal

FunctionRegistry* GetFunctionRegistry() {
  std::call_once(func_registry_initialized, internal::CreateBuiltInRegistry);
  return g_registry.get();
}

}  // namespace compute
}  // namespace arrow
