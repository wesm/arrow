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

#include "arrow/dataset/dataset.h"

#include <memory>
#include <utility>

#include "arrow/dataset/filter.h"
#include "arrow/dataset/scanner.h"
#include "arrow/util/stl.h"

namespace arrow {
namespace dataset {

SimpleDataFragment::SimpleDataFragment(
    std::vector<std::shared_ptr<RecordBatch>> record_batches)
    : record_batches_(std::move(record_batches)) {}

Status SimpleDataFragment::Scan(std::shared_ptr<ScanContext> scan_context,
                                std::unique_ptr<ScanTaskIterator>* out) {
  // Make an explicit copy of record_batches_ to ensure Scan can be called
  // multiple times.
  auto it = MakeVectorIterator(record_batches_);

  // RecordBatch -> ScanTask
  auto fn = [](std::shared_ptr<RecordBatch> batch) -> std::unique_ptr<ScanTask> {
    std::vector<std::shared_ptr<RecordBatch>> batches{batch};
    return internal::make_unique<SimpleScanTask>(std::move(batches));
  };

  *out = MakeMapIterator(fn, std::move(it));
  return Status::OK();
}

Status Dataset::Make(std::vector<std::shared_ptr<DataSource>> sources,
                     std::shared_ptr<Schema> schema, std::shared_ptr<Dataset>* out) {
  // TODO: Ensure schema and sources align.
  *out = std::make_shared<Dataset>(std::move(sources), std::move(schema));

  return Status::OK();
}

Status Dataset::NewScan(std::unique_ptr<ScannerBuilder>* out) {
  auto context = std::make_shared<ScanContext>();
  out->reset(new ScannerBuilder(this->shared_from_this(), context));
  return Status::OK();
}

std::unique_ptr<EmptyIterator<std::shared_ptr<DataFragment>>> DataSource::AssumeCondition(
    std::shared_ptr<ScanOptions>* options) const {
  DCHECK_NE(options, nullptr);
  if (*options == nullptr) {
    // null scan context; no selector to simplify
    return nullptr;
  }

  auto c = SelectorAssume((*options)->selector, partition_expression_);
  DCHECK_OK(c.status());
  auto expr = std::move(c).ValueOrDie();

  BooleanScalar trivial(true);
  if (expr->IsTrivialCondition(&trivial)) {
    if (trivial.is_valid && !trivial.value) {
      // don't yield any fragments
      return internal::make_unique<EmptyIterator<std::shared_ptr<DataFragment>>>();
    }
  }

  *options = std::make_shared<ScanOptions>(**options);
  (*options)->selector = ExpressionSelector(std::move(expr));
  return nullptr;
}

std::unique_ptr<DataFragmentIterator> SimpleDataSource::GetFragments(
    std::shared_ptr<ScanOptions> options) {
  if (auto empty = AssumeCondition(&options)) {
    return empty;
  }
  return MakeVectorIterator(fragments_);
}

}  // namespace dataset
}  // namespace arrow
