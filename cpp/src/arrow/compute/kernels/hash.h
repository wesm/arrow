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

#ifndef ARROW_COMPUTE_HASH_KERNELS_H
#define ARROW_COMPUTE_HASH_KERNELS_H

#include <memory>

#include "arrow/compute/kernel.h"
#include "arrow/status.h"
#include "arrow/util/visibility.h"

namespace arrow {

class Array;
class ArrayData;
class ChunkedArray;
class Column;
class DataType;

namespace compute {

class FunctionContext;

class ARROW_EXPORT HashKernel : public OpKernel {
 public:
  /// \brief Invoke hash table kernel on input array, returning any output
  /// values. Implementations should be thread-safe
  ///
  /// \param[in] ctx a function context
  /// \param[in] input the input array to process
  /// \param[out] out any output arrays (may not return any)
  /// \return Status
  virtual Status Call(FunctionContext* ctx, const Array& input,
                      std::vector<Datum>* out) = 0;

  virtual Status GetDictionary(std::shared_ptr<ArrayData>* out) = 0;
};

/// \since 0.8.0
/// \note API not yet finalized
ARROW_EXPORT
Status GetUniqueFunction(const DataType& in_type,
                         std::unique_ptr<HashKernel>* kernel);

ARROW_EXPORT
Status GetDictionaryEncodeFunction(const DataType& in_type,
                                   std::unique_ptr<HashKernel>* kernel);

/// \brief Compute unique elements from an array-like object
/// \param[in] context the FunctionContext
/// \param[in] datum array-like input
/// \param[out] out result as Array
///
/// \since 0.8.0
/// \note API not yet finalized
ARROW_EXPORT
Status Unique(FunctionContext* context, const Datum& datum,
              std::shared_ptr<Array>* out);


/// \brief Dictionary-encode values in an array-like object
/// \param[in] context the FunctionContext
/// \param[in] datum array-like input
/// \param[out] out result with same shape and type as input
///
/// \since 0.8.0
/// \note API not yet finalized
ARROW_EXPORT
Status DictionaryEncode(FunctionContext* context, const Datum& datum,
                        Datum* out);

ARROW_EXPORT
Status Match(FunctionContext* context, const Datum& values,
             const Array& member_set,
             std::shared_ptr<Array>* out);

ARROW_EXPORT
Status IsIn(FunctionContext* context, const Datum& values,
            const Array& member_set,
            std::shared_ptr<Array>* out);

ARROW_EXPORT
Status CountValues(FunctionContext* context, const Datum& values,
                   std::shared_ptr<Array>* out_uniques,
                   std::shared_ptr<Array>* out_counts);

}  // compute
}  // arrow

#endif
