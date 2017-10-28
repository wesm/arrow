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

#include "arrow/status.h"
#include "arrow/util/visibility.h"

namespace arrow {

class Array;
class ChunkedArray;
class Column;
class DataType;

namespace compute {

class FunctionContext;
class UnaryKernel;

/// \since 0.8.0
/// \note API not yet finalized
ARROW_EXPORT
Status GetUniqueFunction(const DataType& in_type, std::unique_ptr<UnaryKernel>* kernel);

/// \brief Unique elements of an array
/// \param[in] context the FunctionContext
/// \param[in] array array with all possible values
/// \param[out] out resulting array
///
/// \since 0.8.0
/// \note API not yet finalized
ARROW_EXPORT
Status Unique(FunctionContext* context, const Array& array, std::shared_ptr<Array>* out);

/// \brief Unique elements of a chunked array
/// \param[in] context the FunctionContext
/// \param[in] array chunked array with all possible value
/// \param[out] out resulting array
///
/// \since 0.8.0
/// \note API not yet finalized
ARROW_EXPORT
Status Unique(FunctionContext* context, const ChunkedArray& array,
              std::shared_ptr<Array>* out);

/// \brief Unique elements of a column
/// \param[in] context the FunctionContext
/// \param[in] column column with all possible values
/// \param[out] out resulting array
///
/// \since 0.8.0
/// \note API not yet finalized
ARROW_EXPORT
Status Unique(FunctionContext* context, const Column& array, std::shared_ptr<Array>* out);

}  // compute
}  // arrow

#endif
