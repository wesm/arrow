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

#ifndef ARROW_COMPUTE_KERNELS_UTIL_INTERNAL_H
#define ARROW_COMPUTE_KERNELS_UTIL_INTERNAL_H

#include "arrow/type_fwd.h"

namespace arrow {
namespace compute {

template <typename T>
using is_number = std::is_base_of<Number, T>;

template <typename T>
using enable_if_primitive_ctype =
    typename std::enable_if<std::is_base_of<PrimitiveCType, T>::value>::type;

template <typename T>
using enable_if_number = typename std::enable_if<is_number<T>::value>::type;

template <typename T>
inline const T* GetValues(const ArrayData& data, int i) {
  return reinterpret_cast<const T*>(data.buffers[i]->data()) + data.offset;
}

template <typename T>
inline T* GetMutableValues(const ArrayData* data, int i) {
  return reinterpret_cast<T*>(data->buffers[i]->mutable_data()) + data->offset;
}

namespace {

inline void CopyData(const ArrayData& input, ArrayData* output) {
  output->length = input.length;
  output->null_count = input.null_count;
  output->buffers = input.buffers;
  output->offset = input.offset;
  output->child_data = input.child_data;
}

}  // namespace

}  // compute
}  // arrow

#endif  // ARROW_COMPUTE_KERNELS_UTIL_INTERNAL_H
