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

#include <cstdint>
#include <memory>
#include <unordered_set>
#include <vector>

#include "arrow/memory_pool.h"
#include "arrow/util/compression.h"
#include "arrow/util/visibility.h"

namespace arrow {

namespace util {

class Codec;

}  // namespace util

namespace ipc {

// ARROW-109: We set this number arbitrarily to help catch user mistakes. For
// deeply nested schemas, it is expected the user will indicate explicitly the
// maximum allowed recursion depth
constexpr int kMaxNestingDepth = 64;

// TODO: Should IpcOptions be renamed IpcWriteOptions?

/// \brief Options for writing Arrow IPC messages
struct ARROW_EXPORT IpcOptions {
  // If true, allow field lengths that don't fit in a signed 32-bit int.
  // Some implementations may not be able to parse such streams.
  bool allow_64bit = false;
  // The maximum permitted schema nesting depth.
  int max_recursion_depth = kMaxNestingDepth;

  // Write padding after memory buffers to this multiple of
  // bytes. Generally 8 or 64
  int32_t alignment = 8;

  /// \brief Write the pre-0.15.0 encapsulated IPC message format
  /// consisting of a 4-byte prefix instead of 8 byte
  bool write_legacy_ipc_format = false;

  /// \brief The memory pool to use for allocations made during IPC writing
  MemoryPool* memory_pool = default_memory_pool();

  /// \brief EXPERIMENTAL: Codec to use for compressing and decompressing
  /// record batch body buffers. This is not part of the Arrow IPC protocol and
  /// only for internal use (e.g. Feather files)
  Compression::type compression = Compression::UNCOMPRESSED;
  int compression_level = Compression::kUseDefaultCompressionLevel;

  static IpcOptions Defaults();
};

struct ARROW_EXPORT IpcReadOptions {
  // The maximum permitted schema nesting depth.
  int max_recursion_depth = kMaxNestingDepth;

  /// \brief The memory pool to use for allocations made during IPC writing
  MemoryPool* memory_pool = default_memory_pool();

  /// \brief EXPERIMENTAL: Top-level schema fields to include when
  /// deserializing RecordBatch. Null means to return all deserialized fields
  std::shared_ptr<std::unordered_set<int>> included_fields;

  static IpcReadOptions Defaults();
};

}  // namespace ipc
}  // namespace arrow
