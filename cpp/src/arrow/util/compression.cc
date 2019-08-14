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

#include "arrow/util/compression.h"

#include <limits>
#include <memory>

#ifdef ARROW_WITH_BROTLI
#include "arrow/util/compression_brotli.h"
#endif

#ifdef ARROW_WITH_SNAPPY
#include "arrow/util/compression_snappy.h"
#endif

#ifdef ARROW_WITH_LZ4
#include "arrow/util/compression_lz4.h"
#endif

#ifdef ARROW_WITH_ZLIB
#include "arrow/util/compression_zlib.h"
#endif

#ifdef ARROW_WITH_ZSTD
#include "arrow/util/compression_zstd.h"
#endif

#ifdef ARROW_WITH_BZ2
#include "arrow/util/compression_bz2.h"
#endif

#include "arrow/status.h"

namespace arrow {
namespace util {

int GetHintValueForDefaultCompressionLevel() { return std::numeric_limits<int>::min(); }

Compressor::~Compressor() {}

Decompressor::~Decompressor() {}

Codec::~Codec() {}

Status Codec::Create(Compression::type codec_type, std::unique_ptr<Codec>* result) {
  const int compression_level = GetHintValueForDefaultCompressionLevel();
  return Codec::Create(codec_type, compression_level, result);
}

Status Codec::Create(Compression::type codec_type, int compression_level,
                     std::unique_ptr<Codec>* result) {
  const bool use_default_compression_level =
      (compression_level == GetHintValueForDefaultCompressionLevel());
  Codec* codec = nullptr;
  switch (codec_type) {
    case Compression::UNCOMPRESSED:
      break;
    case Compression::SNAPPY:
#ifdef ARROW_WITH_SNAPPY
      codec = new SnappyCodec();
      break;
#else
      return Status::NotImplemented("Snappy codec support not built");
#endif
    case Compression::GZIP:
#ifdef ARROW_WITH_ZLIB
      if (use_default_compression_level) {
        codec = new GZipCodec();
      } else {
        codec = new GZipCodec(compression_level);
      }
      break;
#else
      return Status::NotImplemented("Gzip codec support not built");
#endif
    case Compression::LZO:
      return Status::NotImplemented("LZO codec not implemented");
    case Compression::BROTLI:
#ifdef ARROW_WITH_BROTLI
      if (use_default_compression_level) {
        codec = new BrotliCodec();
      } else {
        codec = new BrotliCodec(compression_level);
      }
      break;
#else
      return Status::NotImplemented("Brotli codec support not built");
#endif
    case Compression::LZ4:
#ifdef ARROW_WITH_LZ4
      codec = new Lz4Codec();
      break;
#else
      return Status::NotImplemented("LZ4 codec support not built");
#endif
    case Compression::ZSTD:
#ifdef ARROW_WITH_ZSTD
      if (use_default_compression_level) {
        codec = new ZSTDCodec();
      } else {
        codec = new ZSTDCodec(compression_level);
      }
      break;
#else
      return Status::NotImplemented("ZSTD codec support not built");
#endif
    case Compression::BZ2:
#ifdef ARROW_WITH_BZ2
      if (use_default_compression_level) {
        codec = new BZ2Codec();
      } else {
        codec = new BZ2Codec(compression_level);
      }
      break;
#else
      return Status::NotImplemented("BZ2 codec support not built");
#endif
    default:
      return Status::Invalid("Unrecognized codec");
  }
  result->reset(codec);
  return Status::OK();
}

}  // namespace util
}  // namespace arrow
