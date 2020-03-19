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

#include "arrow/ipc/reader.h"

#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include <flatbuffers/flatbuffers.h>  // IWYU pragma: export

#include "arrow/array.h"
#include "arrow/buffer.h"
#include "arrow/io/interfaces.h"
#include "arrow/io/memory.h"
#include "arrow/ipc/dictionary.h"
#include "arrow/ipc/message.h"
#include "arrow/ipc/metadata_internal.h"
#include "arrow/record_batch.h"
#include "arrow/sparse_tensor.h"
#include "arrow/status.h"
#include "arrow/tensor.h"
#include "arrow/type.h"
#include "arrow/type_traits.h"
#include "arrow/util/key_value_metadata.h"
#include "arrow/util/logging.h"
#include "arrow/util/ubsan.h"
#include "arrow/visitor_inline.h"

#include "generated/File_generated.h"  // IWYU pragma: export
#include "generated/Message_generated.h"
#include "generated/Schema_generated.h"

namespace arrow {

namespace flatbuf = org::apache::arrow::flatbuf;

using internal::checked_cast;
using internal::checked_pointer_cast;

namespace ipc {

using internal::FileBlock;
using internal::kArrowMagicBytes;

Status InvalidMessageType(Message::Type expected, Message::Type actual) {
  return Status::IOError("Expected IPC message of type ", FormatMessageType(expected),
                         " but got ", FormatMessageType(actual));
}

#define CHECK_MESSAGE_TYPE(expected, actual)           \
  do {                                                 \
    if ((actual) != (expected)) {                      \
      return InvalidMessageType((expected), (actual)); \
    }                                                  \
  } while (0)

#define CHECK_HAS_BODY(message)                                       \
  do {                                                                \
    if ((message).body() == nullptr) {                                \
      return Status::IOError("Expected body in IPC message of type ", \
                             FormatMessageType((message).type()));    \
    }                                                                 \
  } while (0)

#define CHECK_HAS_NO_BODY(message)                                      \
  do {                                                                  \
    if ((message).body_length() != 0) {                                 \
      return Status::IOError("Unexpected body in IPC message of type ", \
                             FormatMessageType((message).type()));      \
    }                                                                   \
  } while (0)

// ----------------------------------------------------------------------
// Record batch read path

/// The field_index and buffer_index are incremented based on how much of the
/// batch is "consumed" (through nested data reconstruction, for example)
class ArrayLoader {
 public:
  explicit ArrayLoader(const flatbuf::RecordBatch* metadata, io::RandomAccessFile* file,
                       const DictionaryMemo* dictionary_memo, const IpcOptions& options)
      : metadata_(metadata),
        file_(file),
        dictionary_memo_(dictionary_memo),
        options_(options),
        max_recursion_depth_(options.max_recursion_depth),
        buffer_index_(0),
        field_index_(0),
        skip_io_(false) {}

  Status ReadBuffer(int64_t offset, int64_t length, std::shared_ptr<Buffer>* out) {
    if (skip_io_) {
      return Status::OK();
    }

    // This construct permits overriding GetBuffer at compile time
    if (!BitUtil::IsMultipleOf8(offset)) {
      return Status::Invalid("Buffer ", buffer_index_,
                             " did not start on 8-byte aligned offset: ", offset);
    }
    return file_->ReadAt(offset, length).Value(out);
  }

  // Use this to disable calls to RandomAccessFile::ReadAt, for field skipping
  void SkipIO(bool skip_io = true) { skip_io_ = skip_io; }

  Status LoadType(const DataType& type) { return VisitTypeInline(type, this); }

  Status DecompressBuffers() {
    // If the buffers are indicated to be compressed, instantiate the codec and
    // decompress them
    std::unique_ptr<util::Codec> codec;
    ARROW_ASSIGN_OR_RAISE(codec, util::Codec::Create(options_.compression));

    // TODO: Parallelize decompression
    for (size_t i = 0; i < out_->buffers.size(); ++i) {
      if (out_->buffers[i] == nullptr) {
        continue;
      }
      if (out_->buffers[i]->size() > 0) {
        const uint8_t* data = out_->buffers[i]->data();
        int64_t compressed_size = out_->buffers[i]->size() - sizeof(int64_t);
        int64_t uncompressed_size = util::SafeLoadAs<int64_t>(data);

        std::shared_ptr<Buffer> uncompressed;
        RETURN_NOT_OK(
            AllocateBuffer(options_.memory_pool, uncompressed_size, &uncompressed));

        int64_t actual_decompressed;
        ARROW_ASSIGN_OR_RAISE(
            actual_decompressed,
            codec->Decompress(compressed_size, data + sizeof(int64_t), uncompressed_size,
                              uncompressed->mutable_data()));
        if (actual_decompressed != uncompressed_size) {
          return Status::Invalid("Failed to fully decompress buffer, expected ",
                                 uncompressed_size, " bytes but decompressed ",
                                 actual_decompressed);
        }
        out_->buffers[i] = uncompressed;
      }
    }
    return Status::OK();
  }

  Status Load(const Field* field, ArrayData* out) {
    if (max_recursion_depth_ <= 0) {
      return Status::Invalid("Max recursion depth reached");
    }

    field_ = field;
    out_ = out;
    out_->type = field_->type();
    RETURN_NOT_OK(LoadType(*field_->type()));

    if (options_.compression != Compression::UNCOMPRESSED) {
      RETURN_NOT_OK(DecompressBuffers());
    }
    return Status::OK();
  }

  Status GetBuffer(int buffer_index, std::shared_ptr<Buffer>* out) {
    auto buffers = metadata_->buffers();
    CHECK_FLATBUFFERS_NOT_NULL(buffers, "RecordBatch.buffers");
    if (buffer_index >= static_cast<int>(buffers->size())) {
      return Status::IOError("buffer_index out of range.");
    }
    const flatbuf::Buffer* buffer = buffers->Get(buffer_index);
    if (buffer->length() == 0) {
      // Should never return a null buffer here.
      // (zero-sized buffer allocations are cheap)
      return AllocateBuffer(0, out);
    } else {
      return ReadBuffer(buffer->offset(), buffer->length(), out);
    }
  }

  Status GetFieldMetadata(int field_index, ArrayData* out) {
    auto nodes = metadata_->nodes();
    CHECK_FLATBUFFERS_NOT_NULL(nodes, "Table.nodes");
    // pop off a field
    if (field_index >= static_cast<int>(nodes->size())) {
      return Status::Invalid("Ran out of field metadata, likely malformed");
    }
    const flatbuf::FieldNode* node = nodes->Get(field_index);

    out->length = node->length();
    out->null_count = node->null_count();
    out->offset = 0;
    return Status::OK();
  }

  Status LoadCommon() {
    // This only contains the length and null count, which we need to figure
    // out what to do with the buffers. For example, if null_count == 0, then
    // we can skip that buffer without reading from shared memory
    RETURN_NOT_OK(GetFieldMetadata(field_index_++, out_));

    // extract null_bitmap which is common to all arrays
    if (out_->null_count == 0) {
      out_->buffers[0] = nullptr;
    } else {
      RETURN_NOT_OK(GetBuffer(buffer_index_, &out_->buffers[0]));
    }
    buffer_index_++;
    return Status::OK();
  }

  template <typename TYPE>
  Status LoadPrimitive() {
    out_->buffers.resize(2);

    RETURN_NOT_OK(LoadCommon());
    if (out_->length > 0) {
      RETURN_NOT_OK(GetBuffer(buffer_index_++, &out_->buffers[1]));
    } else {
      buffer_index_++;
      out_->buffers[1].reset(new Buffer(nullptr, 0));
    }
    return Status::OK();
  }

  template <typename TYPE>
  Status LoadBinary() {
    out_->buffers.resize(3);

    RETURN_NOT_OK(LoadCommon());
    RETURN_NOT_OK(GetBuffer(buffer_index_++, &out_->buffers[1]));
    return GetBuffer(buffer_index_++, &out_->buffers[2]);
  }

  template <typename TYPE>
  Status LoadList(const TYPE& type) {
    out_->buffers.resize(2);

    RETURN_NOT_OK(LoadCommon());
    RETURN_NOT_OK(GetBuffer(buffer_index_++, &out_->buffers[1]));

    const int num_children = type.num_children();
    if (num_children != 1) {
      return Status::Invalid("Wrong number of children: ", num_children);
    }

    return LoadChildren(type.children());
  }

  Status LoadChildren(std::vector<std::shared_ptr<Field>> child_fields) {
    ArrayData* parent = out_;
    parent->child_data.reserve(static_cast<int>(child_fields.size()));
    for (const auto& child_field : child_fields) {
      auto field_array = std::make_shared<ArrayData>();
      --max_recursion_depth_;
      RETURN_NOT_OK(Load(child_field.get(), field_array.get()));
      ++max_recursion_depth_;
      parent->child_data.emplace_back(field_array);
    }
    return Status::OK();
  }

  Status Visit(const NullType& type) {
    out_->buffers.resize(1);

    // ARROW-6379: NullType has no buffers in the IPC payload
    return GetFieldMetadata(field_index_++, out_);
  }

  template <typename T>
  enable_if_t<std::is_base_of<FixedWidthType, T>::value &&
                  !std::is_base_of<FixedSizeBinaryType, T>::value &&
                  !std::is_base_of<DictionaryType, T>::value,
              Status>
  Visit(const T& type) {
    return LoadPrimitive<T>();
  }

  template <typename T>
  enable_if_base_binary<T, Status> Visit(const T& type) {
    return LoadBinary<T>();
  }

  Status Visit(const FixedSizeBinaryType& type) {
    out_->buffers.resize(2);
    RETURN_NOT_OK(LoadCommon());
    return GetBuffer(buffer_index_++, &out_->buffers[1]);
  }

  template <typename T>
  enable_if_var_size_list<T, Status> Visit(const T& type) {
    return LoadList(type);
  }

  Status Visit(const MapType& type) {
    RETURN_NOT_OK(LoadList(type));
    return MapArray::ValidateChildData(out_->child_data);
  }

  Status Visit(const FixedSizeListType& type) {
    out_->buffers.resize(1);

    RETURN_NOT_OK(LoadCommon());

    const int num_children = type.num_children();
    if (num_children != 1) {
      return Status::Invalid("Wrong number of children: ", num_children);
    }

    return LoadChildren(type.children());
  }

  Status Visit(const StructType& type) {
    out_->buffers.resize(1);
    RETURN_NOT_OK(LoadCommon());
    return LoadChildren(type.children());
  }

  Status Visit(const UnionType& type) {
    out_->buffers.resize(3);

    RETURN_NOT_OK(LoadCommon());
    if (out_->length > 0) {
      RETURN_NOT_OK(GetBuffer(buffer_index_, &out_->buffers[1]));
      if (type.mode() == UnionMode::DENSE) {
        RETURN_NOT_OK(GetBuffer(buffer_index_ + 1, &out_->buffers[2]));
      }
    }
    buffer_index_ += type.mode() == UnionMode::DENSE ? 2 : 1;
    return LoadChildren(type.children());
  }

  Status Visit(const DictionaryType& type) {
    RETURN_NOT_OK(LoadType(*type.index_type()));

    // Look up dictionary
    int64_t id = -1;
    RETURN_NOT_OK(dictionary_memo_->GetId(field_, &id));
    RETURN_NOT_OK(dictionary_memo_->GetDictionary(id, &out_->dictionary));

    return Status::OK();
  }

  Status Visit(const ExtensionType& type) { return LoadType(*type.storage_type()); }

 private:
  const flatbuf::RecordBatch* metadata_;
  io::RandomAccessFile* file_;
  const DictionaryMemo* dictionary_memo_;
  const IpcOptions& options_;
  int max_recursion_depth_;
  int buffer_index_;
  int field_index_;
  bool skip_io_;

  const Field* field_;
  ArrayData* out_;
};

Status ReadRecordBatch(const flatbuf::RecordBatch* metadata,
                       const std::shared_ptr<Schema>& schema,
                       const DictionaryMemo* dictionary_memo, const IpcOptions& options,
                       io::RandomAccessFile* file, std::shared_ptr<RecordBatch>* out) {
  ArrayLoader loader(metadata, file, dictionary_memo, options);
  std::vector<std::shared_ptr<ArrayData>> arrays(schema->num_fields());
  for (int i = 0; i < schema->num_fields(); ++i) {
    auto arr = std::make_shared<ArrayData>();
    RETURN_NOT_OK(loader.Load(schema->field(i).get(), arr.get()));
    if (metadata->length() != arr->length) {
      return Status::IOError("Array length did not match record batch length");
    }
    arrays[i] = std::move(arr);
  }
  *out = RecordBatch::Make(schema, metadata->length(), std::move(arrays));
  return Status::OK();
}

Status ReadRecordBatch(const flatbuf::RecordBatch* metadata,
                       const std::shared_ptr<Schema>& schema,
                       const std::unordered_set<int>& selected_fields,
                       const DictionaryMemo* dictionary_memo, const IpcOptions& options,
                       io::RandomAccessFile* file,
                       std::vector<std::shared_ptr<ArrayData>>* fields) {
  ArrayLoader loader(metadata, file, dictionary_memo, options);

  fields->resize(selected_fields.size());

  // The index of the nxt non-skipped field being read
  int current_read_index = 0;

  ArrayData dummy_for_skipped_fields;
  for (int i = 0; i < schema->num_fields(); ++i) {
    if (selected_fields.find(i) != selected_fields.end()) {
      // Read field
      auto arr = std::make_shared<ArrayData>();
      loader.SkipIO(false);
      RETURN_NOT_OK(loader.Load(schema->field(i).get(), arr.get()));
      if (metadata->length() != arr->length) {
        return Status::IOError("Array length did not match record batch length");
      }
      (*fields)[current_read_index++] = std::move(arr);
    } else {
      // Skip field. We run the loading logic so the proper number of fields
      // and buffers are skipped before moving onto the next field
      loader.SkipIO();
      RETURN_NOT_OK(loader.Load(schema->field(i).get(), &dummy_for_skipped_fields));
    }
  }
  return Status::OK();
}

Status ReadRecordBatch(const Buffer& metadata, const std::shared_ptr<Schema>& schema,
                       const DictionaryMemo* dictionary_memo, io::RandomAccessFile* file,
                       std::shared_ptr<RecordBatch>* out) {
  return ReadRecordBatch(metadata, schema, dictionary_memo, IpcOptions::Defaults(), file,
                         out);
}

Status ReadRecordBatch(const Message& message, const std::shared_ptr<Schema>& schema,
                       const DictionaryMemo* dictionary_memo,
                       std::shared_ptr<RecordBatch>* out) {
  CHECK_MESSAGE_TYPE(Message::RECORD_BATCH, message.type());
  CHECK_HAS_BODY(message);
  ARROW_ASSIGN_OR_RAISE(auto reader, Buffer::GetReader(message.body()));
  return ReadRecordBatch(*message.metadata(), schema, dictionary_memo,
                         IpcOptions::Defaults(), reader.get(), out);
}

// ----------------------------------------------------------------------
// Array loading

Status SetCompression(const flatbuf::Message* message, IpcOptions* out) {
  if (message->custom_metadata() != nullptr) {
    // TODO: Ensure this deserialization only ever happens once
    std::shared_ptr<const KeyValueMetadata> metadata;
    RETURN_NOT_OK(internal::GetKeyValueMetadata(message->custom_metadata(), &metadata));
    int index = metadata->FindKey("ARROW:body_compression");
    if (index != -1) {
      ARROW_ASSIGN_OR_RAISE(out->compression,
                            util::Codec::GetCompressionType(metadata->value(index)));
    }
  }
  return Status::OK();
}

Status ReadRecordBatch(const Buffer& metadata, const std::shared_ptr<Schema>& schema,
                       const DictionaryMemo* dictionary_memo, IpcOptions options,
                       io::RandomAccessFile* file, std::shared_ptr<RecordBatch>* out) {
  const flatbuf::Message* message;
  RETURN_NOT_OK(internal::VerifyMessage(metadata.data(), metadata.size(), &message));
  auto batch = message->header_as_RecordBatch();
  if (batch == nullptr) {
    return Status::IOError(
        "Header-type of flatbuffer-encoded Message is not RecordBatch.");
  }
  RETURN_NOT_OK(SetCompression(message, &options));
  return ReadRecordBatch(batch, schema, dictionary_memo, options, file, out);
}

Status ReadDictionary(const Buffer& metadata, DictionaryMemo* dictionary_memo,
                      io::RandomAccessFile* file) {
  IpcOptions options = IpcOptions::Defaults();

  const flatbuf::Message* message;
  RETURN_NOT_OK(internal::VerifyMessage(metadata.data(), metadata.size(), &message));
  auto dictionary_batch = message->header_as_DictionaryBatch();
  if (dictionary_batch == nullptr) {
    return Status::IOError(
        "Header-type of flatbuffer-encoded Message is not DictionaryBatch.");
  }

  RETURN_NOT_OK(SetCompression(message, &options));

  int64_t id = dictionary_batch->id();

  // Look up the field, which must have been added to the
  // DictionaryMemo already prior to invoking this function
  std::shared_ptr<DataType> value_type;
  RETURN_NOT_OK(dictionary_memo->GetDictionaryType(id, &value_type));

  auto value_field = ::arrow::field("dummy", value_type);

  // The dictionary is embedded in a record batch with a single column
  std::shared_ptr<RecordBatch> batch;
  auto batch_meta = dictionary_batch->data();
  CHECK_FLATBUFFERS_NOT_NULL(batch_meta, "DictionaryBatch.data");
  RETURN_NOT_OK(ReadRecordBatch(batch_meta, ::arrow::schema({value_field}),
                                dictionary_memo, options, file, &batch));
  if (batch->num_columns() != 1) {
    return Status::Invalid("Dictionary record batch must only contain one field");
  }
  auto dictionary = batch->column(0);
  return dictionary_memo->AddDictionary(id, dictionary);
}

// ----------------------------------------------------------------------
// RecordBatchStreamReader implementation

class RecordBatchStreamReaderImpl : public RecordBatchReader {
 public:
  Status Open(std::unique_ptr<MessageReader> message_reader) {
    message_reader_ = std::move(message_reader);
    return ReadSchema();
  }

  Status ReadNext(std::shared_ptr<RecordBatch>* batch) override {
    if (!read_initial_dictionaries_) {
      RETURN_NOT_OK(ReadInitialDictionaries());
    }

    if (empty_stream_) {
      // ARROW-6006: Degenerate case where stream contains no data, we do not
      // bother trying to read a RecordBatch message from the stream
      *batch = nullptr;
      return Status::OK();
    }

    std::unique_ptr<Message> message;
    RETURN_NOT_OK(message_reader_->ReadNextMessage(&message));
    if (message == nullptr) {
      // End of stream
      *batch = nullptr;
      return Status::OK();
    }

    if (message->type() == Message::DICTIONARY_BATCH) {
      // TODO(wesm): implement delta dictionaries
      return Status::NotImplemented("Delta dictionaries not yet implemented");
    } else {
      CHECK_HAS_BODY(*message);
      ARROW_ASSIGN_OR_RAISE(auto reader, Buffer::GetReader(message->body()));
      return ReadRecordBatch(*message->metadata(), schema_, &dictionary_memo_,
                             reader.get(), batch);
    }
  }

  std::shared_ptr<Schema> schema() const override { return schema_; }

 private:
  Status ReadSchema() {
    std::unique_ptr<Message> message;
    RETURN_NOT_OK(message_reader_->ReadNextMessage(&message));
    if (!message) {
      return Status::Invalid("Tried reading schema message, was null or length 0");
    }
    CHECK_MESSAGE_TYPE(Message::SCHEMA, message->type());
    CHECK_HAS_NO_BODY(*message);
    return internal::GetSchema(message->header(), &dictionary_memo_, &schema_);
  }

  Status ParseDictionary(const Message& message) {
    // Only invoke this method if we already know we have a dictionary message
    DCHECK_EQ(message.type(), Message::DICTIONARY_BATCH);
    CHECK_HAS_BODY(message);
    ARROW_ASSIGN_OR_RAISE(auto reader, Buffer::GetReader(message.body()));
    return ReadDictionary(*message.metadata(), &dictionary_memo_, reader.get());
  }

  Status ReadInitialDictionaries() {
    // We must receive all dictionaries before reconstructing the
    // first record batch. Subsequent dictionary deltas modify the memo
    std::unique_ptr<Message> message;

    // TODO(wesm): In future, we may want to reconcile the ids in the stream with
    // those found in the schema
    for (int i = 0; i < dictionary_memo_.num_fields(); ++i) {
      RETURN_NOT_OK(message_reader_->ReadNextMessage(&message));
      if (!message) {
        if (i == 0) {
          /// ARROW-6006: If we fail to find any dictionaries in the stream, then
          /// it may be that the stream has a schema but no actual data. In such
          /// case we communicate that we were unable to find the dictionaries
          /// (but there was no failure otherwise), so the caller can decide what
          /// to do
          empty_stream_ = true;
          break;
        } else {
          // ARROW-6126, the stream terminated before receiving the expected
          // number of dictionaries
          return Status::Invalid("IPC stream ended without reading the expected number (",
                                 dictionary_memo_.num_fields(), ") of dictionaries");
        }
      }

      if (message->type() != Message::DICTIONARY_BATCH) {
        return Status::Invalid("IPC stream did not have the expected number (",
                               dictionary_memo_.num_fields(),
                               ") of dictionaries at the start of the stream");
      }
      RETURN_NOT_OK(ParseDictionary(*message));
    }

    read_initial_dictionaries_ = true;
    return Status::OK();
  }

  std::unique_ptr<MessageReader> message_reader_;

  bool read_initial_dictionaries_ = false;

  // Flag to set in case where we fail to observe all dictionaries in a stream,
  // and so the reader should not attempt to parse any messages
  bool empty_stream_ = false;

  DictionaryMemo dictionary_memo_;
  std::shared_ptr<Schema> schema_;
};

Status RecordBatchStreamReader::Open(std::unique_ptr<MessageReader> message_reader,
                                     std::shared_ptr<RecordBatchReader>* reader) {
  // Private ctor
  auto result = std::make_shared<RecordBatchStreamReaderImpl>();
  RETURN_NOT_OK(result->Open(std::move(message_reader)));
  *reader = result;
  return Status::OK();
}

Status RecordBatchStreamReader::Open(std::unique_ptr<MessageReader> message_reader,
                                     std::unique_ptr<RecordBatchReader>* reader) {
  // Private ctor
  auto result =
      std::unique_ptr<RecordBatchStreamReaderImpl>(new RecordBatchStreamReaderImpl());
  RETURN_NOT_OK(result->Open(std::move(message_reader)));
  *reader = std::move(result);
  return Status::OK();
}

Status RecordBatchStreamReader::Open(io::InputStream* stream,
                                     std::shared_ptr<RecordBatchReader>* out) {
  return Open(MessageReader::Open(stream), out);
}

Status RecordBatchStreamReader::Open(const std::shared_ptr<io::InputStream>& stream,
                                     std::shared_ptr<RecordBatchReader>* out) {
  return Open(MessageReader::Open(stream), out);
}

// ----------------------------------------------------------------------
// Reader implementation

static inline FileBlock FileBlockFromFlatbuffer(const flatbuf::Block* block) {
  return FileBlock{block->offset(), block->metaDataLength(), block->bodyLength()};
}

class RecordBatchFileReaderImpl : public RecordBatchFileReader {
 public:
  RecordBatchFileReaderImpl() : file_(NULLPTR), footer_offset_(0), footer_(NULLPTR) {}

  int num_record_batches() const override {
    return static_cast<int>(internal::FlatBuffersVectorSize(footer_->recordBatches()));
  }

  MetadataVersion version() const override {
    return internal::GetMetadataVersion(footer_->version());
  }

  Status ReadRecordBatch(int i, std::shared_ptr<RecordBatch>* batch) override {
    DCHECK_GE(i, 0);
    DCHECK_LT(i, num_record_batches());

    if (!read_dictionaries_) {
      RETURN_NOT_OK(ReadDictionaries());
      read_dictionaries_ = true;
    }

    std::unique_ptr<Message> message;
    RETURN_NOT_OK(ReadMessageFromBlock(GetRecordBatchBlock(i), &message));

    CHECK_HAS_BODY(*message);
    ARROW_ASSIGN_OR_RAISE(auto reader, Buffer::GetReader(message->body()));
    return ::arrow::ipc::ReadRecordBatch(*message->metadata(), schema_, &dictionary_memo_,
                                         reader.get(), batch);
  }

  Status Open(const std::shared_ptr<io::RandomAccessFile>& file, int64_t footer_offset) {
    owned_file_ = file;
    return Open(file.get(), footer_offset);
  }

  Status Open(io::RandomAccessFile* file, int64_t footer_offset) {
    file_ = file;
    footer_offset_ = footer_offset;
    RETURN_NOT_OK(ReadFooter());
    return ReadSchema();
  }

  std::shared_ptr<Schema> schema() const override { return schema_; }

 private:
  FileBlock GetRecordBatchBlock(int i) const {
    return FileBlockFromFlatbuffer(footer_->recordBatches()->Get(i));
  }

  FileBlock GetDictionaryBlock(int i) const {
    return FileBlockFromFlatbuffer(footer_->dictionaries()->Get(i));
  }

  Status ReadMessageFromBlock(const FileBlock& block, std::unique_ptr<Message>* out) {
    if (!BitUtil::IsMultipleOf8(block.offset) ||
        !BitUtil::IsMultipleOf8(block.metadata_length) ||
        !BitUtil::IsMultipleOf8(block.body_length)) {
      return Status::Invalid("Unaligned block in IPC file");
    }

    RETURN_NOT_OK(ReadMessage(block.offset, block.metadata_length, file_, out));

    // TODO(wesm): this breaks integration tests, see ARROW-3256
    // DCHECK_EQ((*out)->body_length(), block.body_length);
    return Status::OK();
  }

  Status ReadDictionaries() {
    // Read all the dictionaries
    for (int i = 0; i < num_dictionaries(); ++i) {
      std::unique_ptr<Message> message;
      RETURN_NOT_OK(ReadMessageFromBlock(GetDictionaryBlock(i), &message));

      CHECK_HAS_BODY(*message);
      ARROW_ASSIGN_OR_RAISE(auto reader, Buffer::GetReader(message->body()));
      RETURN_NOT_OK(
          ReadDictionary(*message->metadata(), &dictionary_memo_, reader.get()));
    }
    return Status::OK();
  }

  Status ReadSchema() {
    // Get the schema and record any observed dictionaries
    return internal::GetSchema(footer_->schema(), &dictionary_memo_, &schema_);
  }

  Status ReadFooter() {
    const int32_t magic_size = static_cast<int>(strlen(kArrowMagicBytes));

    if (footer_offset_ <= magic_size * 2 + 4) {
      return Status::Invalid("File is too small: ", footer_offset_);
    }

    int file_end_size = static_cast<int>(magic_size + sizeof(int32_t));
    ARROW_ASSIGN_OR_RAISE(auto buffer,
                          file_->ReadAt(footer_offset_ - file_end_size, file_end_size));

    const int64_t expected_footer_size = magic_size + sizeof(int32_t);
    if (buffer->size() < expected_footer_size) {
      return Status::Invalid("Unable to read ", expected_footer_size, "from end of file");
    }

    if (memcmp(buffer->data() + sizeof(int32_t), kArrowMagicBytes, magic_size)) {
      return Status::Invalid("Not an Arrow file");
    }

    int32_t footer_length = *reinterpret_cast<const int32_t*>(buffer->data());

    if (footer_length <= 0 || footer_length > footer_offset_ - magic_size * 2 - 4) {
      return Status::Invalid("File is smaller than indicated metadata size");
    }

    // Now read the footer
    ARROW_ASSIGN_OR_RAISE(
        footer_buffer_,
        file_->ReadAt(footer_offset_ - footer_length - file_end_size, footer_length));

    auto data = footer_buffer_->data();
    flatbuffers::Verifier verifier(data, footer_buffer_->size(), 128);
    if (!flatbuf::VerifyFooterBuffer(verifier)) {
      return Status::IOError("Verification of flatbuffer-encoded Footer failed.");
    }
    footer_ = flatbuf::GetFooter(data);

    return Status::OK();
  }

  int num_dictionaries() const {
    return static_cast<int>(internal::FlatBuffersVectorSize(footer_->dictionaries()));
  }

  io::RandomAccessFile* file_;

  std::shared_ptr<io::RandomAccessFile> owned_file_;

  // The location where the Arrow file layout ends. May be the end of the file
  // or some other location if embedded in a larger file.
  int64_t footer_offset_;

  // Footer metadata
  std::shared_ptr<Buffer> footer_buffer_;
  const flatbuf::Footer* footer_;

  bool read_dictionaries_ = false;
  DictionaryMemo dictionary_memo_;

  // Reconstructed schema, including any read dictionaries
  std::shared_ptr<Schema> schema_;
};

Status RecordBatchFileReader::Open(io::RandomAccessFile* file,
                                   std::shared_ptr<RecordBatchFileReader>* reader) {
  ARROW_ASSIGN_OR_RAISE(int64_t footer_offset, file->GetSize());
  return Open(file, footer_offset, reader);
}

Status RecordBatchFileReader::Open(io::RandomAccessFile* file, int64_t footer_offset,
                                   std::shared_ptr<RecordBatchFileReader>* out) {
  auto result = std::make_shared<RecordBatchFileReaderImpl>();
  RETURN_NOT_OK(result->Open(file, footer_offset));
  *out = result;
  return Status::OK();
}

Status RecordBatchFileReader::Open(const std::shared_ptr<io::RandomAccessFile>& file,
                                   std::shared_ptr<RecordBatchFileReader>* out) {
  ARROW_ASSIGN_OR_RAISE(int64_t footer_offset, file->GetSize());
  return Open(file, footer_offset, out);
}

Status RecordBatchFileReader::Open(const std::shared_ptr<io::RandomAccessFile>& file,
                                   int64_t footer_offset,
                                   std::shared_ptr<RecordBatchFileReader>* out) {
  auto result = std::make_shared<RecordBatchFileReaderImpl>();
  RETURN_NOT_OK(result->Open(file, footer_offset));
  *out = result;
  return Status::OK();
}

static Status ReadContiguousPayload(io::InputStream* file,
                                    std::unique_ptr<Message>* message) {
  RETURN_NOT_OK(ReadMessage(file, message));
  if (*message == nullptr) {
    return Status::Invalid("Unable to read metadata at offset");
  }
  return Status::OK();
}

Status ReadSchema(io::InputStream* stream, DictionaryMemo* dictionary_memo,
                  std::shared_ptr<Schema>* out) {
  std::unique_ptr<MessageReader> reader = MessageReader::Open(stream);
  std::unique_ptr<Message> message;
  RETURN_NOT_OK(reader->ReadNextMessage(&message));
  if (!message) {
    return Status::Invalid("Tried reading schema message, was null or length 0");
  }
  CHECK_MESSAGE_TYPE(Message::SCHEMA, message->type());
  return ReadSchema(*message, dictionary_memo, out);
}

Status ReadSchema(const Message& message, DictionaryMemo* dictionary_memo,
                  std::shared_ptr<Schema>* out) {
  std::shared_ptr<RecordBatchReader> reader;
  return internal::GetSchema(message.header(), dictionary_memo, &*out);
}

Status ReadRecordBatch(const std::shared_ptr<Schema>& schema,
                       const DictionaryMemo* dictionary_memo, io::InputStream* file,
                       std::shared_ptr<RecordBatch>* out) {
  auto options = IpcOptions::Defaults();
  std::unique_ptr<Message> message;
  RETURN_NOT_OK(ReadContiguousPayload(file, &message));
  CHECK_HAS_BODY(*message);
  ARROW_ASSIGN_OR_RAISE(auto reader, Buffer::GetReader(message->body()));
  return ReadRecordBatch(*message->metadata(), schema, dictionary_memo, options,
                         reader.get(), out);
}

Result<std::shared_ptr<Tensor>> ReadTensor(io::InputStream* file) {
  std::unique_ptr<Message> message;
  RETURN_NOT_OK(ReadContiguousPayload(file, &message));
  return ReadTensor(*message);
}

Result<std::shared_ptr<Tensor>> ReadTensor(const Message& message) {
  std::shared_ptr<DataType> type;
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
  std::vector<std::string> dim_names;
  CHECK_HAS_BODY(message);
  RETURN_NOT_OK(internal::GetTensorMetadata(*message.metadata(), &type, &shape, &strides,
                                            &dim_names));
  return Tensor::Make(type, message.body(), shape, strides, dim_names);
}

namespace {

Result<std::shared_ptr<SparseIndex>> ReadSparseCOOIndex(
    const flatbuf::SparseTensor* sparse_tensor, const std::vector<int64_t>& shape,
    int64_t non_zero_length, io::RandomAccessFile* file) {
  auto* sparse_index = sparse_tensor->sparseIndex_as_SparseTensorIndexCOO();
  const auto ndim = static_cast<int64_t>(shape.size());

  std::shared_ptr<DataType> indices_type;
  RETURN_NOT_OK(internal::GetSparseCOOIndexMetadata(sparse_index, &indices_type));
  const int64_t indices_elsize =
      checked_cast<const IntegerType&>(*indices_type).bit_width() / 8;

  auto* indices_buffer = sparse_index->indicesBuffer();
  ARROW_ASSIGN_OR_RAISE(auto indices_data,
                        file->ReadAt(indices_buffer->offset(), indices_buffer->length()));
  std::vector<int64_t> indices_shape({non_zero_length, ndim});
  auto* indices_strides = sparse_index->indicesStrides();
  std::vector<int64_t> strides(2);
  if (indices_strides && indices_strides->size() > 0) {
    if (indices_strides->size() != 2) {
      return Status::Invalid("Wrong size for indicesStrides in SparseCOOIndex");
    }
    strides[0] = indices_strides->Get(0);
    strides[1] = indices_strides->Get(1);
  } else {
    // Row-major by default
    strides[0] = indices_elsize * ndim;
    strides[1] = indices_elsize;
  }
  return std::make_shared<SparseCOOIndex>(
      std::make_shared<Tensor>(indices_type, indices_data, indices_shape, strides));
}

Result<std::shared_ptr<SparseIndex>> ReadSparseCSXIndex(
    const flatbuf::SparseTensor* sparse_tensor, const std::vector<int64_t>& shape,
    int64_t non_zero_length, io::RandomAccessFile* file) {
  if (shape.size() != 2) {
    return Status::Invalid("Invalid shape length for a sparse matrix");
  }

  auto* sparse_index = sparse_tensor->sparseIndex_as_SparseMatrixIndexCSX();

  std::shared_ptr<DataType> indptr_type, indices_type;
  RETURN_NOT_OK(
      internal::GetSparseCSXIndexMetadata(sparse_index, &indptr_type, &indices_type));

  auto* indptr_buffer = sparse_index->indptrBuffer();
  ARROW_ASSIGN_OR_RAISE(auto indptr_data,
                        file->ReadAt(indptr_buffer->offset(), indptr_buffer->length()));

  auto* indices_buffer = sparse_index->indicesBuffer();
  ARROW_ASSIGN_OR_RAISE(auto indices_data,
                        file->ReadAt(indices_buffer->offset(), indices_buffer->length()));

  std::vector<int64_t> indices_shape({non_zero_length});
  const auto indices_minimum_bytes =
      indices_shape[0] * checked_pointer_cast<FixedWidthType>(indices_type)->bit_width() /
      CHAR_BIT;
  if (indices_minimum_bytes > indices_buffer->length()) {
    return Status::Invalid("shape is inconsistent to the size of indices buffer");
  }

  switch (sparse_index->compressedAxis()) {
    case flatbuf::SparseMatrixCompressedAxis::Row: {
      std::vector<int64_t> indptr_shape({shape[0] + 1});
      const int64_t indptr_minimum_bytes =
          indptr_shape[0] *
          checked_pointer_cast<FixedWidthType>(indptr_type)->bit_width() / CHAR_BIT;
      if (indptr_minimum_bytes > indptr_buffer->length()) {
        return Status::Invalid("shape is inconsistent to the size of indptr buffer");
      }
      return std::make_shared<SparseCSRIndex>(
          std::make_shared<Tensor>(indptr_type, indptr_data, indptr_shape),
          std::make_shared<Tensor>(indices_type, indices_data, indices_shape));
    }
    case flatbuf::SparseMatrixCompressedAxis::Column: {
      std::vector<int64_t> indptr_shape({shape[1] + 1});
      const int64_t indptr_minimum_bytes =
          indptr_shape[0] *
          checked_pointer_cast<FixedWidthType>(indptr_type)->bit_width() / CHAR_BIT;
      if (indptr_minimum_bytes > indptr_buffer->length()) {
        return Status::Invalid("shape is inconsistent to the size of indptr buffer");
      }
      return std::make_shared<SparseCSCIndex>(
          std::make_shared<Tensor>(indptr_type, indptr_data, indptr_shape),
          std::make_shared<Tensor>(indices_type, indices_data, indices_shape));
    }
    default:
      return Status::Invalid("Invalid value of SparseMatrixCompressedAxis");
  }
}

Result<std::shared_ptr<SparseTensor>> MakeSparseTensorWithSparseCOOIndex(
    const std::shared_ptr<DataType>& type, const std::vector<int64_t>& shape,
    const std::vector<std::string>& dim_names,
    const std::shared_ptr<SparseCOOIndex>& sparse_index, int64_t non_zero_length,
    const std::shared_ptr<Buffer>& data) {
  return SparseCOOTensor::Make(sparse_index, type, data, shape, dim_names);
}

Result<std::shared_ptr<SparseTensor>> MakeSparseTensorWithSparseCSRIndex(
    const std::shared_ptr<DataType>& type, const std::vector<int64_t>& shape,
    const std::vector<std::string>& dim_names,
    const std::shared_ptr<SparseCSRIndex>& sparse_index, int64_t non_zero_length,
    const std::shared_ptr<Buffer>& data) {
  return SparseCSRMatrix::Make(sparse_index, type, data, shape, dim_names);
}

Result<std::shared_ptr<SparseTensor>> MakeSparseTensorWithSparseCSCIndex(
    const std::shared_ptr<DataType>& type, const std::vector<int64_t>& shape,
    const std::vector<std::string>& dim_names,
    const std::shared_ptr<SparseCSCIndex>& sparse_index, int64_t non_zero_length,
    const std::shared_ptr<Buffer>& data) {
  return SparseCSCMatrix::Make(sparse_index, type, data, shape, dim_names);
}

Status ReadSparseTensorMetadata(const Buffer& metadata,
                                std::shared_ptr<DataType>* out_type,
                                std::vector<int64_t>* out_shape,
                                std::vector<std::string>* out_dim_names,
                                int64_t* out_non_zero_length,
                                SparseTensorFormat::type* out_format_id,
                                const flatbuf::SparseTensor** out_fb_sparse_tensor,
                                const flatbuf::Buffer** out_buffer) {
  RETURN_NOT_OK(internal::GetSparseTensorMetadata(
      metadata, out_type, out_shape, out_dim_names, out_non_zero_length, out_format_id));

  const flatbuf::Message* message;
  RETURN_NOT_OK(internal::VerifyMessage(metadata.data(), metadata.size(), &message));

  auto sparse_tensor = message->header_as_SparseTensor();
  if (sparse_tensor == nullptr) {
    return Status::IOError(
        "Header-type of flatbuffer-encoded Message is not SparseTensor.");
  }
  *out_fb_sparse_tensor = sparse_tensor;

  auto buffer = sparse_tensor->data();
  if (!BitUtil::IsMultipleOf8(buffer->offset())) {
    return Status::Invalid(
        "Buffer of sparse index data did not start on 8-byte aligned offset: ",
        buffer->offset());
  }
  *out_buffer = buffer;

  return Status::OK();
}

}  // namespace

namespace internal {

namespace {

Result<size_t> GetSparseTensorBodyBufferCount(SparseTensorFormat::type format_id) {
  switch (format_id) {
    case SparseTensorFormat::COO:
      return 2;

    case SparseTensorFormat::CSR:
      return 3;

    case SparseTensorFormat::CSC:
      return 3;

    default:
      return Status::Invalid("Unrecognized sparse tensor format");
  }
}

Status CheckSparseTensorBodyBufferCount(
    const IpcPayload& payload, SparseTensorFormat::type sparse_tensor_format_id) {
  size_t expected_body_buffer_count = 0;
  ARROW_ASSIGN_OR_RAISE(expected_body_buffer_count,
                        GetSparseTensorBodyBufferCount(sparse_tensor_format_id));
  if (payload.body_buffers.size() != expected_body_buffer_count) {
    return Status::Invalid("Invalid body buffer count for a sparse tensor");
  }

  return Status::OK();
}

}  // namespace

Result<size_t> ReadSparseTensorBodyBufferCount(const Buffer& metadata) {
  SparseTensorFormat::type format_id;

  RETURN_NOT_OK(internal::GetSparseTensorMetadata(metadata, nullptr, nullptr, nullptr,
                                                  nullptr, &format_id));
  return GetSparseTensorBodyBufferCount(format_id);
}

Result<std::shared_ptr<SparseTensor>> ReadSparseTensorPayload(const IpcPayload& payload) {
  std::shared_ptr<DataType> type;
  std::vector<int64_t> shape;
  std::vector<std::string> dim_names;
  int64_t non_zero_length;
  SparseTensorFormat::type sparse_tensor_format_id;
  const flatbuf::SparseTensor* sparse_tensor;
  const flatbuf::Buffer* buffer;

  RETURN_NOT_OK(ReadSparseTensorMetadata(*payload.metadata, &type, &shape, &dim_names,
                                         &non_zero_length, &sparse_tensor_format_id,
                                         &sparse_tensor, &buffer));

  RETURN_NOT_OK(CheckSparseTensorBodyBufferCount(payload, sparse_tensor_format_id));

  switch (sparse_tensor_format_id) {
    case SparseTensorFormat::COO: {
      std::shared_ptr<SparseCOOIndex> sparse_index;
      std::shared_ptr<DataType> indices_type;
      RETURN_NOT_OK(internal::GetSparseCOOIndexMetadata(
          sparse_tensor->sparseIndex_as_SparseTensorIndexCOO(), &indices_type));
      ARROW_ASSIGN_OR_RAISE(sparse_index,
                            SparseCOOIndex::Make(indices_type, shape, non_zero_length,
                                                 payload.body_buffers[0]));
      return MakeSparseTensorWithSparseCOOIndex(type, shape, dim_names, sparse_index,
                                                non_zero_length, payload.body_buffers[1]);
    }
    case SparseTensorFormat::CSR: {
      std::shared_ptr<SparseCSRIndex> sparse_index;
      std::shared_ptr<DataType> indptr_type;
      std::shared_ptr<DataType> indices_type;
      RETURN_NOT_OK(internal::GetSparseCSXIndexMetadata(
          sparse_tensor->sparseIndex_as_SparseMatrixIndexCSX(), &indptr_type,
          &indices_type));
      ARROW_CHECK_EQ(indptr_type, indices_type);
      ARROW_ASSIGN_OR_RAISE(
          sparse_index,
          SparseCSRIndex::Make(indices_type, shape, non_zero_length,
                               payload.body_buffers[0], payload.body_buffers[1]));
      return MakeSparseTensorWithSparseCSRIndex(type, shape, dim_names, sparse_index,
                                                non_zero_length, payload.body_buffers[2]);
    }
    case SparseTensorFormat::CSC: {
      std::shared_ptr<SparseCSCIndex> sparse_index;
      std::shared_ptr<DataType> indptr_type;
      std::shared_ptr<DataType> indices_type;
      RETURN_NOT_OK(internal::GetSparseCSXIndexMetadata(
          sparse_tensor->sparseIndex_as_SparseMatrixIndexCSX(), &indptr_type,
          &indices_type));
      ARROW_CHECK_EQ(indptr_type, indices_type);
      ARROW_ASSIGN_OR_RAISE(
          sparse_index,
          SparseCSCIndex::Make(indices_type, shape, non_zero_length,
                               payload.body_buffers[0], payload.body_buffers[1]));
      return MakeSparseTensorWithSparseCSCIndex(type, shape, dim_names, sparse_index,
                                                non_zero_length, payload.body_buffers[2]);
    }
    default:
      return Status::Invalid("Unsupported sparse index format");
  }
}

}  // namespace internal

Result<std::shared_ptr<SparseTensor>> ReadSparseTensor(const Buffer& metadata,
                                                       io::RandomAccessFile* file) {
  std::shared_ptr<DataType> type;
  std::vector<int64_t> shape;
  std::vector<std::string> dim_names;
  int64_t non_zero_length;
  SparseTensorFormat::type sparse_tensor_format_id;
  const flatbuf::SparseTensor* sparse_tensor;
  const flatbuf::Buffer* buffer;

  RETURN_NOT_OK(ReadSparseTensorMetadata(metadata, &type, &shape, &dim_names,
                                         &non_zero_length, &sparse_tensor_format_id,
                                         &sparse_tensor, &buffer));

  ARROW_ASSIGN_OR_RAISE(auto data, file->ReadAt(buffer->offset(), buffer->length()));

  std::shared_ptr<SparseIndex> sparse_index;
  switch (sparse_tensor_format_id) {
    case SparseTensorFormat::COO: {
      ARROW_ASSIGN_OR_RAISE(
          sparse_index, ReadSparseCOOIndex(sparse_tensor, shape, non_zero_length, file));
      return MakeSparseTensorWithSparseCOOIndex(
          type, shape, dim_names, checked_pointer_cast<SparseCOOIndex>(sparse_index),
          non_zero_length, data);
    }
    case SparseTensorFormat::CSR: {
      ARROW_ASSIGN_OR_RAISE(
          sparse_index, ReadSparseCSXIndex(sparse_tensor, shape, non_zero_length, file));
      return MakeSparseTensorWithSparseCSRIndex(
          type, shape, dim_names, checked_pointer_cast<SparseCSRIndex>(sparse_index),
          non_zero_length, data);
    }
    case SparseTensorFormat::CSC: {
      ARROW_ASSIGN_OR_RAISE(
          sparse_index, ReadSparseCSXIndex(sparse_tensor, shape, non_zero_length, file));
      return MakeSparseTensorWithSparseCSCIndex(
          type, shape, dim_names, checked_pointer_cast<SparseCSCIndex>(sparse_index),
          non_zero_length, data);
    }
    default:
      return Status::Invalid("Unsupported sparse index format");
  }
}

Result<std::shared_ptr<SparseTensor>> ReadSparseTensor(const Message& message) {
  CHECK_HAS_BODY(message);
  ARROW_ASSIGN_OR_RAISE(auto reader, Buffer::GetReader(message.body()));
  return ReadSparseTensor(*message.metadata(), reader.get());
}

Result<std::shared_ptr<SparseTensor>> ReadSparseTensor(io::InputStream* file) {
  std::unique_ptr<Message> message;
  RETURN_NOT_OK(ReadContiguousPayload(file, &message));
  CHECK_MESSAGE_TYPE(Message::SPARSE_TENSOR, message->type());
  CHECK_HAS_BODY(*message);
  ARROW_ASSIGN_OR_RAISE(auto reader, Buffer::GetReader(message->body()));
  return ReadSparseTensor(*message->metadata(), reader.get());
}

///////////////////////////////////////////////////////////////////////////
// Helpers for fuzzing

namespace internal {

Status FuzzIpcStream(const uint8_t* data, int64_t size) {
  auto buffer = std::make_shared<Buffer>(data, size);
  io::BufferReader buffer_reader(buffer);

  std::shared_ptr<RecordBatchReader> batch_reader;
  RETURN_NOT_OK(RecordBatchStreamReader::Open(&buffer_reader, &batch_reader));

  while (true) {
    std::shared_ptr<arrow::RecordBatch> batch;
    RETURN_NOT_OK(batch_reader->ReadNext(&batch));
    if (batch == nullptr) {
      break;
    }
    RETURN_NOT_OK(batch->ValidateFull());
  }

  return Status::OK();
}

Status FuzzIpcFile(const uint8_t* data, int64_t size) {
  auto buffer = std::make_shared<Buffer>(data, size);
  io::BufferReader buffer_reader(buffer);

  std::shared_ptr<RecordBatchFileReader> batch_reader;
  RETURN_NOT_OK(RecordBatchFileReader::Open(&buffer_reader, &batch_reader));

  const int n_batches = batch_reader->num_record_batches();
  for (int i = 0; i < n_batches; ++i) {
    std::shared_ptr<arrow::RecordBatch> batch;
    RETURN_NOT_OK(batch_reader->ReadRecordBatch(i, &batch));
    RETURN_NOT_OK(batch->ValidateFull());
  }

  return Status::OK();
}

}  // namespace internal
}  // namespace ipc
}  // namespace arrow
