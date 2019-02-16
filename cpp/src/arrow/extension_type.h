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
#include <string>

#include "arrow/array.h"
#include "arrow/type.h"
#include "arrow/util/visibility.h"

namespace arrow {

/// \brief The base class for custom / user-defined types.
class ARROW_EXPORT ExtensionType : public DataType {
 public:
  std::shared_ptr<DataType> storage_type() const { return storage_type_; }

 protected:
  explicit ExtensionType(std::shared_ptr<DataType> storage_type)
    : DataType(Type::EXTENSION),
      storage_type_(storage_type) {}

  std::shared_ptr<DataType> storage_type_;
};

class ARROW_EXPORT ExtensionArray : public Array {
 protected:
  ExtensionArray
};

/// \brief Serializer interface for user-defined types
class ExtensionTypeAdapter {
 public:
  /// \brief Wrap built-in Array type in a user-defined ExtensionArray instance
  /// \param[in] ext_type an instance of ExtensionType
  /// \param[in] data the physical storage for the extension type
  virtual std::shared_ptr<Array> WrapArray(std::shared_ptr<DataType> ext_type,
                                           std::shared_ptr<Array> data) = 0;

  virtual std::shared_ptr<DataType> Deserialize(const std::string& serialized) = 0;

  virtual std::string& Serialize(const ExtensionType& type) = 0;
};

/// \brief
ARROW_EXPORT
Status RegisterExtensionType(const std::string& type_name,
                             std::unique_ptr<ExtensionTypeAdapter> wrapper);

ARROW_EXPORT
Status UnregisterExtensionType(const std::string& type_name);

ARROW_EXPORT
ExtensionTypeAdapter* GetExtensionType(const std::string& type_name);

}  // namespace arrow
