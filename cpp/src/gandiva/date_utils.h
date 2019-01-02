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

#ifndef TO_DATE_HELPER_H
#define TO_DATE_HELPER_H

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(_MSC_VER)
#include <ctime>
#include <iomanip>
#include <sstream>

#include "arrow/vendored/date.h"
#endif

#include "arrow/util/macros.h"

#include "gandiva/arrow.h"
#include "gandiva/visibility.h"

namespace gandiva {

/// \brief Utility class for converting sql date patterns to internal date patterns.
class GANDIVA_EXPORT DateUtils {
 public:
  static Status ToInternalFormat(const std::string& format,
                                 std::shared_ptr<std::string>* internal_format);

 private:
  using date_format_converter = std::unordered_map<std::string, std::string>;

  static date_format_converter sql_date_format_to_boost_map_;

  static date_format_converter InitMap();

  static std::vector<std::string> GetMatches(std::string pattern, bool exactMatch);

  static std::vector<std::string> GetPotentialMatches(const std::string& pattern);

  static std::vector<std::string> GetExactMatches(const std::string& pattern);
};

namespace internal {

/// \brief Returns seconds since the UNIX epoch
static inline bool ParseTimestamp(const char* buf, const char* format, int64_t* out) {
#if defined(_MSC_VER)
  static std::locale lc_all(setlocale(LC_ALL, NULLPTR));
  std::istringstream stream(buf);
  stream.imbue(lc_all);

  date::sys_seconds secs;
  stream >> date::parse(format, secs);
  if (stream.fail()) {
    return false;
  }

  *out = secs.time_since_epoch().count();
  return true;
#else
  struct tm result;
  char* ret = strptime(buf, format, &result);
  if (ret == NULLPTR) {
    return false;
  }
  // ignore the time part
  *out = date::sys_days(date::year(result.tm_year + 1900) / (result.tm_mon + 1) /
                        result.tm_mday);
  *out = secs.time_since_epoch().count();
  return true;
#endif
}

}  // namespace internal
}  // namespace gandiva

#endif  // TO_DATE_HELPER_H
