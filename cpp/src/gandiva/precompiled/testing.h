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

#include <ctime>

#include <gtest/gtest.h>
#include "gandiva/date_utils.h"
#include "gandiva/precompiled/types.h"

namespace gandiva {

timestamp StringToTimestamp(const char* buf) {
  struct tm tm;
  memset(&tm, 0, sizeof(struct tm));
  internal::strptime_compat(buf, "%Y-%m-%d %H:%M:%S", &tm);

  struct tm epoch;
  memset(&epoch, 0, sizeof(struct tm));
  epoch.tm_year = 70;
  epoch.tm_mday = 1;

  // Return as milliseconds
  return static_cast<int64_t>(difftime(mktime(&tm), mktime(&epoch))) * 1000;
}

}  // namespace gandiva
