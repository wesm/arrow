# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

build_features <- c(
  arrow_info()$capabilities,
  # Special handling for "uncompressed", for tests that iterate over compressions
  uncompressed = TRUE
)

skip_if_not_available <- function(feature) {
  yes <- feature %in% names(build_features) && build_features[feature]
  if (!yes) {
    skip(paste("Arrow C++ not built with", feature))
  }
}

skip_if_no_pyarrow <- function() {
  skip_if_not_installed("reticulate")
  if (!reticulate::py_module_available("pyarrow")) {
    skip("pyarrow not available for testing")
  }
}

skip_if_not_dev_mode <- function() {
  skip_if_not(
    identical(tolower(Sys.getenv("ARROW_R_DEV")), "true"),
    "environment variable ARROW_R_DEV"
  )
}

skip_if_not_running_large_memory_tests <- function() {
  skip_if_not(
    identical(tolower(Sys.getenv("ARROW_LARGE_MEMORY_TESTS")), "true"),
    "environment variable ARROW_LARGE_MEMORY_TESTS"
  )
}

process_is_running <- function(x) {
  cmd <- sprintf("ps aux | grep '%s' | grep -v grep", x)
  tryCatch(system(cmd, ignore.stdout = TRUE) == 0, error = function(e) FALSE)
}
