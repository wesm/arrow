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

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>

#include <gtest/gtest.h>

#include "arrow/buffer.h"
#include "arrow/memory_pool.h"
#include "arrow/result.h"
#include "arrow/testing/gtest_common.h"
#include "arrow/testing/gtest_util.h"
#include "arrow/testing/util.h"
#include "arrow/util/bit_block_counter.h"
#include "arrow/util/bit_util.h"

namespace arrow {
namespace internal {

class TestBitBlockCounter : public ::testing::Test {
 public:
  void Create(int64_t nbytes, int64_t offset, int64_t length) {
    ASSERT_OK_AND_ASSIGN(buf_, AllocateBuffer(nbytes));
    // Start with data zeroed out
    std::memset(buf_->mutable_data(), 0, nbytes);
    counter_.reset(new BitBlockCounter(buf_->data(), offset, length));
  }

 protected:
  std::shared_ptr<Buffer> buf_;
  std::unique_ptr<BitBlockCounter> counter_;
};

static constexpr int64_t kWordSize = 64;

TEST_F(TestBitBlockCounter, OneWordBasics) {
  const int64_t nbytes = 1024;

  Create(nbytes, 0, nbytes * 8);

  int64_t bits_scanned = 0;
  for (int64_t i = 0; i < nbytes / 8; ++i) {
    BitBlockCount block = counter_->NextWord();
    ASSERT_EQ(block.length, kWordSize);
    ASSERT_EQ(block.popcount, 0);
    bits_scanned += block.length;
  }
  ASSERT_EQ(bits_scanned, 1024 * 8);

  auto block = counter_->NextWord();
  ASSERT_EQ(block.length, 0);
  ASSERT_EQ(block.popcount, 0);
}

TEST_F(TestBitBlockCounter, FourWordsBasics) {
  const int64_t nbytes = 1024;

  Create(nbytes, 0, nbytes * 8);

  int64_t bits_scanned = 0;
  for (int64_t i = 0; i < nbytes / 32; ++i) {
    BitBlockCount block = counter_->NextFourWords();
    ASSERT_EQ(block.length, 4 * kWordSize);
    ASSERT_EQ(block.popcount, 0);
    bits_scanned += block.length;
  }
  ASSERT_EQ(bits_scanned, 1024 * 8);

  auto block = counter_->NextFourWords();
  ASSERT_EQ(block.length, 0);
  ASSERT_EQ(block.popcount, 0);
}

TEST_F(TestBitBlockCounter, OneWordWithOffsets) {
  auto CheckWithOffset = [&](int64_t offset) {
    const int64_t nwords = 4;

    const int64_t total_bytes = nwords * 8 + 1;
    // Trim a bit from the end of the bitmap so we can check the remainder bits
    // behavior
    Create(total_bytes, offset, nwords * kWordSize - offset - 1);

    // Start with data all set
    std::memset(buf_->mutable_data(), 0xFF, total_bytes);

    BitBlockCount block = counter_->NextWord();
    ASSERT_EQ(kWordSize, block.length);
    ASSERT_EQ(block.popcount, 64);

    // Add a false value to the next word
    BitUtil::SetBitTo(buf_->mutable_data(), kWordSize + offset, false);
    block = counter_->NextWord();
    ASSERT_EQ(block.length, 64);
    ASSERT_EQ(block.popcount, 63);

    // Set the next word to all false
    BitUtil::SetBitsTo(buf_->mutable_data(), 2 * kWordSize + offset, kWordSize, false);

    if (offset == 0) {
      block = counter_->NextWord();
      ASSERT_EQ(block.length, 64);
      ASSERT_EQ(block.popcount, 0);

      // Last block
      block = counter_->NextWord();
      ASSERT_EQ(block.length, kWordSize - 1);
      ASSERT_EQ(block.length, block.popcount);
    } else {
      // In the last block, we are unable to do the required bit-shifting so
      // more than 64 bits are returned.
      block = counter_->NextWord();
      ASSERT_EQ(block.length, 2 * kWordSize - offset - 1);

      // The first 64 bits were zeroed
      ASSERT_EQ(block.popcount, block.length - 64);
    }

    // We can keep calling NextWord safely
    block = counter_->NextWord();
    ASSERT_EQ(block.length, 0);
    ASSERT_EQ(block.popcount, 0);
  };

  for (int64_t offset_i = 0; offset_i < 8; ++offset_i) {
    CheckWithOffset(offset_i);
  }
}

TEST_F(TestBitBlockCounter, FourWordsWithOffsets) {
  auto CheckWithOffset = [&](int64_t offset) {
    const int64_t nwords = 15;

    const int64_t total_bytes = nwords * 8 + 1;
    // Trim a bit from the end of the bitmap so we can check the remainder bits
    // behavior
    Create(total_bytes, offset, nwords * kWordSize - offset - 1);

    // Start with data all set
    std::memset(buf_->mutable_data(), 0xFF, total_bytes);

    BitBlockCount block = counter_->NextFourWords();
    ASSERT_EQ(4 * kWordSize, block.length);
    ASSERT_EQ(block.popcount, 256);

    // Add some false values to the next 3 shifted words
    BitUtil::SetBitTo(buf_->mutable_data(), 4 * kWordSize + offset, false);
    BitUtil::SetBitTo(buf_->mutable_data(), 5 * kWordSize + offset, false);
    BitUtil::SetBitTo(buf_->mutable_data(), 6 * kWordSize + offset, false);
    block = counter_->NextFourWords();

    ASSERT_EQ(block.length, 256);
    ASSERT_EQ(block.popcount, 253);

    BitUtil::SetBitsTo(buf_->mutable_data(), 8 * kWordSize + offset, 2 * kWordSize,
                       false);

    block = counter_->NextFourWords();
    ASSERT_EQ(block.length, 256);
    ASSERT_EQ(block.popcount, 128);

    // Last block
    block = counter_->NextFourWords();
    ASSERT_EQ(block.length, 3 * kWordSize - offset - 1);
    ASSERT_EQ(block.length, block.popcount);

    // We can keep calling NextFourWords safely
    block = counter_->NextFourWords();
    ASSERT_EQ(block.length, 0);
    ASSERT_EQ(block.popcount, 0);
  };

  for (int64_t offset_i = 0; offset_i < 8; ++offset_i) {
    CheckWithOffset(offset_i);
  }
}

TEST_F(TestBitBlockCounter, FourWordsRandomData) {
  const int64_t nbytes = 1024;
  auto buffer = *AllocateBuffer(nbytes);
  random_bytes(nbytes, 0, buffer->mutable_data());

  auto CheckWithOffset = [&](int64_t offset) {
    BitBlockCounter counter(buffer->data(), offset, nbytes * 8 - offset);
    for (int64_t i = 0; i < nbytes / 32; ++i) {
      BitBlockCount block = counter.NextFourWords();
      ASSERT_EQ(block.popcount,
                CountSetBits(buffer->data(), i * 256 + offset, block.length));
    }
  };
  for (int64_t offset_i = 0; offset_i < 8; ++offset_i) {
    CheckWithOffset(offset_i);
  }
}

TEST(TestBinaryBitBlockCounter, NextAndWord) {
  const int64_t nbytes = 1024;
  auto left = *AllocateBuffer(nbytes);
  auto right = *AllocateBuffer(nbytes);
  random_bytes(nbytes, 0, left->mutable_data());
  random_bytes(nbytes, 0, right->mutable_data());

  auto CheckWithOffsets = [&](int left_offset, int right_offset) {
    int64_t overlap_length = nbytes * 8 - std::max(left_offset, right_offset);
    BinaryBitBlockCounter counter(left->data(), left_offset, right->data(), right_offset,
                                  overlap_length);
    int64_t position = 0;
    while (true) {
      BitBlockCount block = counter.NextAndWord();
      if (block.length == 0) {
        break;
      }
      int expected_popcount = 0;
      for (int j = 0; j < block.length; ++j) {
        expected_popcount +=
            static_cast<int>(BitUtil::GetBit(left->data(), position + left_offset + j) &&
                             BitUtil::GetBit(right->data(), position + right_offset + j));
      }
      ASSERT_EQ(block.popcount, expected_popcount);
      position += block.length;
    }
    // We made it through all the data
    ASSERT_EQ(position, overlap_length);

    BitBlockCount block = counter.NextAndWord();
    ASSERT_EQ(block.length, 0);
    ASSERT_EQ(block.popcount, 0);
  };

  for (int left_i = 0; left_i < 7; ++left_i) {
    for (int right_i = 0; right_i < 7; ++right_i) {
      CheckWithOffsets(left_i, right_i);
    }
  }
}

}  // namespace internal
}  // namespace arrow
