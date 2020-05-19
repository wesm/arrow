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

#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "arrow/compute/kernel.h"
#include "arrow/status.h"
#include "arrow/testing/gtest_util.h"
#include "arrow/type.h"
#include "arrow/util/key_value_metadata.h"

namespace arrow {
namespace compute {

// ----------------------------------------------------------------------
// InputType

TEST(InputType, AnyTypeConstructor) {
  // Check the ANY_TYPE ctors
  InputType ty;
  ASSERT_EQ(InputType::ANY_TYPE, ty.kind());
  ASSERT_EQ(ValueDescr::ANY, ty.shape());

  ty = InputType(ValueDescr::SCALAR);
  ASSERT_EQ(ValueDescr::SCALAR, ty.shape());

  ty = InputType(ValueDescr::ARRAY);
  ASSERT_EQ(ValueDescr::ARRAY, ty.shape());
}

TEST(InputType, Constructors) {
  // Exact type constructor
  InputType ty1(int8());
  ASSERT_EQ(InputType::EXACT_TYPE, ty1.kind());
  ASSERT_EQ(ValueDescr::ANY, ty1.shape());
  AssertTypeEqual(*int8(), *ty1.type());

  InputType ty1_implicit = int8();
  ASSERT_TRUE(ty1.Equals(ty1_implicit));

  InputType ty1_array(int8(), ValueDescr::ARRAY);
  ASSERT_EQ(ValueDescr::ARRAY, ty1_array.shape());

  InputType ty1_scalar(int8(), ValueDescr::SCALAR);
  ASSERT_EQ(ValueDescr::SCALAR, ty1_scalar.shape());

  // Same type id constructor
  InputType ty2 = Type::DECIMAL;
  ASSERT_EQ(InputType::SAME_TYPE_ID, ty2.kind());

  InputType ty2_array(Type::DECIMAL, ValueDescr::ARRAY);
  ASSERT_EQ(ValueDescr::ARRAY, ty2_array.shape());

  InputType ty2_scalar(Type::DECIMAL, ValueDescr::SCALAR);
  ASSERT_EQ(ValueDescr::SCALAR, ty2_scalar.shape());

  // Implicit construction in a vector
  std::vector<InputType> types = {int8(), Type::DECIMAL};
  ASSERT_TRUE(types[0].Equals(ty1));
  ASSERT_TRUE(types[1].Equals(ty2));

  // Copy constructor
  InputType ty3 = ty1;
  InputType ty4 = ty2;
  ASSERT_TRUE(ty3.Equals(ty1));
  ASSERT_TRUE(ty4.Equals(ty2));

  // Move constructor
  InputType ty5 = std::move(ty3);
  InputType ty6 = std::move(ty4);
  ASSERT_TRUE(ty5.Equals(ty1));
  ASSERT_TRUE(ty6.Equals(ty2));

  // ToString
  ASSERT_EQ("any[int8]", ty1.ToString());
  ASSERT_EQ("array[int8]", ty1_array.ToString());
  ASSERT_EQ("scalar[int8]", ty1_scalar.ToString());

  ASSERT_EQ("any[decimal*]", ty2.ToString());
  ASSERT_EQ("array[decimal*]", ty2_array.ToString());
  ASSERT_EQ("scalar[decimal*]", ty2_scalar.ToString());
}

TEST(InputType, Equals) {
  InputType t1 = int8();
  InputType t2 = int8();
  InputType t3(int8(), ValueDescr::ARRAY);
  InputType t3_i32(int32(), ValueDescr::ARRAY);
  InputType t3_scalar(int8(), ValueDescr::SCALAR);
  InputType t4(int8(), ValueDescr::ARRAY);
  InputType t4_i32(int32(), ValueDescr::ARRAY);

  InputType t5 = Type::DECIMAL;
  InputType t6 = Type::DECIMAL;
  InputType t7(Type::DECIMAL, ValueDescr::SCALAR);
  InputType t7_i32(Type::INT32, ValueDescr::SCALAR);
  InputType t8(Type::DECIMAL, ValueDescr::SCALAR);
  InputType t8_i32(Type::INT32, ValueDescr::SCALAR);

  ASSERT_TRUE(t1.Equals(t2));
  ASSERT_EQ(t1, t2);

  // ANY vs SCALAR
  ASSERT_NE(t1, t3);

  ASSERT_EQ(t3, t4);

  // both ARRAY, but different type
  ASSERT_NE(t3, t3_i32);

  // ARRAY vs SCALAR
  ASSERT_NE(t3, t3_scalar);

  ASSERT_EQ(t3_i32, t4_i32);

  ASSERT_FALSE(t1.Equals(t5));
  ASSERT_NE(t1, t5);

  ASSERT_EQ(t5, t5);
  ASSERT_EQ(t5, t6);
  ASSERT_NE(t5, t7);
  ASSERT_EQ(t7, t8);
  ASSERT_EQ(t7, t8);
  ASSERT_NE(t7, t7_i32);
  ASSERT_EQ(t7_i32, t8_i32);

  // NOTE: For the time being, we treat int32() and Type::INT32 as being
  // different. This could obviously be fixed later to make these equivalent
  ASSERT_NE(InputType(int8()), InputType(Type::INT32));

  // Check that field metadata excluded from equality checks
  InputType t9 = list(
      field("item", utf8(), /*nullable=*/true, key_value_metadata({"foo"}, {"bar"})));
  InputType t10 = list(field("item", utf8()));
  ASSERT_TRUE(t9.Equals(t10));
}

TEST(InputType, Hash) {
  InputType t0;
  InputType t0_scalar(ValueDescr::SCALAR);
  InputType t0_array(ValueDescr::ARRAY);

  InputType t1 = int8();
  InputType t2 = Type::DECIMAL;

  // These checks try to determine first of all whether Hash always returns the
  // same value, and whether the elements of the type are all incorporated into
  // the Hash
  ASSERT_EQ(t0.Hash(), t0.Hash());
  ASSERT_NE(t0.Hash(), t0_scalar.Hash());
  ASSERT_NE(t0.Hash(), t0_array.Hash());
  ASSERT_NE(t0_scalar.Hash(), t0_array.Hash());

  ASSERT_EQ(t1.Hash(), t1.Hash());
  ASSERT_EQ(t2.Hash(), t2.Hash());

  ASSERT_NE(t0.Hash(), t1.Hash());
  ASSERT_NE(t0.Hash(), t2.Hash());
  ASSERT_NE(t1.Hash(), t2.Hash());
}

TEST(InputType, Matches) {
  InputType ty1 = int8();

  ASSERT_TRUE(ty1.Matches(ValueDescr::Scalar(int8())));
  ASSERT_TRUE(ty1.Matches(ValueDescr::Array(int8())));
  ASSERT_TRUE(ty1.Matches(ValueDescr::Any(int8())));
  ASSERT_FALSE(ty1.Matches(ValueDescr::Any(int16())));

  InputType ty2 = Type::DECIMAL;
  ASSERT_TRUE(ty2.Matches(ValueDescr::Scalar(decimal(12, 2))));
  ASSERT_TRUE(ty2.Matches(ValueDescr::Array(decimal(12, 2))));
  ASSERT_FALSE(ty2.Matches(ValueDescr::Any(float64())));

  InputType ty3(int64(), ValueDescr::SCALAR);
  ASSERT_FALSE(ty3.Matches(ValueDescr::Array(int64())));
  ASSERT_TRUE(ty3.Matches(ValueDescr::Scalar(int64())));
  ASSERT_FALSE(ty3.Matches(ValueDescr::Scalar(int32())));
  ASSERT_FALSE(ty3.Matches(ValueDescr::Any(int64())));
}

// ----------------------------------------------------------------------
// OutputType

TEST(OutputType, Constructors) {
  OutputType ty1 = int8();
  ASSERT_EQ(OutputType::FIXED, ty1.kind());
  AssertTypeEqual(*int8(), *ty1.type());

  auto DummyResolver = [](const std::vector<ValueDescr>& args) {
    return ValueDescr(int32(), GetBroadcastShape(args));
  };
  OutputType ty2(DummyResolver);
  ASSERT_EQ(OutputType::COMPUTED, ty2.kind());

  ASSERT_OK_AND_ASSIGN(ValueDescr out_descr2, ty2.Resolve({}));
  ASSERT_EQ(ValueDescr::Scalar(int32()), out_descr2);

  // Copy constructor
  OutputType ty3 = ty1;
  ASSERT_EQ(OutputType::FIXED, ty3.kind());
  AssertTypeEqual(*ty1.type(), *ty3.type());

  OutputType ty4 = ty2;
  ASSERT_EQ(OutputType::COMPUTED, ty4.kind());
  ASSERT_OK_AND_ASSIGN(ValueDescr out_descr4, ty4.Resolve({}));
  ASSERT_EQ(ValueDescr::Scalar(int32()), out_descr4);

  // Move constructor
  OutputType ty5 = std::move(ty1);
  ASSERT_EQ(OutputType::FIXED, ty5.kind());
  AssertTypeEqual(*int8(), *ty5.type());

  OutputType ty6 = std::move(ty4);
  ASSERT_EQ(OutputType::COMPUTED, ty6.kind());
  ASSERT_OK_AND_ASSIGN(ValueDescr out_descr6, ty6.Resolve({}));
  ASSERT_EQ(ValueDescr::Scalar(int32()), out_descr6);

  // ToString

  // ty1 was copied to ty3
  ASSERT_EQ("int8", ty3.ToString());
  ASSERT_EQ("computed", ty2.ToString());
}

TEST(OutputType, Resolve) {
  // Check shape promotion rules for FIXED kind
  OutputType ty1(int32());

  ASSERT_OK_AND_ASSIGN(ValueDescr descr, ty1.Resolve({}));
  ASSERT_EQ(ValueDescr::Scalar(int32()), descr);

  ASSERT_OK_AND_ASSIGN(descr, ty1.Resolve({ValueDescr(int8(), ValueDescr::SCALAR)}));
  ASSERT_EQ(ValueDescr::Scalar(int32()), descr);

  ASSERT_OK_AND_ASSIGN(descr, ty1.Resolve({ValueDescr(int8(), ValueDescr::SCALAR),
                                           ValueDescr(int8(), ValueDescr::ARRAY)}));
  ASSERT_EQ(ValueDescr::Array(int32()), descr);

  OutputType ty2([](const std::vector<ValueDescr>& args) -> Result<ValueDescr> {
    return ValueDescr(args[0].type, GetBroadcastShape(args));
  });

  ASSERT_OK_AND_ASSIGN(descr, ty2.Resolve({ValueDescr::Array(utf8())}));
  ASSERT_EQ(ValueDescr::Array(utf8()), descr);

  // Type resolver that returns an error
  OutputType ty3([](const std::vector<ValueDescr>& args) -> Result<ValueDescr> {
    // NB: checking the value types versus the function arity should be
    // validated elsewhere, so this is just for illustration purposes
    if (args.size() == 0) {
      return Status::Invalid("Need at least one argument");
    }
    return ValueDescr(args[0]);
  });
  ASSERT_RAISES(Invalid, ty3.Resolve({}));
}

TEST(OutputType, ResolveDescr) {
  ValueDescr d1 = ValueDescr::Scalar(int32());
  ValueDescr d2 = ValueDescr::Array(int32());

  OutputType ty1(d1);
  OutputType ty2(d2);

  ASSERT_EQ(ValueDescr::SCALAR, ty1.shape());
  ASSERT_EQ(ValueDescr::ARRAY, ty2.shape());

  {
    ASSERT_OK_AND_ASSIGN(ValueDescr descr, ty1.Resolve({}));
    ASSERT_EQ(d1, descr);
  }

  {
    ASSERT_OK_AND_ASSIGN(ValueDescr descr, ty2.Resolve({}));
    ASSERT_EQ(d2, descr);
  }
}

// ----------------------------------------------------------------------
// KernelSignature

TEST(KernelSignature, Basics) {
  // (any[int8], scalar[decimal]) -> utf8
  std::vector<InputType> in_types({int8(), InputType(Type::DECIMAL, ValueDescr::SCALAR)});
  OutputType out_type(utf8());

  KernelSignature sig(in_types, out_type);
  ASSERT_EQ(2, sig.in_types().size());
  ASSERT_TRUE(sig.in_types()[0].type()->Equals(*int8()));
  ASSERT_TRUE(sig.in_types()[0].Matches(ValueDescr::Scalar(int8())));
  ASSERT_TRUE(sig.in_types()[0].Matches(ValueDescr::Array(int8())));

  ASSERT_TRUE(sig.in_types()[1].Matches(ValueDescr::Scalar(decimal(12, 2))));
  ASSERT_FALSE(sig.in_types()[1].Matches(ValueDescr::Array(decimal(12, 2))));
}

TEST(KernelSignature, Equals) {
  KernelSignature sig1({}, utf8());
  KernelSignature sig1_copy({}, utf8());
  KernelSignature sig2({int8()}, utf8());

  // Output type doesn't matter (for now)
  KernelSignature sig3({int8()}, int32());

  KernelSignature sig4({int8(), int16()}, utf8());
  KernelSignature sig4_copy({int8(), int16()}, utf8());
  KernelSignature sig5({int8(), int16(), int32()}, utf8());

  // Differ in shape
  KernelSignature sig6({ValueDescr::Scalar(int8())}, utf8());
  KernelSignature sig7({ValueDescr::Array(int8())}, utf8());

  ASSERT_EQ(sig1, sig1);

  ASSERT_EQ(sig2, sig3);
  ASSERT_NE(sig3, sig4);

  // Different sig objects, but same sig
  ASSERT_EQ(sig1, sig1_copy);
  ASSERT_EQ(sig4, sig4_copy);

  // Match first 2 args, but not third
  ASSERT_NE(sig4, sig5);

  ASSERT_NE(sig6, sig7);
}

TEST(KernelSignature, VarargsEquals) {
  KernelSignature sig1({int8()}, utf8(), /*is_varargs=*/true);
  KernelSignature sig2({int8()}, utf8(), /*is_varargs=*/true);
  KernelSignature sig3({int8()}, utf8());

  ASSERT_EQ(sig1, sig2);
  ASSERT_NE(sig2, sig3);
}

TEST(KernelSignature, Hash) {
  // Some basic tests to ensure that the hashes are deterministic and that all
  // input arguments are incorporated
  KernelSignature sig1({}, utf8());
  KernelSignature sig2({int8()}, utf8());
  KernelSignature sig3({int8(), int32()}, utf8());

  ASSERT_EQ(sig1.Hash(), sig1.Hash());
  ASSERT_EQ(sig2.Hash(), sig2.Hash());
  ASSERT_NE(sig1.Hash(), sig2.Hash());
  ASSERT_NE(sig2.Hash(), sig3.Hash());
}

TEST(KernelSignature, MatchesInputs) {
  // () -> boolean
  KernelSignature sig1({}, boolean());

  ASSERT_TRUE(sig1.MatchesInputs({}));
  ASSERT_FALSE(sig1.MatchesInputs({int8()}));

  // (any[int8], any[decimal]) -> boolean
  KernelSignature sig2({int8(), Type::DECIMAL}, boolean());

  ASSERT_FALSE(sig2.MatchesInputs({}));
  ASSERT_FALSE(sig2.MatchesInputs({int8()}));
  ASSERT_TRUE(sig2.MatchesInputs({int8(), decimal(12, 2)}));
  ASSERT_TRUE(sig2.MatchesInputs(
      {ValueDescr::Scalar(int8()), ValueDescr::Scalar(decimal(12, 2))}));
  ASSERT_TRUE(
      sig2.MatchesInputs({ValueDescr::Array(int8()), ValueDescr::Array(decimal(12, 2))}));

  // (scalar[int8], array[int32]) -> boolean
  KernelSignature sig3({ValueDescr::Scalar(int8()), ValueDescr::Array(int32())},
                       boolean());

  ASSERT_FALSE(sig3.MatchesInputs({}));

  // Unqualified, these are ANY type and do not match because the kernel
  // requires a scalar and an array
  ASSERT_FALSE(sig3.MatchesInputs({int8(), int32()}));
  ASSERT_TRUE(
      sig3.MatchesInputs({ValueDescr::Scalar(int8()), ValueDescr::Array(int32())}));
  ASSERT_FALSE(
      sig3.MatchesInputs({ValueDescr::Array(int8()), ValueDescr::Array(int32())}));
}

TEST(KernelSignature, VarargsMatchesInputs) {
  KernelSignature sig({int8()}, utf8(), /*is_varargs=*/true);

  std::vector<ValueDescr> args = {int8()};
  ASSERT_TRUE(sig.MatchesInputs(args));
  args.push_back(ValueDescr::Scalar(int8()));
  args.push_back(ValueDescr::Array(int8()));
  ASSERT_TRUE(sig.MatchesInputs(args));
  args.push_back(int32());
  ASSERT_FALSE(sig.MatchesInputs(args));
}

TEST(KernelSignature, ToString) {
  std::vector<InputType> in_types = {InputType(int8(), ValueDescr::SCALAR),
                                     InputType(Type::DECIMAL, ValueDescr::ARRAY),
                                     InputType(utf8())};
  KernelSignature sig(in_types, utf8());
  ASSERT_EQ("(scalar[int8], array[decimal*], any[string]) -> string", sig.ToString());

  OutputType out_type(
      [](const std::vector<ValueDescr>& args) { return Status::Invalid("NYI"); });
  KernelSignature sig2({int8(), Type::DECIMAL}, out_type);
  ASSERT_EQ("(any[int8], any[decimal*]) -> computed", sig2.ToString());
}

TEST(KernelSignature, VarargsToString) {
  KernelSignature sig({int8()}, utf8(), /*is_varargs=*/true);
  ASSERT_EQ("varargs[any[int8]] -> string", sig.ToString());
}

}  // namespace compute
}  // namespace arrow
