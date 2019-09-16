/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Tests that diag_slice operations can be performed.

#include <numeric>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/array3d.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"

namespace xla {
namespace {

class DiagSliceTest : public ClientLibraryTestBase {};

TEST_F(DiagSliceTest, Slice3x3x3_To_3x3_F32) {
  Array3D<float> values(3, 3, 3);
  values.FillIota(0);

  XlaBuilder builder(TestName());
  auto original = ConstantR3FromArray3D<float>(&builder, values);
  DiagSlice(original);

  Array2D<float> expected{
      {0.0, 4.0, 8.0}, {9.0, 13.0, 17.0}, {18.0, 22.0, 26.0}};
  ComputeAndCompareR2<float>(&builder, expected, {}, ErrorSpec(0.000001));
}

TEST_F(DiagSliceTest, Slice3x4_To_3_F32) {
  Array2D<float> values(3, 4);
  values.FillIota(0);

  XlaBuilder builder(TestName());
  auto original = ConstantR2FromArray2D<float>(&builder, values);
  DiagSlice(original);

  absl::InlinedVector<float, 3> expected;
  for (int i = 0; i < 3; ++i) {
    expected.push_back(i * 5);
  }

  ComputeAndCompareR1<float>(&builder, expected, {}, ErrorSpec(0.000001));
}

TEST_F(DiagSliceTest, Slice4x3_To_3_F32) {
  Array2D<float> values(4, 3);
  values.FillIota(0);

  XlaBuilder builder(TestName());
  auto original = ConstantR2FromArray2D<float>(&builder, values);
  DiagSlice(original);

  absl::InlinedVector<float, 3> expected;
  for (int i = 0; i < 3; ++i) {
    expected.push_back(i * 4);
  }

  ComputeAndCompareR1<float>(&builder, expected, {}, ErrorSpec(0.000001));
}

TEST_F(DiagSliceTest, Slice3x4_To_3_Positive_Offset_F32) {
  Array2D<float> values(3, 4);
  values.FillIota(0);

  XlaBuilder builder(TestName());
  auto original = ConstantR2FromArray2D<float>(&builder, values);
  DiagSlice(original, 1);

  absl::InlinedVector<float, 3> expected;
  for (int i = 0; i < 3; ++i) {
    expected.push_back(i * 5 + 1);
  }

  ComputeAndCompareR1<float>(&builder, expected, {}, ErrorSpec(0.000001));
}

TEST_F(DiagSliceTest, Slice3x4_To_2_Negative_Offset_F32) {
  Array2D<float> values(3, 4);
  values.FillIota(0);

  XlaBuilder builder(TestName());
  auto original = ConstantR2FromArray2D<float>(&builder, values);
  DiagSlice(original, -1);

  absl::InlinedVector<float, 3> expected;
  for (int i = 0; i < 2; ++i) {
    expected.push_back(i * 5 + 4);
  }

  ComputeAndCompareR1<float>(&builder, expected, {}, ErrorSpec(0.000001));
}

TEST_F(DiagSliceTest, Slice4x3_To_2_Positive_Offset_F32) {
  Array2D<float> values(4, 3);
  values.FillIota(0);

  XlaBuilder builder(TestName());
  auto original = ConstantR2FromArray2D<float>(&builder, values);
  DiagSlice(original, 1);

  absl::InlinedVector<float, 2> expected;
  for (int i = 0; i < 2; ++i) {
    expected.push_back(i * 4 + 1);
  }

  ComputeAndCompareR1<float>(&builder, expected, {}, ErrorSpec(0.000001));
}

TEST_F(DiagSliceTest, Slice4x3_To_3_Negative_Offset_F32) {
  Array2D<float> values(4, 3);
  values.FillIota(0);

  XlaBuilder builder(TestName());
  auto original = ConstantR2FromArray2D<float>(&builder, values);
  DiagSlice(original, -1);

  absl::InlinedVector<float, 3> expected;
  for (int i = 0; i < 3; ++i) {
    expected.push_back(i * 4 + 3);
  }

  ComputeAndCompareR1<float>(&builder, expected, {}, ErrorSpec(0.000001));
}

}  // namespace
}  // namespace xla
