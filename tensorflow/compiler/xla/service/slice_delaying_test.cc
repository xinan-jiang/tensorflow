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

#include <memory>
#include "tensorflow/compiler/xla/service/slice_delaying.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/service/pattern_matcher_gmock.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

using ::testing::ElementsAre;
namespace m = match;

class SliceDelayingTest : public HloTestBase {
};

TEST_F(SliceDelayingTest, Basic) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[8,8] parameter(0)
      p1 = f32[8,8] parameter(1)
      s00 = f32[2,8] slice(f32[8,8] p0), slice={[0:2], [0:8]}
      s01 = f32[6,8] slice(f32[8,8] p0), slice={[2:8], [0:8]}
      s10 = f32[2,8] slice(f32[8,8] p1), slice={[0:2], [0:8]}
      s11 = f32[6,8] slice(f32[8,8] p1), slice={[2:8], [0:8]}
      add0 = f32[2,8] add(f32[2,8] s00, f32[2,8] s10)
      add1 = f32[6,8] add(f32[6,8] s01, f32[6,8] s11)
      ROOT tuple = (f32[2,8], f32[6,8]) tuple(add0, add1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  EXPECT_EQ(9, module->entry_computation()->instruction_count());
  SliceDelaying slice_delaying;
  TF_ASSERT_OK_AND_ASSIGN(bool result,
                          RunHloPass(&slice_delaying, module.get()));
  EXPECT_TRUE(result);
  HloDCE dce;
  TF_ASSERT_OK_AND_ASSIGN(result,
                          RunHloPass(&dce, module.get()));
  EXPECT_TRUE(result);
  EXPECT_EQ(6, module->entry_computation()->instruction_count());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(m::Slice(m::Add(m::Parameter(0), m::Parameter(1))),
          m::Slice(m::Add(m::Parameter(0), m::Parameter(1))))));
}

}  // namespace
}  // namespace xla
