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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_SLICE_DELAYING_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_SLICE_DELAYING_H_

#include <map>
#include <set>
#include <vector>

#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "absl/container/flat_hash_map.h"

namespace xla {

// An HLO pass that delay slice instruction to merge the same operations
// togather.
class SliceDelaying: public HloModulePass {
 public:
  tensorflow::StringPiece name() const override { return "slice delaying"; }

  StatusOr<bool> Run(HloModule* module) override;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_SLICE_DELAYING_H_
