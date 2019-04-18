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

#include "tensorflow/compiler/xla/service/slice_delaying.h"
#include <algorithm>
#include <utility>
#include <set>
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {

namespace {

class SliceDelayer {
 public:
  // Returns whether the instruction has been visited and computed change cost
  bool IsVisited(const HloInstruction* instruction) const;

  // Bundles all the slices from the same operand.
  // Returns whether bundle new slices.
  bool BundleSlices(const HloInstruction* inst);

  // Returns whether the slices are delayed successfully.
  Status MergeWithPeers(const HloInstruction* inst);

  // Elimenate dead instructions.
  void EliminateDeadInstructions();

  // Clear containers.
  void Clear();

 private:
  // Returns whether slice is a bundled-slice of operand
  bool IsBundledSlice(const HloInstruction* operand,
                      const HloInstruction* slice) const;

  // Collects true operands of inst.
  StatusOr<std::vector<HloInstruction*>> GetTrueOperands(
      const HloInstruction* inst) const;

  // Collects true users of operands with the same opcode of inst, and update
  // visited_users_
  StatusOr<std::vector<HloInstruction*>> GetTrueUsers(
      const HloInstruction* inst,
      const std::vector<HloInstruction*>& operands);

  // Generate new operation instead of sliced operations, then slice the result.
  // Record the bundled-slices in bundled_slices_.
  void GenerateNewOp(const std::vector<HloInstruction*>& operands,
      const std::vector<HloInstruction*>& users);

  absl::flat_hash_map<const HloInstruction*,
                      std::set<const HloInstruction*>> bundled_slices_;

  std::set<HloInstruction*> visited_users_;

  std::set<HloInstruction*> removed_;
};

// Computes the cost of implimentation of delaying slice, and returns whether
// it should be changed.
bool ShouldReplace(const std::vector<HloInstruction*>& operands,
    const std::vector<HloInstruction*>& users) {
  // operands and user have the same shape because of elementwise operation
  int64 sum = 0;
  for (HloInstruction* user : users) {
    sum += xla::ShapeUtil::ElementsIn(user->shape());
  }
  return sum >= xla::ShapeUtil::ElementsIn(operands[0]->shape());
}

}  // namespace

bool SliceDelayer::IsVisited(const HloInstruction* instruction) const {
  return std::find(visited_users_.begin(), visited_users_.end(), instruction)
      != visited_users_.end();
}

bool SliceDelayer::BundleSlices(const HloInstruction* inst) {
  auto iter = bundled_slices_.find(inst);
  // slice has been collected into bundled_slices_
  if (iter != bundled_slices_.end()) {
    return false;
  }

  VLOG(0) << "CheckOperand: inst " << inst->ToString();
  std::set<const HloInstruction*> slices;
  bool collected = false;
  for (auto user : inst->users()) {
    if (user->opcode() == HloOpcode::kSlice) {
  VLOG(0) << "CheckOperand: slice user " << user->ToString();
      slices.insert(user);
      collected = true;
    }
  }

  for (auto slice : slices) {
    VLOG(0) << "CheckOperand: " << slice->ToString();
  }
  if (!slices.empty()) {
    bundled_slices_.insert(std::make_pair(inst, slices));
  }
  return collected;
}

// =================================Before======================================
//
//       +-----operand-----+        <operands>
//       |                 |
//       v                 v
// bundled-slice     bundled-slice   <bundled-slices>
//       |                 |
//       v                 v
//      user              user      <users>
//
// ==================================After======================================
//
//            operand
//               |
//               v
//       +----new-user-----+
//       |                 |
//       v                 v
// bundled-slice     bundled-slice   <bundled-slices>
//
Status SliceDelayer::MergeWithPeers(const HloInstruction* inst) {
  TF_ASSIGN_OR_RETURN(std::vector<HloInstruction*> operands,
                      GetTrueOperands(inst));
  for (int64 i = 0; i < operands.size(); ++i) {
    VLOG(0) << "CheckOperand: ops[" << i << "]: " << operands[i]->ToString();
  }
  TF_ASSIGN_OR_RETURN(std::vector<HloInstruction*> users,
                      GetTrueUsers(inst, operands));
  for (int64 i = 0; i < users.size(); ++i) {
    VLOG(0) << "CheckOperand: users[" << i << "]: " << users[i]->ToString();
  }

  // Change HLO graph
  GenerateNewOp(operands, users);
  return Status::OK();
}

void SliceDelayer::EliminateDeadInstructions() {
  // Remove dead users
  for (auto inst : removed_) {
    VLOG(0) << "Delete: " << inst->ToString();
    inst->parent()->RemoveInstruction(inst);
  }
  // Remove dead slice
  for (auto pair : bundled_slices_) {
    for (const HloInstruction* inst : pair.second) {
      if (inst->user_count() == 0) {
       HloInstruction* replica = const_cast<HloInstruction*>(inst);
       replica->parent()->RemoveInstruction(replica);
      }
    }  // end for slices
  }  // end for bundled_slices
}

void SliceDelayer::Clear() {
  bundled_slices_.clear();
  visited_users_.clear();
  removed_.clear();
}

bool SliceDelayer::IsBundledSlice(
    const HloInstruction* operand, const HloInstruction* slice) const {
  auto iter_m = bundled_slices_.find(operand);
  if (bundled_slices_.end() == iter_m) {
    return false;
  }
  auto iter_s = iter_m->second.find(slice);
  if (iter_m->second.end() == iter_s) {
    return false;
  }
  VLOG(0) << "Split: " << operand->ToString() << "\nSlice: "
          << slice->ToString();
  return true;
}

StatusOr<std::vector<HloInstruction*>> SliceDelayer::GetTrueOperands(
    const HloInstruction* inst) const {
  std::vector<HloInstruction*> operands;
  // Check operand:
  // the inst's operand is a bundled-slice of the true operand.
  // the operands-vector keeps the true-operands.
  for (HloInstruction* slice : inst->operands()) {
    if (slice->opcode() != HloOpcode::kSlice) {
      return tensorflow::errors::FailedPrecondition(
          "Operation's operand should be bundled slice");
    }
    HloInstruction* operand = slice->mutable_operand(0);
    if (!IsBundledSlice(operand, slice)) {
      return tensorflow::errors::FailedPrecondition(
          "Operation's operand should be bundled slice");
    }
    operands.push_back(operand);
  }
  for (int64 i = 0; i < operands.size(); ++i) {
    VLOG(0) << "CheckOperand: ops[" << i << "]: " << operands[i]->ToString();
  }

  // Check operands:
  // operands should have the same shape.
  const Shape shape = operands[0]->shape();
  for (const HloInstruction* operand : operands) {
    // Only support element-wise now
    if (!ShapeUtil::Equal(operand->shape(), shape)) {
      return tensorflow::errors::FailedPrecondition(
          "Operation's true operand should be the same shape");
    }
  }
  for (int64 i = 0; i < operands.size(); ++i) {
    VLOG(0) << "CheckOperand: ops[" << i << "]: " << operands[i]->ToString();
  }
  return operands;
}

StatusOr<std::vector<HloInstruction*>> SliceDelayer::GetTrueUsers(
    const HloInstruction* inst,
    const std::vector<HloInstruction*>& operands) {
  std::vector<HloInstruction*> users;
  HloInstruction* operand0 = operands[0];

  for (const HloInstruction* slice_0 : bundled_slices_.find(operand0)->second) {
    VLOG(0) << "GetUserSlice: " << slice_0->ToString();

    for (HloInstruction* user : slice_0->users()) {
      VLOG(0) << "TryGetUser: " << user->ToString();
      // user should be the same operation and same operand count as inst
      // skip the visited user to avoid redundant computation
      if (IsVisited(user) || user->opcode() != inst->opcode() ||
          user->operand_count() != inst->operand_count()) {
        continue;
      }

      // user's operand should be a bundled-slice of the true operand in order
      bool bContinue = false;
      for (int64 j = 0; j < operands.size(); ++j) {
        const HloInstruction* slice_j = user->operand(j);
        if (!IsBundledSlice(operands[j], slice_j)
            || slice_0->slice_starts() != slice_j->slice_starts()
            || slice_0->slice_limits() != slice_j->slice_limits()
            || slice_0->slice_strides() != slice_j->slice_strides()) {
          bContinue = true;
          break;
        }
      }
      if (bContinue) {
        continue;
      }

      // found the user
      VLOG(0) << "GetUser: " << user->ToString();
      users.push_back(user);
      visited_users_.insert(user);
      break;
    }  // end for loop slice_0->users
  }
  for (int64 i = 0; i < users.size(); ++i) {
    VLOG(0) << "CheckOperand: users[" << i << "]: " << users[i]->ToString();
  }

  if (users.empty()) {
    return tensorflow::errors::FailedPrecondition(
        "No found valid users");
  } else if (!ShouldReplace(operands, users)) {
    return tensorflow::errors::FailedPrecondition(
        "No Enough elements slice");
  } else {
    return users;
  }
}

void SliceDelayer::GenerateNewOp(const std::vector<HloInstruction*>& operands,
    const std::vector<HloInstruction*>& users) {
  // generate new ops
  const Shape shape = operands[0]->shape();
  HloComputation* computation = users[0]->parent();
  auto new_op = computation->AddInstruction(
      users[0]->CloneWithNewOperands(shape, operands));
  VLOG(0) << "Add NewOp: " << new_op->ToString();

  std::set<const HloInstruction*> slices;
  // replace the old user and its slice operands with new operation and its
  // slice user.
  for (int64 i = 0; i < users.size(); ++i) {
    HloInstruction* user = users[i];
    const HloInstruction* slice = user->operand(0);
    auto new_user = computation->AddInstruction(
        slice->CloneWithNewOperands(user->shape(), {new_op}));
    VLOG(0) << "Add NewSlice: " << new_user->ToString()
            << "\nReplace: " << user->ToString();
    user->ReplaceAllUsesWith(new_user);
    slices.insert(new_user);
    removed_.insert(user);
  }  // end for users

  bundled_slices_.insert(std::make_pair(new_op, slices));
}

StatusOr<bool> SliceDelaying::Run(HloModule* module) {
  VLOG(0) << "Run Pass: " << name();
  VLOG(0) << "before: " << name() << "\n" << module->ToString();
  SliceDelayer slice_delayer;
  bool changed = false;

  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* instruction :
        computation->MakeInstructionPostOrder()) {
      // Skip the removed instruction
      if (slice_delayer.IsVisited(instruction)) {
        VLOG(0) << "The instruction has been removed"
                << instruction->ToString();
        continue;
      }

      // Bundles bundled-slices and tries merge elementwise instruction with its
      // peers.
      if (instruction->opcode() == HloOpcode::kSlice) {
        VLOG(0) << "Bundle slice: " << instruction->ToString();
        slice_delayer.BundleSlices(instruction->mutable_operand(0));
      } else if (instruction->IsElementwise()
                 && instruction->operand_count() != 0) {
        // TODO(xinan): more other instructions
        VLOG(0) << "Merge inst: " << instruction->ToString();
        changed |= slice_delayer.MergeWithPeers(instruction).ok();
      }
    }  // end for instructions in computation
  }  // end for computations in module

  // Clears dead nodes
  slice_delayer.EliminateDeadInstructions();
  slice_delayer.Clear();
  VLOG(0) << "after: " << name() << "\n" <<  module->ToString();
  return changed;
}

}  // namespace xla
