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
#include <map>
#include <set>
#include <vector>

namespace xla {

namespace {

class SliceDelayer {
 public:
  // Returns wheter the instruction has been removed.
  bool IsRemoved(const HloInstruction* instruction);

  // Bundles the slices from the same operand if they are equivalent to a split
  // operation. Returns whether bundle new slices.
  bool BundleSlices(const HloInstruction* inst);

  // Returns whether the slices are delayed successfully.
  Status MergeWithPeers(const HloInstruction* inst);

  // Elimenate dead instructions.
  void EliminateDeadInstructions();

  // Clear containers.
  void Clear();

 private:
  // Returns whether slice is a split-slice of operand
  bool IsSplitSlice(const HloInstruction* operand, const HloInstruction* slice);

  // Returns i-th slice of operand
  const HloInstruction* GetSlice(const HloInstruction* operand, int64 i);

  // Collects true operands of inst, Returns whether the collection succeed.
  StatusOr<std::vector<HloInstruction*>> GetTrueOperands(
      const HloInstruction* inst);

  // Collects true users of operands with the same opcode of inst.
  // Returns whether the collection succeed.
  StatusOr<std::vector<HloInstruction*>> GetSlicingUsers(
      const HloInstruction* inst,
      const std::vector< HloInstruction*>& operands);

  // Generate new operation instead of sliced operations, then slice the result.
  // Record the split-slices in split_slices_.
  void GenerateNewOp(const std::vector<HloInstruction*>& operands,
      const std::vector<HloInstruction*>& users);

  absl::flat_hash_map<const HloInstruction*,
                      std::vector<const HloInstruction*>> split_slices_;

  std::set<HloInstruction*> to_remove_;
};

bool IsUnstridedSlice(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kSlice &&
      absl::c_all_of(inst->slice_strides(),
                     [](int64 stride) { return stride == 1; });
}

// Returns the dimension which the tensor is splited along
// Returns -1 if no dimension could be found or more than one dimension
int64 GetSplitedDim(const HloInstruction* inst, const HloInstruction* slice) {
  const Shape shape = inst->shape();
  const Shape slice_shape = slice->shape();
  if (ShapeUtil::TrueRank(shape) != ShapeUtil::TrueRank(slice_shape))
    return -1;

  int64 slice_dimension = -1;
  for (int64 i = 0; i < ShapeUtil::TrueRank(shape); ++i) {
    if (shape.dimensions(i) == slice_shape.dimensions(i))
      continue;
    // more then one dimension
    if (slice_dimension != -1)
      return -1;
    slice_dimension = i;
  }
  return slice_dimension;
}

// Returns whether the slices cover each element
bool IsSlicesContinuous(
    const std::vector<const HloInstruction*>& slices, int64 dim) {
  int64 k = slices[0]->slice_limits(dim);
  for (int64 i = 1; i < slices.size(); ++i) {
    if (slices[i]->slice_starts(dim) != k) {
      return false;
    }
    k = slices[i]->slice_limits(dim);
  }
  return true;
}

// Returns whether the slices are equivalent to a split operation
bool IsSplitSlices(const HloInstruction* inst, int64 dim,
    const std::vector<const HloInstruction*>& slices) {
  if (!IsSlicesContinuous(slices, dim))
    return false;
  const HloInstruction* first = slices.front();
  const HloInstruction* last = slices.back();
  if (last->slice_limits(dim) != inst->shape().dimensions(dim)
      || first->slice_starts(dim) != 0) {
    // TODO(xinan): support partially split
    return false;
  }
  return true;
}

}  // namespace

bool SliceDelayer::IsRemoved(const HloInstruction* instruction) {
  return std::find(to_remove_.begin(), to_remove_.end(), instruction)
      != to_remove_.end();
}

bool SliceDelayer::BundleSlices(const HloInstruction* inst) {
  auto iter = split_slices_.find(inst);
  // slice has been collected into split_slices_
  if (iter != split_slices_.end())
    return false;

  // Check whether slice is a candidate of split-slices.
  // First just check slice self-attribute.
  std::map<int64, std::vector<const HloInstruction*>> dim_slices;
  for (auto user : inst->users()) {
    if (!IsUnstridedSlice(user))
      continue;
    int64 dim = GetSplitedDim(inst, user);
    if (dim == -1)
      continue;
    dim_slices[dim].push_back(user);
  }

  // Then, check the relation between slices.
  for (auto pair : dim_slices) {
    int64 dim = pair.first;
    std::vector<const HloInstruction*>& vec = pair.second;
    if (vec.size() < 2)
      continue;
    std::sort(vec.begin(), vec.end(),
              [](const HloInstruction* lhs, const HloInstruction* rhs) {
                return lhs->slice_starts() < rhs->slice_starts();
              });
    if (IsSplitSlices(inst, dim, vec)) {
      split_slices_.insert(std::make_pair(inst, vec));
      // TODO(xinan): keep more splits for a struction
      return true;
    }
  }
  return false;
}

// =================================Before======================================
//
//      +----operand----+        <operands>
//      |               |
//      v               v
// split-slice     split-slice   <split-slices>
//      |               |
//      v               v
//     user            user      <users>
//
// ==================================After======================================
//
//           operand
//              |
//              v
//      +---new-user----+
//      |               |
//      v               v
// split-slice     split-slice   <split-slices>
//
Status SliceDelayer::MergeWithPeers(const HloInstruction* inst) {
  TF_ASSIGN_OR_RETURN(std::vector<HloInstruction*> operands,
                      GetTrueOperands(inst));
  for (int64 i = 0; i < operands.size(); ++i) {
    VLOG(0) << "CheckOperand: ops[" << i << "]: " << operands[i]->ToString();
  }
  TF_ASSIGN_OR_RETURN(std::vector<HloInstruction*> users,
                      GetSlicingUsers(inst, operands));
  for (int64 i = 0; i < users.size(); ++i) {
    VLOG(0) << "CheckOperand: users[" << i << "]: " << users[i]->ToString();
  }

  // Change HLO graph
  GenerateNewOp(operands, users);
  return Status::OK();
}

void SliceDelayer::EliminateDeadInstructions() {
  // Remove dead users
  for (auto inst : to_remove_) {
    VLOG(0) << "Delete: " << inst->ToString();
    inst->parent()->RemoveInstruction(inst);
  }
  // Remove dead slice
  for (auto pair : split_slices_) {
    for (const HloInstruction* inst : pair.second) {
      if (inst->user_count() == 0) {
       HloInstruction* replica = const_cast<HloInstruction*>(inst);
       replica->parent()->RemoveInstruction(replica);
      }
    }  // end for slices
  }  // end for split_slices
}

void SliceDelayer::Clear() {
  split_slices_.clear();
  to_remove_.clear();
}

bool SliceDelayer::IsSplitSlice(
    const HloInstruction* operand, const HloInstruction* slice) {
  auto iter_m = split_slices_.find(operand);
  if (split_slices_.end() == iter_m)
    return false;
  auto iter_s = std::find(iter_m->second.begin(), iter_m->second.end(), slice);
  if (iter_m->second.end() == iter_s)
    return false;
  VLOG(0) << "Split: " << operand->ToString() << "\nSlice: "
          << slice->ToString();
  return true;
}

const HloInstruction* SliceDelayer::GetSlice(
    const HloInstruction* operand, int64 i) {
  return split_slices_.find(operand)->second[i];
}

StatusOr<std::vector<HloInstruction*>> SliceDelayer::GetTrueOperands(
    const HloInstruction* inst) {
  std::vector<HloInstruction*> operands;
  // Check operand:
  // the inst's operand is a split-slice of the true operand.
  // the operands-vector keeps the true-operands.
  for (HloInstruction* slice : inst->operands()) {
    if (!IsUnstridedSlice(slice)) {
      return tensorflow::errors::FailedPrecondition(
          "Operation's operand should be unstride slice");
    }
    HloInstruction* operand = slice->mutable_operand(0);
    if (!IsSplitSlice(operand, slice)) {
      return tensorflow::errors::FailedPrecondition(
          "Operation's operand should be split slice");
    }
    operands.push_back(operand);
  }

  for (int64 i = 0; i < operands.size(); ++i) {
    VLOG(0) << "CheckOperand: ops[" << i << "]: " << operands[i]->ToString();
  }

  // Check operands:
  // operands should have the same shape and the same num_split,
  // Additionally num_split should be the same as user-count.
  const Shape shape = operands[0]->shape();
  int64 split_size = split_slices_.find(operands[0])->second.size();
  for (const HloInstruction* operand : operands) {
    // Only support element-wise now
    if (!ShapeUtil::Equal(operand->shape(), shape)) {
      return tensorflow::errors::FailedPrecondition(
          "Operation's true operand should be the same shape");
    }
    if (split_slices_.find(operand)->second.size() != split_size) {
      return tensorflow::errors::FailedPrecondition(
          "Operation's true operand should split to the same number of shards");
    }
  }
  // Check split-slices:
  // operands' split-slices should be in the same location adn same shape.
  for (int64 i = 0; i < split_size; ++i) {
    const HloInstruction *slice_0 = GetSlice(operands[0], i);
    for (int64 j = 1; j < operands.size(); ++j) {
      const HloInstruction *slice_j = GetSlice(operands[j], i);
      if (!ShapeUtil::Equal(slice_0->shape(), slice_j->shape())) {
        return tensorflow::errors::FailedPrecondition(
            "Operation's operand should be the same shape");
      }
      if (slice_0->slice_starts() != slice_0->slice_starts()) {
        return tensorflow::errors::FailedPrecondition(
            "Operation's operand should be start from the same idx");
      }
      if (slice_0->slice_limits() != slice_0->slice_limits()) {
        return tensorflow::errors::FailedPrecondition(
            "Operation's operand should be end to the same idx");
      }
      // no match stride because the stride is always 1. we have checked.
    }
  }

  return operands;
}

StatusOr<std::vector<HloInstruction*>> SliceDelayer::GetSlicingUsers(
    const HloInstruction* inst, const std::vector<HloInstruction*>& operands) {
  std::vector<HloInstruction*> users;
  HloInstruction* operand0 = operands[0];
  int64 split_size = split_slices_.find(operand0)->second.size();
  for (int64 i = 0; i < split_size; ++i) {
    const HloInstruction* slice_0 = GetSlice(operand0, i);
    VLOG(0) << "GetUserSlice: " << slice_0->ToString();

    HloInstruction* selected_user = nullptr;
    for (HloInstruction* user : slice_0->users()) {
      VLOG(0) << "TryGetUser: " << user->ToString();
      // user should be the same operation and same operand count as inst
      if (user->opcode() != inst->opcode() ||
          user->operand_count() != inst->operand_count()) {
        continue;
      }

      // user's operand should be a split-slice of the true operand in order
      bool bContinue = false;
      for (int64 j = 0; j < operands.size(); ++j) {
        const HloInstruction* slice_j = GetSlice(operands[j], i);
        if (user->operand(j) != slice_j) {
          bContinue = true;
          break;
        }
        // TODO(xinan): if overlapping is supported, it needs to check
        // slice_start, slice_limit
      }
      if (bContinue) {
        continue;
      }

      // found the user
      selected_user = user;
      break;
    }

    // if no such user found, return false
    if (selected_user == nullptr) {
      return tensorflow::errors::FailedPrecondition(
          "Could not found valid user inorder ", i);
    }
    users.push_back(selected_user);
  }
  return users;
}

void SliceDelayer::GenerateNewOp(const std::vector<HloInstruction*>& operands,
    const std::vector<HloInstruction*>& users) {
  // generate new ops
  const Shape shape = operands[0]->shape();
  HloComputation* computation = users[0]->parent();
  auto new_op = computation->AddInstruction(
      users[0]->CloneWithNewOperands(shape, operands));
  VLOG(0) << "Add NewOp: " << new_op->ToString();

  std::vector<const HloInstruction*> slices;
  // replace the old users and their slice operands with new op and its slices,
  // new users
  for (int64 i = 0; i < users.size(); ++i) {
    HloInstruction* user = users[i];
    const HloInstruction* slice = user->operand(0);
    auto new_user = computation->AddInstruction(
        slice->CloneWithNewOperands(user->shape(), {new_op}));
    VLOG(0) << "Add NewSlice: " << new_user->ToString()
            << "\nReplace: " << user->ToString();
    user->ReplaceAllUsesWith(new_user);
    slices.push_back(new_user);
    to_remove_.insert(user);
  }  // end for users

  split_slices_.insert(std::make_pair(new_op, slices));
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
      if (slice_delayer.IsRemoved(instruction)) {
        VLOG(0) << "The instruction has been removed"
                << instruction->ToString();
        continue;
      }

      // Bundles split-slices and tries merge elementwise instruction with its
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
