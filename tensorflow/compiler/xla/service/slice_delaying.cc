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

namespace xla {

namespace {

using ConstInstructionVector = std::vector<const HloInstruction*>;

class SliceDelayer {
 public:
  bool Run(HloComputation* computation);

 private:
  bool BundleSlices(const HloInstruction* inst);

  bool IsSplitSlice(const HloInstruction* operand, const HloInstruction* slice);

  const HloInstruction* GetSlice(const HloInstruction* operand, int64 i);

  bool CheckPattern(const std::vector< HloInstruction*>& operands,
      const std::vector<HloInstruction*>& users);

  void GenerateNewOp(const std::vector<HloInstruction*>& operands,
      const std::vector<HloInstruction*>& users);

  bool Merge(const HloInstruction* inst);

  absl::flat_hash_map<const HloInstruction*,
                      ConstInstructionVector> split_slices_;

  std::set<HloInstruction*> to_remove_;
};

bool IsUnstridedSlice(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kSlice &&
      absl::c_all_of(inst->slice_strides(),
                     [](int64 stride) { return stride == 1; });
}

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

bool CheckContinuous(const ConstInstructionVector& slices, int64 dim) {
  int64 k = slices[0]->slice_limits(dim);
  for (int64 i = 1; i < slices.size(); ++i) {
    if (slices[i]->slice_starts(dim) != k) {
      return false;
    }
    k = slices[i]->slice_limits(dim);
  }
  return true;
}

bool IsSplitSlices(const HloInstruction* inst, int64 dim,
    const ConstInstructionVector& slices) {
  if (!CheckContinuous(slices, dim))
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

bool SliceDelayer::BundleSlices(const HloInstruction* inst) {
  auto iter = split_slices_.find(inst);
  if (iter != split_slices_.end())
    return false;
  std::map<int64, ConstInstructionVector> dim_slices;
  for (auto user : inst->users()) {
    if (!IsUnstridedSlice(user))
      continue;
    int64 dim = GetSplitedDim(inst, user);
    if (dim == -1)
      continue;
    dim_slices[dim].push_back(user);
  }

  for (auto pair : dim_slices) {
    int64 dim = pair.first;
    ConstInstructionVector& vec = pair.second;
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

bool SliceDelayer::IsSplitSlice(
    const HloInstruction* operand, const HloInstruction* slice) {
  auto iter_m = split_slices_.find(operand);
  if (split_slices_.end() == iter_m)
    return false;
  auto iter_s = std::find(iter_m->second.begin(), iter_m->second.end(), slice);
  if (iter_m->second.end() == iter_s)
    return false;
  VLOG(1) << "Split: " << operand->ToString() << "\nSlice: "
          << slice->ToString();
  return true;
}

const HloInstruction* SliceDelayer::GetSlice(
    const HloInstruction* operand, int64 i) {
  return split_slices_.find(operand)->second[i];
}

bool SliceDelayer::CheckPattern(const std::vector< HloInstruction*>& operands,
    const std::vector<HloInstruction*>& users) {
  for (int64 i = 0; i < operands.size(); ++i) {
    VLOG(1) << "CheckPattern: ops[" << i << "]: " << operands[i]->ToString();
  }
  for (int64 i = 0; i < users.size(); ++i) {
    VLOG(1) << "CheckPattern: users[" << i << "]: " << users[i]->ToString();
  }
  const Shape shape = operands[0]->shape();
  int64 split_size = users.size();
  for (const HloInstruction* operand : operands) {
    // Only support element-wise now
    if (!ShapeUtil::Equal(operand->shape(), shape))
      return false;
    if (split_slices_.find(operand)->second.size() != split_size)
      return false;
  }
  const HloInstruction* split_op = nullptr;
  HloOpcode op = users[0]->opcode();
  int64 operand_num = operands.size();
  for (int64 i = 0; i < split_size; ++i) {
    if (users[i]->opcode() != op)
      return false;
    if (users[i]->operand_count() != operand_num)
      return false;
    const Shape split_shape = users[i]->shape();
    for (int64 j = 0; j < operands.size(); ++j) {
      const HloInstruction* operand = operands[j];
      split_op = GetSlice(operand, i);
      // Only support element-wise now
      if (!ShapeUtil::Equal(split_op->shape(), split_shape))
        return false;
      // TODO(xinan): support more users
      if (split_op->user_count() > 1)
        return false;
      // Match user i operand j
      if (users[i]->operand(j) != split_op)
        return false;
    }
  }
  VLOG(1) << "CheckPattern: succeed";
  return true;
}

void SliceDelayer::GenerateNewOp(const std::vector<HloInstruction*>& operands,
    const std::vector<HloInstruction*>& users) {
  // generate new ops
  const Shape shape = operands[0]->shape();
  HloComputation* computation = users[0]->parent();
  auto new_op = computation->AddInstruction(
      users[0]->CloneWithNewOperands(shape, operands));
  VLOG(1) << "Add NewOp: " << new_op->ToString();

  ConstInstructionVector slices;
  for (int64 i = 0; i < users.size(); ++i) {
    HloInstruction* user = users[i];
    const HloInstruction* slice = user->operand(0);
    auto new_user = computation->AddInstruction(
        slice->CloneWithNewOperands(user->shape(), {new_op}));
    VLOG(1) << "Add NewSlice: " << new_user->ToString()
            << "\nReplace: " << user->ToString();
    user->ReplaceAllUsesWith(new_user);
    slices.push_back(new_user);
    to_remove_.insert(user);
  }

  split_slices_.insert(std::make_pair(new_op, slices));
}

bool SliceDelayer::Merge(const HloInstruction* inst) {
  // check operands
  std::vector<HloInstruction*> operands;
  for (HloInstruction* slice : inst->operands()) {
    if (!IsUnstridedSlice(slice))
      return false;
    HloInstruction* operand = slice->mutable_operand(0);
    if (!IsSplitSlice(operand, slice))
      return false;
    operands.push_back(operand);
  }
  for (int64 i = 0; i < operands.size(); ++i) {
    VLOG(1) << "CheckOperand: ops[" << i << "]: " << operands[i]->ToString();
  }

  // check users
  std::vector<HloInstruction*> users;
  for (int64 i = 0; i < split_slices_.find(operands[0])->second.size(); ++i) {
    const HloInstruction* slice = GetSlice(operands[0], i);
    VLOG(1) << "CheckUser: slice[" << i << "]: " << slice->ToString();
    // TODO(xinan): support multi users
    if (slice->user_count() > 1)
      return false;
    HloInstruction* user = slice->users()[0];
    VLOG(1) << "CheckUser: user[" << i << "]: " << user->ToString();
    if (user->opcode() != inst->opcode())
      return false;
    if (user->operand(0) != slice)
      return false;
    users.push_back(user);
  }
  for (int64 i = 0; i < users.size(); ++i) {
    VLOG(1) << "CheckUser: users[" << i << "]: " << users[i]->ToString();
  }

  if (!CheckPattern(operands, users))
    return false;

  GenerateNewOp(operands, users);
  return true;
}

bool SliceDelayer::Run(HloComputation* computation) {
  to_remove_.clear();
  split_slices_.clear();
  bool changed = false;

  for (HloInstruction* instruction : computation->MakeInstructionPostOrder()) {
    if (std::find(to_remove_.begin(), to_remove_.end(), instruction)
        != to_remove_.end()) {
      VLOG(1) << "The instruction has been removed" << instruction->ToString();
      continue;
    }
    if (instruction->opcode() == HloOpcode::kSlice) {
      VLOG(1) << "Bundle slice: " << instruction->ToString();
      BundleSlices(instruction->mutable_operand(0));
    } else if (instruction->IsElementwise()
               && instruction->operand_count() != 0) {
      // TODO(xinan): more other instructions
      VLOG(1) << "Merge inst: " << instruction->ToString();
      changed |= Merge(instruction);
    }
  }

  for (auto inst : to_remove_) {
    VLOG(1) << "Delete: " << inst->ToString();
    inst->parent()->RemoveInstruction(inst);
  }

  to_remove_.clear();
  split_slices_.clear();

  return changed;
}

StatusOr<bool> SliceDelaying::Run(HloModule* module) {
  VLOG(0) << "Run Pass: " << name();
  VLOG(1) << "before: " << name() << "\n" << module->ToString();
  SliceDelayer slice_delayer;
  bool changed = false;

  for (HloComputation* computation : module->computations()) {
    changed |= slice_delayer.Run(computation);
  }

  VLOG(1) << "after: " << name() << "\n" <<  module->ToString();
  return changed;
}

}  // namespace xla
