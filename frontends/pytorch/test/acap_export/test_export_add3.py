# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.



import torch
import torch_mlir

# RUN: %PYTHON %s | npcomp-opt | FileCheck %s

t0 = torch.randn((1,2,3,4))
t1 = torch.randn((1,2,3,4))
t2 = torch.randn((1,2,3,4))

mb = torch_mlir.ModuleBuilder()
with mb.capture_function("add3", [t0, t1, t2]) as f:
  t3 = t0 + t1 + t2
  f.returns([t3])
# NOTE: Assertions have been autogenerated by utils/generate-test-checks.py
# CHECK-LABEL:   func @add3(
# CHECK-SAME:               %[[VAL_0:.*]]: !torch.tensor<[1,2,3,4],f32>, %[[VAL_1:.*]]: !torch.tensor<[1,2,3,4],f32>,
# CHECK-SAME:               %[[VAL_2:.*]]: !torch.tensor<[1,2,3,4],f32>) -> !torch.tensor<[1,2,3,4],f32> {
# CHECK:           %[[VAL_3:.*]] = torch.constant.int 1 : i64
# CHECK:           %[[VAL_4:.*]] = torch.constant.int 1 : i64
# CHECK:           %[[VAL_5:.*]] = torch.tensor(dense<0.000000e+00> : tensor<1x2x3x4xf32>) : !torch.tensor<[1,2,3,4],f32>
# CHECK:           %[[VAL_6:.*]] = torch.operator "aten.add.out"(%[[VAL_0]], %[[VAL_1]], %[[VAL_3]], %[[VAL_5]]) : (!torch.tensor<[1,2,3,4],f32>, !torch.tensor<[1,2,3,4],f32>, i64, !torch.tensor<[1,2,3,4],f32>) -> !torch.tensor<[1,2,3,4],f32>
# CHECK:           %[[VAL_7:.*]] = torch.tensor(dense<0.000000e+00> : tensor<1x2x3x4xf32>) : !torch.tensor<[1,2,3,4],f32>
# CHECK:           %[[VAL_8:.*]] = torch.operator "aten.add.out"(%[[VAL_6]], %[[VAL_2]], %[[VAL_4]], %[[VAL_7]]) : (!torch.tensor<[1,2,3,4],f32>, !torch.tensor<[1,2,3,4],f32>, i64, !torch.tensor<[1,2,3,4],f32>) -> !torch.tensor<[1,2,3,4],f32>
# CHECK:           return %[[VAL_8]] : !torch.tensor<[1,2,3,4],f32>
# CHECK:         }

print(mb.module)
