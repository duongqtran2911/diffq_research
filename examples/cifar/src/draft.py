import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import json
from diffq import DiffQuantizer
from resnet import ResNet18
from sklearn.metrics import accuracy_score
import distrib
import numpy as np

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torchvision
import torchvision.transforms as transforms
logger = logging.getLogger(__name__)



print("\nTHIS IS NORMAL MODEL FROM NOW ON")

model0 = torch.load("/u/60/trand7/unix/ResearchProject/diffq/examples/cifar/outputs/exp_db.name=cifar10,model=resnet/checkpoint.th")
print(model0.keys())
# state0 = model0['state']
for key, value in model0['state'].items():
    print(key, value.shape)

print("\nTHIS IS DIFFQ MODEL FROM NOW ON")

model = torch.load("/u/60/trand7/unix/ResearchProject/diffq/examples/cifar/outputs/exp_db.name=cifar10,model=resnet,quant.group_size=4,quant.penalty=5/checkpoint.th")
# quantizer = DiffQuantizer(
#     model, group_size=8,
#     min_size=0.01,
#     min_bits=2,
#     init_bits=8,
#     max_bits=15,
#     exclude=[])
print(model.keys())
# state = model['state']
print("\nSTATE")
for key, value in model['state'].items():
    print(key, "   ||   ", value.shape)

print("\nQUANT_OPT")
for key, value in model['quant_opt'].items():
    print(key, "   ||   ", value)

# print("\nQUANTIZED_STATE")
# for key, value in model['quantized_state'].items():
#     print(key, value.shape)
# for i in model['quantized_state']:
#     print(i)

img_resize = 32
transform_test = transforms.Compose([
    transforms.Resize(img_resize),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = torchvision.datasets.CIFAR10(
    root="./data/cifar10", train=False, download=False, transform=transform_test)





