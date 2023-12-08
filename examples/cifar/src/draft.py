import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import json
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
state0 = model0['state']
for key, value in model0['state'].items():
    print(key, value.shape)

print("\nTHIS IS DIFFQ MODEL FROM NOW ON")

model = torch.load("/u/60/trand7/unix/ResearchProject/diffq/examples/cifar/outputs/exp_db.name=cifar10,model=resnet,quant.group_size=16,quant.penalty=5/checkpoint.th")
print(model.keys())
state = model['state']
for key, value in model['state'].items():
    print(key, value.shape)

