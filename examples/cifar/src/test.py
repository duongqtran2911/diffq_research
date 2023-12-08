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


def get_loader():
    
    img_resize = 32

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(img_resize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    transform_test = transforms.Compose([
        transforms.Resize(img_resize),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    # trainset = torchvision.datasets.CIFAR10(
    #     root="diffq/examples/cifar/data/cifar10/cifar-10-batches-py", train=True, download=False, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root="diffq/examples/cifar/data/cifar10/cifar-10-batches-py", train=False, download=False, transform=transform_test)
    num_classes = 10

    return testset, num_classes



device = 'cuda'

def load_model(checkpoint_path, quant=False):
    model = ResNet18(num_classes=10)  # CIFAR10 has 10 classes
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = {key: value for key, value in checkpoint['state'].items() if not key.endswith('_diffq')}
    model.load_state_dict(state_dict)

    if torch.cuda.is_available():
        model.cuda()

    model.eval()  # Set the model to evaluation mode
    return model


def test_fn(test_loader, model, device):
    model.eval()
    preds = []
    valid_labels = []
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        with torch.no_grad():
            outputs = model(images)
        valid_labels.append(labels.cpu().numpy())
        preds.append(outputs.softmax(1).to('cpu').numpy())
        torch.cuda.empty_cache()
    predictions = np.concatenate(preds)
    valid_labels = np.concatenate(valid_labels)
    return predictions, valid_labels

def get_score(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def change_name(state_dict, quant = True):
    new_state_dict = {}
    for key in state_dict.keys():
        if key.endswith("_diffq"):
            new_key = key.removesuffix("_diffq")
            new_state_dict[new_key] = state_dict[key]
    return new_state_dict


if __name__ == "__main__":
    # For the standard ResNet model
    testset, num_classes = get_loader()
    model_1 = load_model("/u/60/trand7/unix/ResearchProject/diffq/examples/cifar/outputs/exp_db.name=cifar10,model=resnet/checkpoint.th",
                        quant=True)
    tt_loader = distrib.loader(testset, batch_size=128, num_workers=1)
    pred, valid_labels = test_fn(tt_loader, model_1, device)
    acc = get_score(valid_labels, pred.argmax(1))
    print(acc)
    # For the quantized ResNet model
    quant_params = {
        'group_size': 8,
        'penalty': 5
    }


# resnet: 0.9503


