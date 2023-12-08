import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from diffq import DiffQuantizer
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

# model = torch.load("/u/60/trand7/unix/ResearchProject/diffq/examples/cifar/outputs/exp_db.name=cifar10,model=resnet,quant.group_size=16,quant.penalty=5/checkpoint.th")
# print(model.keys())
# state = model['state']
# state_keys = state.keys()
# print("state_keys: ", state_keys)
# for key in state_keys:
#     print(state_keys[key].shape)


def change_name(state_dict, use_diffq=False):
    new_state_dict = {}
    for key in state_dict.keys():
        if use_diffq and key.endswith("_diffq"):
            new_key = key.removesuffix("_diffq")
            new_state_dict[new_key] = state_dict[key]
        elif not use_diffq and not key.endswith("_diffq"):
            new_state_dict[key] = state_dict[key]
    return new_state_dict

device = 'cuda'
def load_model(checkpoint_path, use_diffq=False):
    model = ResNet18(num_classes=10)  # Adjust num_classes if needed
    checkpoint = torch.load(checkpoint_path)
    state_dict = change_name(checkpoint['state'], use_diffq=use_diffq)
    model.load_state_dict(state_dict)

    if torch.cuda.is_available():
        model.cuda()

    model.eval()  # Set the model to evaluation mode
    return model

def load_quantized_model(checkpoint_path, model_class=ResNet18, num_classes=10):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['state']

    # Initialize the model
    model = model_class(num_classes=num_classes)

    # Apply standard weights
    standard_state_dict = {k: v for k, v in state_dict.items() if not k.endswith('_diffq')}
    model.load_state_dict(standard_state_dict, strict=False)

    # Apply DiffQ quantization
    # This is a placeholder: replace with the actual logic of applying DiffQ quantization
    for name, param in model.named_parameters():
        diffq_param_name = name + "_diffq"
        if diffq_param_name in state_dict:
            diffq_param = state_dict[diffq_param_name]
            # Apply quantization logic
            # It typically involves using diffq_param to determine the quantization level
            # and then quantizing 'param' accordingly
            quantized_value = apply_quantization_logic(param, diffq_param)
            setattr(model, name, quantized_value)

    if torch.cuda.is_available():
        model.cuda()

    model.eval()  # Set the model to evaluation mode
    return model

def apply_quantization_logic(param, diffq_logit):
    # Define constants used in DiffQuantizer
    min_bits = 2
    max_bits = 15

    # Calculate the number of bits from diffq_logit
    t = torch.sigmoid(diffq_logit)
    bits = max_bits * t + (1 - t) * min_bits

    # Dequantization logic - this is an approximation
    scale = param.max() - param.min()
    unit = 1 / (2 ** bits - 1)
    dequantized_param = param * scale * unit

    return dequantized_param

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
        root="diffq/examples/cifar/data/cifar10/cifar-10-batches-py", train=False, download=True, transform=transform_test)
    num_classes = 10

    return testset, num_classes

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

if __name__ == "__main__":
    testset, num_classes = get_loader()

    # Load normally trained model
    model_normal = load_model("/u/60/trand7/unix/ResearchProject/diffq/examples/cifar/outputs/exp_db.name=cifar10,model=resnet/checkpoint.th", 
                                use_diffq=False)
    # Load DiffQ-trained model
    model_diffq = load_model("/u/60/trand7/unix/ResearchProject/diffq/examples/cifar/outputs/exp_db.name=cifar10,model=resnet,quant.group_size=8,quant.penalty=5/checkpoint.th", 
                                use_diffq=True)

    # Example of running inference
    # You need to define tt_loader or adjust as per your data loading logic
    tt_loader = distrib.loader(testset, batch_size=128, num_workers=1)

    # Test the normally trained model
    # pred_normal, valid_labels_normal = test_fn(tt_loader, model_normal, device)
    # acc_normal = get_score(valid_labels_normal, pred_normal.argmax(1))
    # print("Accuracy of normally trained model:", acc_normal)

    # Test the DiffQ-trained model
    pred_diffq, valid_labels_diffq = test_fn(tt_loader, model_diffq, device)
    acc_diffq = get_score(valid_labels_diffq, pred_diffq.argmax(1))
    print("Accuracy of DiffQ-trained model:", acc_diffq)


