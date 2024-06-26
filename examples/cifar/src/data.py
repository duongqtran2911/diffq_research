# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torchvision
import torch
from torch.utils.data import random_split
import torchvision.transforms as transforms
logger = logging.getLogger(__name__)


def get_loader(args, model_name):
    if args.model == 'vit_timm':
        img_resize = 224
    else:
        img_resize = 32
    if args.db.name.lower() == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.Resize(img_resize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        transform_val = transforms.Compose([
            transforms.Resize(img_resize),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        transform_test = transforms.Compose([
            transforms.Resize(img_resize),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        trainset = torchvision.datasets.CIFAR10(
            root=args.db.root, train=True, download=True, transform=transform_train)
        val_set_size = int(len(trainset) * 0.2)
        valset, trainset = random_split(trainset, [val_set_size, len(trainset) - val_set_size],
                                generator=torch.Generator().manual_seed(0))
        testset = torchvision.datasets.CIFAR10(
            root=args.db.root, train=False, download=True, transform=transform_test)
        num_classes = 10

    elif args.db.name.lower() == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.Resize(img_resize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2671)),
            ])
        transform_val = transforms.Compose([
            transforms.Resize(img_resize),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2671)),
            ])
        transform_test = transforms.Compose([
            transforms.Resize(img_resize),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2671)),
            ])
        trainset = torchvision.datasets.CIFAR100(
            root=args.db.root, train=True, download=True, transform=transform_train)
        val_set_size = int(len(trainset) * 0.2)
        valset, trainset = random_split(trainset, [val_set_size, len(trainset) - val_set_size],
                                generator=torch.Generator().manual_seed(0))
        testset = torchvision.datasets.CIFAR100(
            root=args.db.root, train=False, download=True, transform=transform_test)
        num_classes = 100
    else:
        logger.error("DB not supported.")
        assert False

    return trainset, valset, testset, num_classes
