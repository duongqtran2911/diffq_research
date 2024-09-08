#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import socket
import numpy as np
import hydra
from torch.utils.data.sampler import SubsetRandomSampler


logger = logging.getLogger(__name__)

random_seed = 10

def run(args):
    from src import distrib
    from src import data
    from src import solver as slv
    from src.mobilenet import MobileNet
    from src.resnet import ResNet18
    from src.resnet20 import resnet20
    from src.wide_resnet import Wide_ResNet

    from src.densenet import DenseNet
    from src.dla import DLA
    from src.dla_simple import SimpleDLA
    from src.dpn import DPN
    from src.efficientnet import EfficientNet
    from src.googlenet import GoogLeNet
    from src.lenet import LeNet
    from src.mobilenetv2 import MobileNetV2
    from src.pnasnet import PNASNet
    from src.preact_resnet import PreActResNet
    from src.regnet import RegNet
    from src.resnext import ResNeXt
    from src.senet import SENet
    from src.shufflenet import ShuffleNet
    from src.shufflenetv2 import ShuffleNetV2
    from src.vgg import VGG


    from src.vit import ViT

    import torch
    import torch.nn as nn
    from diffq import DiffQuantizer, UniformQuantizer, LSQ

    logger.info("Running on host %s", socket.gethostname())
    distrib.init(args, args.rendezvous_file)
    # torch also initialize cuda seed if available
    torch.manual_seed(args.seed)

    # validate distributed training
    assert args.batch_size % distrib.world_size == 0
    args.batch_size //= distrib.world_size

    # setup data loaders

    trainset, valset, testset, num_classes = data.get_loader(args, model_name=args.model.lower())
    tr_loader = distrib.loader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = distrib.loader(
        valset, batch_size=args.batch_size, num_workers=args.num_workers)
    tt_loader = distrib.loader(
        testset, batch_size=args.batch_size, num_workers=args.num_workers)

    data = {"tr": tr_loader, "val": val_loader, "tt": tt_loader}
    

    # build the model
    if args.model.lower() == 'resnet':
        model = ResNet18(num_classes=num_classes)
    elif args.model.lower() == 'resnet20':
        model = resnet20(num_classes=num_classes)
    elif args.model.lower() == 'w_resnet':
        # WideResNet params
        depth = 28
        widen_factor = 10
        do = 0.3
        model = Wide_ResNet(depth=depth, widen_factor=widen_factor,
                            dropout_rate=do, num_classes=num_classes)
    elif args.model.lower() == 'mobilenet':
        model = MobileNet(num_classes=num_classes)

    elif args.model.lower() == 'densenet':
        model = DenseNet(num_classes=num_classes)
    elif args.model.lower() == 'dla_simple':
        model = SimpleDLA(num_classes=num_classes)
    elif args.model.lower() == 'dla':
        model = DLA(num_classes=num_classes)
    # elif args.model.lower() == 'dpn':
    #     model = DPN(num_classes=num_classes)
    elif args.model.lower() == 'efficientnet':
        cfg = {
            'num_blocks': [1, 2, 2, 3, 3, 4, 1],
            'expansion': [1, 6, 6, 6, 6, 6, 6],
            'out_channels': [16, 24, 40, 80, 112, 192, 320],
            'kernel_size': [3, 3, 5, 3, 5, 5, 3],
            'stride': [1, 2, 2, 2, 1, 2, 1],
            'dropout_rate': 0.2,
            'drop_connect_rate': 0.2,
        }
        model = EfficientNet(cfg=cfg, num_classes=num_classes)
    # elif args.model.lower() == 'googlenet':
    #     model = GoogLeNet(num_classes=num_classes)
    elif args.model.lower() == 'lenet':
        model = LeNet(num_classes=num_classes)
    elif args.model.lower() == 'mobilenetv2':
        model = MobileNetV2(num_classes=num_classes)
    # elif args.model.lower() == 'pnasnet':
    #     model = PNASNet(num_classes=num_classes)
    elif args.model.lower() == 'preact_resnet':
        model = PreActResNet(num_classes=num_classes)
    elif args.model.lower() == 'regnet':
        model = RegNet(num_classes=num_classes)
    elif args.model.lower() == 'resnext':
        model = ResNeXt(num_classes=num_classes)
    elif args.model.lower() == 'senet':
        model = SENet(num_classes=num_classes)
    # elif args.model.lower() == 'shufflenet':
    #     model = ShuffleNet(num_classes=num_classes)
    # elif args.model.lower() == 'shufflenetv2':
    #     model = ShuffleNetV2(num_classes=num_classes)
    # elif args.model.lower() == 'vgg':
    #     model = VGG(num_classes=num_classes)

    elif args.model.lower() == 'vit':
        model = ViT(
            image_size=32,
            patch_size=4,
            num_classes=num_classes,
            dim=512,
            depth=6,
            heads=8,
            mlp_dim=512,
            dropout=0.1,
            emb_dropout=0.1)
    elif args.model.lower() == 'vit_timm':
        import timm
        model = timm.create_model("vit_base_patch16_224", pretrained=True)
        model.head = nn.Linear(model.head.in_features, num_classes)
    else:
        print('Arch not supported.')
        os._exit(1)

    logger.debug(model)
    model_size = sum(p.numel() for p in model.parameters()) * 4 / 2**20
    logger.info('Size: %.1f MB', model_size)

    if torch.cuda.is_available():
        model.cuda()

    params = model.parameters()
    # optimizer
    if args.optim == "adam":
        optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0.9, args.beta2))
    elif args.optim == "adamw":
        optimizer = torch.optim.AdamW(params, lr=args.lr,
                                      betas=(0.9, args.beta2), weight_decay=args.w_decay)
    elif args.optim == "rmsprop":
        optimizer = torch.optim.RMSprop(
            params, lr=args.lr, weight_decay=args.w_decay,
            momentum=args.momentum, alpha=args.alpha)
    elif args.optim == "sgd":
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.w_decay, nesterov=args.nesterov)
    else:
        logger.fatal('Invalid optimizer %s', args.optim)
        os._exit(1)

    if args.quant.penalty:
        quantizer = DiffQuantizer(
            model, group_size=args.quant.group_size,
            min_size=args.quant.min_size,
            min_bits=args.quant.min_bits,
            init_bits=args.quant.init_bits,
            max_bits=args.quant.max_bits,
            exclude=args.quant.exclude)
        if args.quant.adam:
            quantizer.opt = torch.optim.Adam([{"params": []}])
            quantizer.setup_optimizer(quantizer.opt, lr=args.quant.lr)
        else:
            quantizer.setup_optimizer(optimizer, lr=args.quant.lr)
    elif args.quant.lsq:
        quantizer = LSQ(
            model, bits=args.quant.bits, min_size=args.quant.min_size,
            exclude=args.quant.exclude)
        quantizer.setup_optimizer(optimizer)
    elif args.quant.bits:
        quantizer = UniformQuantizer(
            model, min_size=args.quant.min_size,
            bits=args.quant.bits, qat=args.quant.qat, exclude=args.quant.exclude)
    else:
        quantizer = None

    criterion = torch.nn.CrossEntropyLoss()

    # Construct Solver
    solver = slv.Solver(data, model, criterion, optimizer, quantizer, args, model_size)
    solver.train()
    print("training finished")
    print(f"Model is {quantizer.true_model_size()} MB")
    print(f"Quantized model is {quantizer.compressed_model_size()} MB")
    print("test size by quantizer")
    torch.save(model, "quantized_model.pth")

    # # Construct Solver
    # solver = slv.Solver(data, model, criterion, optimizer, quantizer, args, model_size)
    # solver.train()
    # print("training finished")
    # print(f"Model is {quantizer.true_model_size():.1f} MB")
    # print("test size by quantizer")
    # torch.save(model, 'saved_model.pth')
    
    # # Load the saved quantized model and test it
    # # model = ResNet18(num_classes=10)
    # model = torch.load('saved_model.pth')
    # device = torch.device('cuda')
    # model.to(device)
    # model.eval()
    
    # # Test the model
    # with torch.no_grad():
    #     total = 0
    #     correct = 0
    #     for images, targets in tt_loader:
    #         images, targets = images.to(device), targets.to(device)
    #         with torch.no_grad():
    #             outputs = model(images)
            
    #         print("new inference method")
    #         _, predicted = outputs.max(1)
    #         total += targets.size(0)
    #         correct += predicted.eq(targets).sum().item()
    #     total_acc = 100. * (correct/total)

    # world_size=1
    # def average(metrics, count=1.):
    #     if world_size == 1:
    #         return metrics
    #     tensor = torch.tensor(list(metrics) + [1], device='cuda', dtype=torch.float32)
    #     tensor *= count
    #     torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    #     return (tensor[:-1] / tensor[-1]).cpu().numpy().tolist()

    # acc = average([total_acc], total)[0]

    # print(f'Accuracy of the model on the test images: {acc:.2f}')


def _main(args):
    global __file__
    __file__ = hydra.utils.to_absolute_path(__file__)
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("diffq_cifar").setLevel(logging.DEBUG)

    # Updating paths in config
    if args.continue_from:
        args.continue_from = os.path.join(
            os.getcwd(), "..", args.continue_from, args.checkpoint_file)
    args.db.root = hydra.utils.to_absolute_path(args.db.root + '/' + args.db.name)
    logger.info("For logs, checkpoints and samples check %s", os.getcwd())
    logger.debug(args)
    run(args)


@hydra.main(config_path="conf", config_name='config.yaml')
def main(args):
    try:
        if args.ddp and args.rank is None:
            from src.executor import start_ddp_workers
            start_ddp_workers(args)
            return
        _main(args)
    except Exception:
        logger.exception("Some error happened")
        os._exit(1)  # a bit dangerous for the logs, but Hydra intercepts exit code
        # fixed in beta but I could not get the beta to work


if __name__ == "__main__":
    main()
