import torch
import torchvision
from torchvision import datasets, transforms
from diffq import DiffQuantizer
import numpy as np
import diffq
from sklearn.metrics import accuracy_score
from torch.utils.data import random_split

from mobilenet import MobileNet
from resnet import ResNet18
from resnet20 import resnet20
from wide_resnet import Wide_ResNet

from densenet import DenseNet
from dla import DLA
from dla_simple import SimpleDLA
from dpn import DPN
from efficientnet import EfficientNet
from googlenet import GoogLeNet
from lenet import LeNet
from mobilenetv2 import MobileNetV2
from pnasnet import PNASNet
from preact_resnet import PreActResNet
from regnet import RegNet
from resnext import ResNeXt
from senet import SENet18
from shufflenet import ShuffleNet
from shufflenetv2 import ShuffleNetV2
from vgg import VGG

import sys
sys.path.append("/u/60/trand7/unix/ResearchProject/diffq/examples/cifar")

# Load the trained and quantized model
model = LeNet(num_classes=10)
# path = 
path = "/u/60/trand7/unix/ResearchProject/diffq/examples/cifar/outputs/exp_db.name=cifar10,model=senet,quant.group_size=4,quant.penalty=5/quantized_model.pth"
model = torch.load(path)

# Prepare the data loader
img_resize = 32
transform_test = transforms.Compose([
    transforms.Resize(img_resize),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = torchvision.datasets.CIFAR10(root="./data/cifar10", train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=1)


# transform_train = transforms.Compose([
#             transforms.RandomCrop(32, padding=4),
#             transforms.Resize(img_resize),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#             ])
# trainset = torchvision.datasets.CIFAR10(
#             root="./data/cifar10", train=True, download=True, transform=transform_train)
# val_set_size = int(len(trainset) * 0.2)
# valset, trainset = random_split(trainset, [val_set_size, len(trainset) - val_set_size], generator=torch.Generator().manual_seed(0))
# val_loader = torch.utils.data.DataLoader(valset, batch_size=128, shuffle=False, num_workers=1)


# Inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# with torch.no_grad():
#     preds, valid_labels = [], []
#     for images, labels in val_loader:
#         images, labels = images.to(device), labels.to(device)
#         with torch.no_grad():
#             outputs = model(images)        
#         valid_labels.append(labels.cpu().numpy())
#         preds.append(outputs.softmax(1).cpu().numpy())
#     predictions = np.concatenate(preds)
#     valid_labels = np.concatenate(valid_labels)

# acc = 100 * accuracy_score(valid_labels, predictions.argmax(1))


with torch.no_grad():
    total = 0
    correct = 0
    for images, targets in test_loader:
        images, targets = images.to(device), targets.to(device)
        with torch.no_grad():
            outputs = model(images)
        
        print("new inference method")
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    total_acc = 100. * (correct/total)

world_size=1
def average(metrics, count=1.):
    if world_size == 1:
        return metrics
    tensor = torch.tensor(list(metrics) + [1], device='cuda', dtype=torch.float32)
    tensor *= count
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return (tensor[:-1] / tensor[-1]).cpu().numpy().tolist()

acc = average([total_acc], total)[0]

print(f'Accuracy of the model on the test images: {acc:.2f}')

