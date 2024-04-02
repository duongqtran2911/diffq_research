import torch
import torchvision
from torchvision import datasets, transforms
from resnet import ResNet18  # Ensure this matches your model architecture
from torch.utils.data import DataLoader
from diffq import DiffQuantizer
import numpy as np
import diffq
from sklearn.metrics import accuracy_score

# Load the trained and quantized model
# model = ResNet18(num_classes=10)
# quantized_state = torch.load("/u/60/trand7/unix/ResearchProject/diffq/examples/cifar/outputs/exp_db.name=cifar10,model=resnet,quant.group_size=4,quant.penalty=5/checkpoint.th")
# diffq.restore_quantized_state(model, quantized_state)
# model.eval()

def load_model(checkpoint_path, num_classes):
    # Initialize the model
    model = ResNet18(num_classes=num_classes)
    if torch.cuda.is_available():
        model.cuda()
    
    # Load the saved state
    checkpoint = torch.load(checkpoint_path)
    state_dict = {key: value for key, value in checkpoint['state'].items() if not key.endswith('_diffq')}
    model.load_state_dict(state_dict)
    
    model.eval()  # Set the model to inference mode
    return model

def load_test_dataset(batch_size=64):
    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1)
    return testloader

def infer(model, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%')

if __name__ == "__main__":
    checkpoint_path = '/u/60/trand7/unix/ResearchProject/diffq/examples/cifar/outputs/exp_db.name=cifar10,model=resnet,quant.group_size=4,quant.penalty=5/checkpoint.th'
    num_classes = 10  # CIFAR10 has 10 classes
    batch_size = 100

    model = load_model(checkpoint_path, num_classes)
    testloader = load_test_dataset(batch_size)
    infer(model, testloader)
