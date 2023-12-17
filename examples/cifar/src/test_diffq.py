import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import json
from resnet import ResNet18
from sklearn.metrics import accuracy_score
import numpy as np
import logging
import torchvision
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)

# def apply_quantization_logic(param, diffq_logit, group_size):
#     device = param.device
#     diffq_logit = diffq_logit.to(device) 
#     # Constants used in DiffQuantizer
#     min_bits = 2
#     max_bits = 15
#     # Calculate the number of bits from diffq_logit
#     t = torch.sigmoid(diffq_logit)
#     bits = max_bits * t + (1 - t) * min_bits
#     # Reshape bits to match the group size
#     num_groups = param.nelement() // group_size
#     bits = bits.view(num_groups, -1).mean(dim=1)
#     # Scale and quantize each group
#     reshaped_param = param.view(num_groups, -1)
#     min_vals, _ = reshaped_param.min(dim=1, keepdim=True)
#     max_vals, _ = reshaped_param.max(dim=1, keepdim=True)
#     scale = (max_vals - min_vals) / (2 ** bits - 1)
#     normalized_param = (reshaped_param - min_vals) / (max_vals - min_vals)
#     quantized_param = torch.round(normalized_param * scale)
#     # Dequantize
#     dequantized_param = (quantized_param / scale) * (max_vals - min_vals) + min_vals
#     # Reshape back to original shape
#     dequantized_param = dequantized_param.view_as(param)
#     return dequantized_param

# def apply_quantization_logic(param, diffq_logit, group_size):
#     device = param.device
#     diffq_logit = diffq_logit.to(device) 

#     # Constants used in DiffQuantizer
#     min_bits = 2
#     max_bits = 15

#     # Calculate the number of groups based on group size
#     num_groups = param.numel() // group_size

#     # Reshape diffq_logit to match the number of groups
#     reshaped_diffq_logit = diffq_logit[:num_groups].view(-1)

#     # Calculate the number of bits for each group
#     t = torch.sigmoid(reshaped_diffq_logit)
#     bits = max_bits * t + (1 - t) * min_bits

#     # Reshape the parameter tensor for group-wise processing
#     reshaped_param = param.view(-1, group_size)

#     # Quantization and dequantization logic for each group
#     min_vals = reshaped_param.min(dim=1, keepdim=True)[0]
#     max_vals = reshaped_param.max(dim=1, keepdim=True)[0]
#     scale = (max_vals - min_vals) / (2 ** bits.unsqueeze(1) - 1)

#     normalized_param = (reshaped_param - min_vals) / (max_vals - min_vals)
#     quantized_param = torch.round(normalized_param * scale)

#     # Dequantization logic
#     dequantized_param = (quantized_param / scale) * (max_vals - min_vals) + min_vals

#     # Reshape back to the original shape
#     dequantized_param = dequantized_param.view_as(param)

#     return dequantized_param

def apply_quantization_logic(param, diffq_logit, group_size):
    device = param.device
    diffq_logit = diffq_logit.to(device)

    # Constants used in DiffQuantizer
    min_bits = 2
    max_bits = 15

    # Reshape diffq_logit to match the parameter's group size
    num_groups = param.numel() // group_size
    reshaped_diffq_logit = diffq_logit[:num_groups].view(-1)

    # Calculate the number of bits for each group
    t = torch.sigmoid(reshaped_diffq_logit)
    bits = max_bits * t + (1 - t) * min_bits
    bits = bits.round().clamp(min_bits, max_bits)

    # Reshape the parameter tensor for group-wise processing
    reshaped_param = param.view(-1, group_size)

    # Apply quantization and dequantization for each group
    scale = (2 ** bits.unsqueeze(1) - 1)
    quantized_param = torch.round(reshaped_param * scale)
    dequantized_param = quantized_param / scale

    # Reshape back to the original shape
    dequantized_param = dequantized_param.view_as(param)

    return dequantized_param




def load_quantized_model(checkpoint_path, model_class=ResNet18, num_classes=10):
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['state']

    model = model_class(num_classes=num_classes)
    standard_state_dict = {k: v for k, v in state_dict.items() if not k.endswith('_diffq')}
    model.load_state_dict(standard_state_dict, strict=False)

    for name, param in model.named_parameters():
        diffq_param_name = name + "_diffq"
        if diffq_param_name in state_dict:
            diffq_param = state_dict[diffq_param_name]
            dequantized_value = apply_quantization_logic(param.data, diffq_param, 8)
            param.data = dequantized_value

    if torch.cuda.is_available():
        model.cuda()

    model.eval()
    return model


def get_loader():
    img_resize = 32
    transform_test = transforms.Compose([
        transforms.Resize(img_resize),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(
        root="./data/cifar10", train=False, download=True, transform=transform_test)
    return testset

def test_fn(test_loader, model, device):
    model.eval()
    preds, valid_labels = [], []
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(images)
        valid_labels.append(labels.cpu().numpy())
        preds.append(outputs.softmax(1).to('cpu').numpy())
    predictions = np.concatenate(preds)
    valid_labels = np.concatenate(valid_labels)
    return predictions, valid_labels

def get_score(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

if __name__ == "__main__":
    testset = get_loader()
    tt_loader = torch.utils.data.DataLoader(testset, batch_size=128, num_workers=1, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load DiffQ-trained model
    model_diffq = load_quantized_model("/u/60/trand7/unix/ResearchProject/diffq/examples/cifar/outputs/exp_db.name=cifar10,model=resnet,quant.group_size=8,quant.penalty=5/checkpoint.th", 
                                       ResNet18, 10)
    model_diffq.to(device)

    # Test the DiffQ-trained model
    pred_diffq, valid_labels_diffq = test_fn(tt_loader, model_diffq, device)
    acc_diffq = get_score(valid_labels_diffq, pred_diffq.argmax(1))
    print("Accuracy of DiffQ-trained model:", acc_diffq)
