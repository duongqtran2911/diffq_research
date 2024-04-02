import torch
import torchvision
from torch.nn import functional as F
import diffq
from diffq import DiffQuantizer
from resnet import ResNet18
import torchvision.transforms as transforms
import numpy as np
# from diffq import restore_quantized_state

model = ResNet18(num_classes=10)
diffq.restore_quantized_state(model, torch.load("/u/60/trand7/unix/ResearchProject/diffq/examples/cifar/outputs/exp_db.name=cifar10,model=resnet,quant.group_size=4,quant.penalty=5/checkpoint.th"))


optim = torch.optim.Aadam(betas=(0.9, 0.999)) # The optimizer must be created before the quantizer
quantizer = DiffQuantizer(model)
quantizer.setup_optimizer(optim)

# Distributed data parallel must be created after DiffQuantizer!
# dmodel = torch.distributed.DistributedDataParallel(...)

img_resize = 32
transform_test = transforms.Compose([
    transforms.Resize(img_resize),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = torchvision.datasets.CIFAR10(
    root="./data/cifar10", train=False, download=False, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

penalty = 1e-3
preds, valid_labels = [], []
model.eval()  # call model.eval() on eval to automatically use true quantized weights.
for batch in test_loader:
    images, labels = batch
    images, labels = images.to(device), labels.to(device)
    with torch.no_grad():
        outputs = model(images)
    valid_labels.append(labels.to(device).numpy())
    preds.append(outputs.softmax(1).to(device).numpy())
    predictions = np.concatenate(preds)
    valid_labels = np.concatenate(valid_labels)
    # optim.zero_grad()

    # # The `penalty` parameter here will control the tradeoff between model size and model accuracy.
    # loss = F.mse_loss(preds, valid_labels.argmax(1)) + penalty * quantizer.model_size()
    # optim.step()


# To get the true model size with when doing proper bit packing.
print(f"Model is {quantizer.true_model_size():.1f} MB")

# When you want to dump your final model:
# torch.save(quantizer.get_quantized_state(), "some_file.th")


# For DiffQ models, we support exporting the model to Torscript with optimal storage.
# Once loaded, the model will be stored in fp32 in memory (int8 support coming up).
# from diffq.ts_export import export
# export(quantizer, 'quantized.ts')


