import torch
from torch.nn import functional as F
import diffq
from diffq import DiffQuantizer
from resnet import ResNet18

model = ResNet18(num_classes=10)
optim =   # The optimizer must be created before the quantizer
quantizer = DiffQuantizer(model)
quantizer.setup_optimizer(optim)

# Distributed data parallel must be created after DiffQuantizer!
dmodel = torch.distributed.DistributedDataParallel(...)

penalty = 1e-3
model.train()  # call model.eval() on eval to automatically use true quantized weights.
for batch in loader:
    ...
    optim.zero_grad()

    # The `penalty` parameter here will control the tradeoff between model size and model accuracy.
    loss = F.mse_loss(x, y) + penalty * quantizer.model_size()
    optim.step()

# To get the true model size with when doing proper bit packing.
print(f"Model is {quantizer.true_model_size():.1f} MB")

# When you want to dump your final model:
torch.save(quantizer.get_quantized_state(), "some_file.th")

# You can later load back the model with
model = MyModel()
diffq.restore_quantized_state(model, torch.load("some_file.th"))

# For DiffQ models, we support exporting the model to Torscript with optimal storage.
# Once loaded, the model will be stored in fp32 in memory (int8 support coming up).
from diffq.ts_export import export
export(quantizer, 'quantized.ts')