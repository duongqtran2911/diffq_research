# DiffQ for CIFAR-10/100

## Requirements

You must first install `diffq`, then the requirements for this example. To do so, run **from the root of the repository**:
```bash
pip install .
cd examples/cifar
pip install -r requirements.txt
```

## Training

In order to train a model (without quantization) you can run

```
./train.py db.name={DATASET} model={MODEL}
```
with DATASET either `cifar10` or `cifar100` and model one of
`resnet` (ResNet 18), `mobilenet` (MobileNet), or `w_resnet` (Wide ResNet).
The datasets will be automatically downloaded in the `./data` folder, and
the checkpoints stored in the `./outputs` folder.

In order to train a model with DiffQ, you can run

To train with diffq, with a given model size penalty and group size.
```
./train.py db.name={DATASET} model={MODEL} quant.penalty={PENALTY} quant.group_size={GROUP_SIZE}
```

for instance:
```
./train.py db.name=cifar100 model=w_resnet quant.penalty=5 quant.group_size=16
```


## Inference

After training, models checkpoints are stored in `./outputs` folder.

Inference code is implemented in file `inference_ex2.py` and `test.py`.

To run an inference on a normally trained model, you should first change the path to the trained model checkpoints inside file `test.py`, then run these commands:

```
cd examples/cifar/src

python test.py
```

To run inference on a DiffQ-trained model (quantized model), you should first change the path to the trained model inside file `inference_ex2.py`, then run these commands:

```
cd examples/cifar/src

python inference_ex2.py
```
