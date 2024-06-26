# Report:

## Task ongoing:
1. Re-training models with Differential Quantization:
    * Wide ResNet <br>
    * MobileNet <br>
    * DenseNet <br>
    * DLA Simple <br>
    * LeNet <br>
    * DLA <br>


## Difficulties/TODO:
1. Error in ongoing models:
- Unexpected error when training at Epoch 20 for mobilenet
```
[05-20 00:54:14][__main__][ERROR] - Some error happened
Traceback (most recent call last):
  File "/m/home/home6/60/trand7/unix/ResearchProject/diffq/examples/cifar/train.py", line 276, in main
    _main(args)
  File "/m/home/home6/60/trand7/unix/ResearchProject/diffq/examples/cifar/train.py", line 266, in _main
    run(args)
  File "/m/home/home6/60/trand7/unix/ResearchProject/diffq/examples/cifar/train.py", line 196, in run
    solver.train()
  File "/m/home/home6/60/trand7/unix/ResearchProject/diffq/examples/cifar/src/solver.py", line 141, in train
    train_loss, train_acc = self._run_one_epoch(epoch)
  File "/m/home/home6/60/trand7/unix/ResearchProject/diffq/examples/cifar/src/solver.py", line 215, in _run_one_epoch
    model_size = self.quantizer.model_size() if self.quantizer else 0
  File "/u/60/trand7/unix/.local/lib/python3.10/site-packages/diffq/diffq.py", line 224, in model_size
    bits_bits = math.ceil(math.log2(1 + (bits.max().round().item() - self.min_bits)))
ValueError: cannot convert float NaN to integer
```

- Unexpected error when training at Epoch 147 for dla_simple
```
Traceback (most recent call last):
  File "/m/home/home6/60/trand7/unix/ResearchProject/diffq/examples/cifar/train.py", line 270, in main
    _main(args)
  File "/m/home/home6/60/trand7/unix/ResearchProject/diffq/examples/cifar/train.py", line 260, in _main
    run(args)
  File "/m/home/home6/60/trand7/unix/ResearchProject/diffq/examples/cifar/train.py", line 196, in run
    solver.train()
  File "/m/home/home6/60/trand7/unix/ResearchProject/diffq/examples/cifar/src/solver.py", line 141, in train
    train_loss, train_acc = self._run_one_epoch(epoch)
  File "/m/home/home6/60/trand7/unix/ResearchProject/diffq/examples/cifar/src/solver.py", line 215, in _run_one_epoch
    model_size = self.quantizer.model_size() if self.quantizer else 0
  File "/u/60/trand7/unix/.local/lib/python3.10/site-packages/diffq/diffq.py", line 224, in model_size
    bits_bits = math.ceil(math.log2(1 + (bits.max().round().item() - self.min_bits)))
ValueError: cannot convert float NaN to integer
```





