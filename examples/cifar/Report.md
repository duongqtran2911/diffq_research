# Report:

## Task ongoing:
1. Re-training models with Differential Quantization:

+ can we quantize all models reliably using DiffQ?

+ any reduction in model size or latency?

+ can we reproduce the results from table 1 regarding model size & accuracy? (mobileNet?)

+ any difference in accuracy?

+ fixed group_size = 4:

    * Wide ResNet <br>
    * MobileNet <br>
    * DenseNet <br>
    * DLA Simple <br>
    * LeNet <br>
    * DLA <br>
    * mobilenetv2 <br>

+ why not working for 16 or 8?



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

- Unexpected error when training at Epoch 157 for dla
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
epoch 140, group_size=8
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


- Unexpected error when training at Epoch 36 for mobilenetv2
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
epoch 13, group_size=8
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




### 1. densenet
#### a. group_size = 4, penalty = 5
=> stopped at epoch 58
#### b. group_size = 8, penalty = 5
=> success
#### c. group_size = 16, penalty = 5
=> success


### 2. dla_simple
#### a. group_size = 4, penalty = 5
=> stopped at epoch 193
#### b. group_size = 8, penalty = 5
=> success
#### c. group_size = 16, penalty = 5
=> stopped at epoch 146


### 3. dla
#### a. group_size = 4, penalty = 5
=> stopped at epoch 131
#### b. group_size = 8, penalty = 5
=> stopped at epoch 139
#### c. group_size = 16, penalty = 5
=> stopped at epoch 156


### 4. efficientnet
#### a. group_size = 4, penalty = 5
=> stopped at epoch 1
#### b. group_size = 8, penalty = 5
=> stopped at epoch 1
#### c. group_size = 16, penalty = 5
=> stopped at epoch 1


### 5. lenet
#### a. group_size = 4, penalty = 5
=> success
#### b. group_size = 8, penalty = 5
=> success
#### c. group_size = 16, penalty = 5
=> success


### 6. mobilenet
#### a. group_size = 4, penalty = 5
=> stopped at epoch 15
#### b. group_size = 8, penalty = 5
=> stopped at epoch 16
#### c. group_size = 16, penalty = 5
=> stopped at epoch 16


### 7. mobilenetv2
#### a. group_size = 8, penalty = 5
=> stopped at epoch 13
#### b. group_size = 16, penalty = 5
=> stopped at epoch 35


### 8. resnet
#### a. group_size = 4, penalty = 5
=> success
#### b. group_size = 8, penalty = 5
=> success
#### c. group_size = 16, penalty = 5
=> success


### 9. resnet20
#### a. group_size = 8, penalty = 5
=> success


### 10. senet
#### a. group_size = 4, penalty = 5
=> stopped at epoch 7
#### b. group_size = 8, penalty = 5
=> 
#### c. group_size = 16, penalty = 5
=> 


### 11. w_resnet
#### a. group_size = 4, penalty = 5
=> success
#### b. group_size = 8, penalty = 5
=> success
#### c. group_size = 16, penalty = 5
=> success




## Ideas:

### 1. reducing learning rate

### 2. check where the NaN happens




densenet 8 - DONE (0.05)
0: dla_simple 4 -> epoch 137
1: densenet 4 -> epoch 67
2: lenet 8 -> DONE (0.05)
    dla_simple 16 -> epoch 149
3: senet 4


(lr=0.01)
0 resnet 4
1 resnet 8
2 resnet 16
3 resnet20 4
4 resnet20 16
5 w_resnet 16
6 resnet20 8
7 senet 8
8 senet 16

##########
Results (compressed):

densenet:
4 - 90.36
8 - 91.45
16 - 89.32

32/64(?) - og acc

dla_simple:
4 - 92.15
8 - 91.78
16 - 91.29
32/64(?) - og acc

lenet:
4 - 71.04
8 - 75.32
16 - 70.70
32/64(?) - og acc

resnet:
4 - 
8 - 
16 - 
32/64(?) - og acc

resnet20:
4 - 
8 - 
16 - 
32/64(?) - og acc

senet:
4 - 92.15
32/64(?) - og acc

w_resnet:
4 - 93.96
8 - 93.62
16 - 
32/64(?) - og acc

- accuracy vs model size
- accuracy vs quantization precision
- lr: training cost? 
  smaller lr -> stop training after loss saturated

=> why are these different for different models? 

eg. model size or accuracy constraints -> select the best model -> different trade-offs btw models -> different compressed model would be optimal


or

does one model has the best tradeoff all the time? -> why this model is special?



- compare the before vs after compression accuracy too 






