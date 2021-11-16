#  

### ABOUT

### DEPENDENCIES
Our code is implemented and tested on PyTorch. Following packages are used by our code.
- `tensorflow-gpu==1.9.0`
- `torchvision==0.10.0`
- `numpy==1.21.2`

### RUN
Train RBNN on CIFAR10, Fashion-MNIST, SVHN, and the binary neural network model used is ReActNet
```python
python train.py --epochs 200 --datasets cifar10 fashionmnist svhn --model reactnet --seed 0 --save result
```
Evaluate RBNN's metrics on all tasks
```python
python eval.py --datasets cifar10 fashionmnist svhn --model reactnet --seed 0 --load result
```
