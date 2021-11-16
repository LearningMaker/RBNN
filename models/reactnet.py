import torch
import torch.nn as nn
import torch.nn.functional as F
from .reconfigurable_conv2d import ReconfigurableConv2d
from .reconfigurable_linear import ReconfigurableLinear
from .seeded_batchnorm import SeededBatchNorm2d

__all__ = ['reactnet']


def conv1x1(in_planes, out_planes, stride=1, seed=0):
    """1x1 convolution"""
    return ReconfigurableConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, seed=seed)


class BinaryActivation(nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x * x + 2 * x) * (1 - mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x * x + 2 * x) * (1 - mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1 - mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3

        return out


class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1, out_chn, 1, 1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out


class HardBinaryConv(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1, seed=0):
        super(HardBinaryConv, self).__init__()
        self.stride = stride
        self.padding = padding
        self.number_of_weights = in_chn * out_chn * kernel_size * kernel_size
        self.shape = (out_chn, in_chn, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.rand(self.number_of_weights) * 0.001, requires_grad=True)

        torch.manual_seed(seed)
        indices_permutation = torch.randperm(self.number_of_weights)
        if torch.cuda.is_available():
            self.indices_permutation = indices_permutation.long().cuda()
        else:
            self.indices_permutation = indices_permutation.long()
        setattr(self, 'indices_permutation_seed_' + str(seed), self.indices_permutation)

    def forward(self, x):
        real_weights = self.weight[self.indices_permutation].view(self.shape)
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights), dim=3, keepdim=True), dim=2, keepdim=True),
                                    dim=1, keepdim=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding)
        return y

    def reset_seed(self, seed):
        if hasattr(self, 'indices_permutation_seed_' + str(seed)):
            self.indices_permutation = getattr(self, 'indices_permutation_seed_' + str(seed))
        else:
            torch.manual_seed(seed)
            indices_permutation = torch.randperm(self.number_of_weights)
            if torch.cuda.is_available():
                self.indices_permutation = indices_permutation.long().cuda()
            else:
                self.indices_permutation = indices_permutation.long()
            setattr(self, 'indices_permutation_seed_' + str(seed), self.indices_permutation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, seed=0):
        super(BasicBlock, self).__init__()

        self.move0 = LearnableBias(inplanes)
        self.binary_activation = BinaryActivation()
        self.binary_conv = HardBinaryConv(inplanes, planes, stride=stride, seed=seed)
        self.bn1 = SeededBatchNorm2d(planes, seed=seed)
        self.move1 = LearnableBias(planes)
        self.prelu = nn.PReLU(planes)
        self.move2 = LearnableBias(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.move0(x)
        out = self.binary_activation(out)
        out = self.binary_conv(out)
        out = self.bn1(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.move1(out)
        out = self.prelu(out)
        out = self.move2(out)

        return out

    def reset_seed(self, seed):
        self.binary_conv.reset_seed(seed)
        self.bn1.reset_seed(seed)
        seed = seed + 1
        if self.downsample is not None:
            self.downsample[1].reset_seed(seed)
            self.downsample[2].reset_seed(seed)


class Net(nn.Module):

    def __init__(self, block, layers, num_classes=10, seed=0, seed_list=None):
        super(Net, self).__init__()
        if seed_list is None:
            seed_list = [seed, seed + 111, seed + 222, seed + 333, seed + 444]

        self.inplanes = 64
        self.conv1 = ReconfigurableConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False, seed=seed_list[0])
        self.bn1 = SeededBatchNorm2d(64, seed=seed_list[0])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], seed=seed_list[1])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, seed=seed_list[2])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, seed=seed_list[3])
        self.layer4 = lambda x: x
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = ReconfigurableLinear(256 * block.expansion, num_classes, seed=seed_list[4])

        self.regime = {
            0: {'optimizer': 'Adam', 'betas': (0.9, 0.999), 'lr': 5e-3},
            40: {'lr': 1e-3},
            80: {'lr': 5e-4},
            100: {'lr': 1e-4},
            120: {'lr': 5e-5},
            140: {'lr': 1e-5}
        }

    def _make_layer(self, block, planes, blocks, stride=1, seed=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=stride),
                conv1x1(self.inplanes, planes * block.expansion, seed=seed),
                SeededBatchNorm2d(planes * block.expansion, seed=seed),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, seed=seed))
        seed += 10
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, seed=seed))
            seed += 10

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def reset_seed(self, seed, seed_list=None):
        if seed_list is None:
            seed_list = [seed, seed + 111, seed + 222, seed + 333, seed + 444]
        self.conv1.reset_seed(seed_list[0])
        self.bn1.reset_seed(seed_list[0])
        seed_in = seed_list[1]
        for module in self.layer1._modules.values():
            module.reset_seed(seed_in)
            seed_in = seed_in + 10
        seed_in = seed_list[2]
        for module in self.layer2._modules.values():
            module.reset_seed(seed_in)
            seed_in = seed_in + 10
        seed_in = seed_list[3]
        for module in self.layer3._modules.values():
            module.reset_seed(seed_in)
            seed_in = seed_in + 10
        self.fc.reset_seed(seed_list[4])


def reactnet(**kwargs):
    num_classes, dataset = map(
        kwargs.get, ['num_classes', 'datasets'])
    return Net(BasicBlock, [4, 4, 4, 4], num_classes=num_classes)
