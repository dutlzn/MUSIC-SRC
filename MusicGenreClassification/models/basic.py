import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv2d(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, **kwargs):
        super().__init__()

        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          bias=False, **kwargs))

    
class InceptionLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super().__init__()

        self.branch1x1 = BasicConv2d(
            num_input_features, growth_rate, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(
            num_input_features, growth_rate * bn_size, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(
            growth_rate * bn_size, growth_rate, kernel_size=3, padding=1)

        self.branch5x5_1 = BasicConv2d(
            num_input_features, growth_rate * bn_size, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(
            growth_rate * bn_size, growth_rate, kernel_size=5, padding=2)

        self.mixed = BasicConv2d(
            3 * growth_rate, growth_rate, kernel_size=1)
        self.drop_rate = drop_rate

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        outputs = [branch1x1, branch3x3, branch5x5]
        new_features = self.mixed(torch.cat(outputs, dim=1))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return torch.cat([x, new_features], dim=1)

class DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super().__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super().forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], dim=1)
    

class DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features,
                 growth_rate, bn_size, drop_rate):
        super().__init__()

        for i in range(num_layers):
            layer = InceptionLayer(num_input_features + i * growth_rate,
                                   growth_rate, bn_size, drop_rate)
            self.add_module('inceptionlayer%d' % i, layer)

            
class NativeDenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features,
                 growth_rate, bn_size, drop_rate):
        super().__init__()

        for i in range(num_layers):
            layer = DenseLayer(num_input_features + i * growth_rate,
                               growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % i, layer)

            
class Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super().__init__()

        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(
            num_input_features, num_output_features, kernel_size=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x
