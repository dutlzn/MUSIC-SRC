import os.path as osp
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import manual_seed


class BasicConv2d(nn.Sequential):
    def __init__(self, num_in_channels, num_out_channels, **kwargs):
        super().__init__()
        
        self.add_module('bn', nn.BatchNorm2d(num_in_channels, eps=1e-3))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_in_channels, num_out_channels, **kwargs))

    
class InceptionModule(nn.Module):
    def __init__(self, num_in_channels, num_out_channels):
        super().__init__()

        self.branch1x1 = BasicConv2d(num_in_channels, num_out_channels,
                                     kernel_size=1)

        self.branch5x5_1 = BasicConv2d(num_in_channels, num_out_channels,
                                       kernel_size=1)
        self.branch5x5_2 = BasicConv2d(num_out_channels, num_out_channels, 
                                       kernel_size=5, padding=2)

        self.branch3x3_1 = BasicConv2d(num_in_channels, num_out_channels,
                                       kernel_size=1)
        self.branch3x3_2 = BasicConv2d(num_out_channels, num_out_channels,
                                       kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(num_in_channels, num_out_channels,
                                       kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch_pool = F.max_pool2d(x, kernel_size=3,
                                   stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        return torch.cat(outputs, dim=1)


class DenseModule(nn.Module):
    def __init__(self, num_dense_modules, num_in_channels, num_out_channels):
        super().__init__()
        
        for i in range(num_dense_modules):
            layer = InceptionModule(num_in_channels, num_out_channels)
            self.add_module('dense_layer_%d' % i, layer)
            num_in_channels += 4 * num_out_channels

    
    def forward(self, x):
        for name, m in self.named_children():
            outputs = m(x)
            x = torch.cat([x, outputs], dim=1)
        return x


class TransitionModule(nn.Sequential):
    def __init__(self, num_in_channels, num_out_channels):
        super().__init__()
        
        self.add_module('bn', nn.BatchNorm2d(num_in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_in_channels, num_out_channels,
                                          kernel_size=1))
        self.add_module('avg_pool', nn.AvgPool2d(kernel_size=2, stride=2))
            

class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x
    
    
class ExtractLastHiddenStateFromLSTM(nn.Module):
    def forward(self, x):
        outputs, (hn, cn) = x
        n, *_ = hn.shape
        return hn[n-1]
    
    
class AverageHiddenStateFromLSTM(nn.Module):
    def forward(self, x):
        outputs, (hn, cn) = x
        return outputs.mean(1)

    
class Aggregate(nn.Module):
    def __init__(self, in_features, out_features, alpha=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.alpha = alpha

    def forward(self, x):
        num_samples, num_segments, num_features = x.shape
        hk = torch.zeros((num_samples, self.out_features),
                         dtype=x.dtype, device=x.device)
        alpha = self.alpha
        hiddens = []
        for k in range(num_segments):
            xk = torch.select(x, 1, k)
            hk = alpha * hk + (1 - alpha) * self.linear(xk)
            hiddens.append(hk)
        hiddens = torch.stack(hiddens, dim=1)
        return hk, hiddens

    
class DenseInception(nn.Module):
    def __init__(self, num_in_channels, num_out_channels, num_dense_modules, num_classes=10):
        manual_seed()
        super().__init__()
        
        self.preprocessing = nn.Sequential(
            nn.Conv2d(num_in_channels, num_out_channels,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(num_out_channels, eps=0.001),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(4, 1)))
        num_in_channels = num_out_channels

        self.features = nn.Sequential(
            DenseModule(num_dense_modules, num_in_channels,
                        num_out_channels),
            TransitionModule(num_in_channels + num_dense_modules * 4 * num_out_channels,
                             num_out_channels))
        num_in_channels = num_out_channels
        
        self.classifier = nn.Sequential(
            nn.BatchNorm2d(num_in_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Linear(num_in_channels, num_classes))

        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
            elif isinstance(module, nn.LSTM):
                for layer in module.all_weights:
                    for weight in layer:
                        nn.init.normal_(weight)
                        
    def _forward(self, x):
        x = self.preprocessing(x)
        x = self.features(x)
        x = self.classifier(x)
        return x
        
    def forward(self, x):
        # batch_size x segment_size ( x num_channels) x num_freqs x num_frames
        batch_size, segment_size, num_channels, num_freqs, num_frames = x.shape
        x = torch.stack([
            self._forward(x.select(1, segment).view(batch_size, num_channels, num_freqs, num_frames))
            for segment in range(segment_size)], dim=1)
        x = F.log_softmax(x, dim=2)
        return x.mean(dim=1)
    
    
class DenseInceptionLSTM(nn.Module):
    def __init__(self, num_in_channels, num_out_channels, num_dense_modules, num_classes=10):
        manual_seed()
        super().__init__()
        
        self.preprocessing = nn.Sequential(
            nn.Conv2d(num_in_channels, num_out_channels,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(num_out_channels, eps=0.001),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(4, 1)))
        num_in_channels = num_out_channels

        self.features = nn.Sequential(
            DenseModule(num_dense_modules, num_in_channels,
                        num_out_channels),
            TransitionModule(num_in_channels + num_dense_modules * 4 * num_out_channels,
                             num_out_channels))
        num_in_channels = num_out_channels
        
        self.classifier = nn.Sequential(
            nn.BatchNorm2d(num_in_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Linear(num_in_channels, num_classes))

        self.aggregate = nn.Sequential(
            nn.LSTM(10, 10, batch_first=True),
            AverageHiddenStateFromLSTM(),
            nn.Linear(10, 10))
        
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
            elif isinstance(module, nn.LSTM):
                for layer in module.all_weights:
                    for weight in layer:
                        nn.init.normal_(weight)
                        
        self.load_state_dict(torch.load('checkpoints/densenet_randomized_segment_fold_2.pth'), strict=False)

    def _forward(self, x):
        x = self.preprocessing(x)
        x = self.features(x)
        x = self.classifier(x)
        return x
        
    def forward(self, x):
        # batch_size x segment_size x num_freqs x num_frames
        batch_size, segment_size, num_freqs, num_frames = x.shape
        x = torch.stack([
            self._forward(x.select(1, segment).view(batch_size, 1, num_freqs, num_frames))
            for segment in range(segment_size)], dim=1)
        x = self.aggregate(x)
        return x