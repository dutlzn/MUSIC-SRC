import os.path as osp
import sys
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import entropy

from models.basic import DenseBlock, NativeDenseBlock, Transition


class DCIN(nn.Module):
    name = 'DCIN'
    
    def __init__(self, num_input_features, growth_rate=32,
                 block_config=(3, 3, 6, 4), num_init_features=64,
                 bn_size=4, drop_rate=0, num_classes=10, Block=DenseBlock):
        super().__init__()
        
        self.num_input_features = num_input_features
        self.growth_rate = growth_rate
        self.block_config = block_config
        self.num_init_features = num_init_features
        self.bn_size = bn_size
        self.drop_rate = drop_rate
        self.num_classes = num_classes

        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(num_input_features, num_init_features,
                                kernel_size=3, padding=1)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=(4, 1))),
        ]))

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = Block(
                num_layers, num_features, growth_rate, bn_size, drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = Transition(num_input_features=num_features,
                                   num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        self.num_final_features = num_features
        self.features.add_module('norm%d' % (len(block_config) + 1),
            nn.BatchNorm2d(num_features))
        self.features.add_module('relu%d' % (len(block_config) + 1),
            nn.ReLU(inplace=True))
        self.features.add_module('pool%d' % (len(block_config) + 1),
            nn.AdaptiveAvgPool2d(1))

        self.classifier = nn.Sequential(
            nn.Linear(num_features, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def compute_features(self, x):
        num_samples, num_segs, num_feats, num_freqs, num_frames = x.shape
        x = torch.stack([self._forward(x.select(1, seg_idx).view(
            num_samples, num_feats, num_freqs, num_frames))
            for seg_idx in range(num_segs)], dim=1)
        return x

    def forward(self, x):
        x = self.compute_features(x)
        x = F.log_softmax(x, dim=2)
        return x.mean(dim=1)

    
class DCIN_V2(DCIN):
    name = 'DCIN_V2'
    
    def forward(self, x):
        x = self.compute_features(x)
        x = F.softmax(x, dim=2)
        x = x.mean(dim=1)
        return torch.log(x)
    

class WDCIN(DCIN):
    name = 'WDCIN'
    
    def confidence_weights(self, x):
        N, S, C = x.shape
        probs = x.view(N * S, C).transpose(0, 1)
        probs = probs.detach().cpu().numpy()
        # Calculate and rescale weights to [0, 1]
        weights = 1 / np.log(C) * (np.log(C) - entropy(probs))
        weights = weights.T.reshape(N, S)
        # Normalize with softmax to make sure that probabilities sums to 1
        weights = np.exp(weights - np.max(weights, axis=1, keepdims=True))
        weights = weights / weights.sum(axis=1, keepdims=True)
        weights = torch.FloatTensor(weights).to(x.device).unsqueeze_(2)
        return weights

    def forward(self, x):
        x = self.compute_features(x)
        x = F.softmax(x, dim=2)
        weights = self.confidence_weights(x)
        weights = F.softmax(weights, dim=1)
        x = weights * x
        x = x.sum(dim=1)
        return torch.log(x)

    
class AttentionBlock(nn.Module):
    def __init__(self, num_in_features, num_hidden_features=None):
        super().__init__()
        if num_hidden_features is None:
            num_hidden_features = num_in_features
        
        self.attention = nn.Sequential(OrderedDict([
            ('imp0', nn.Linear(num_in_features, num_hidden_features, bias=False)),
            ('relu0', nn.ReLU(inplace=True)),
            ('imp1', nn.Linear(num_hidden_features, 1, bias=False)),
            ('softmax2', nn.Softmax(dim=1))]))
        
    def forward(self, x):
        x = F.softmax(x, dim=2)
        weights = self.attention(x)
        x = weights * x
        x = x.sum(dim=1)
        return torch.log(x)

    
class ContextFreeDCIN(DCIN):
    name = 'CF-DCIN'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.attention_block = AttentionBlock(kwargs['num_classes'])

    def forward(self, x):
        x = self.compute_features(x)
        return self.attention_block(x)

    
class AttentionBlock2(nn.Module):
    def __init__(self, num_in_features, num_hidden_features=None):
        super().__init__()
        if num_hidden_features is None:
            num_hidden_features = num_in_features
        
        self.attention = nn.Sequential(OrderedDict([
            ('imp0', nn.Linear(num_in_features, num_hidden_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('imp1', nn.Linear(num_hidden_features, 1))]))
        
    def forward(self, x):
        num_samples, num_segments, num_features, *_ = x.shape
        x = x.view(num_samples * num_segments, num_features)
        attentions = self.attention(x)
        attentions = attentions.view(num_samples, num_segments, 1)
        attentions = F.softmax(attentions, dim=1)
        return attentions

    
class ContextFreeFeatureDCIN(DCIN):
    name = 'CFF-DCIN'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        num_in_features = self.num_final_features
        self.attention_block = AttentionBlock2(num_in_features, num_in_features)

    def _forward(self, x):
        # num_samples x num_features1 x num_freqs x num_frames
        x = self.features(x)
        # num_samples x num_features2 x 1
        return x
    
    def compute_features(self, x):
        num_samples, num_segs, num_feats, num_freqs, num_frames = x.shape
        x = torch.stack([self._forward(x.select(1, seg_idx).view(
            num_samples, num_feats, num_freqs, num_frames))
            for seg_idx in range(num_segs)], dim=1)
        # num_samples x num_segments x num_features2 x 1
        return x

    def forward(self, x):
        x = self.compute_features(x)
        attentions = self.attention_block(x)
        x = x.squeeze()
        x = self.classifier(x)

        x = F.softmax(x, dim=2)
        x = attentions * x
        x = x.sum(dim=1)
        return torch.log(x)
    
    
class ContextRelatedDCIN(DCIN):
    name = 'CR-DCIN'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.gru = nn.GRU(
            kwargs['num_classes'], kwargs['num_classes'],
            bias=False, batch_first=True)
        self.linear = nn.Linear(
            kwargs['num_classes'], kwargs['num_classes'])

    def forward(self, x):
        x = self.compute_features(x)
        _, x = self.gru(x)
        x = x[0]
        x = self.linear(x)
        return F.log_softmax(x, dim=1)