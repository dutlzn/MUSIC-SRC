
# coding: utf-8

# In[1]:


import os
import os.path as osp
import sys
from collections import OrderedDict
sys.path.append(osp.abspath('..'))
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import torch
import torch.cuda as cuda
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm as tqdm

from datasets.gtzan import GTZAN_MELSPEC as GTZAN


# In[2]:


# Random seeds
torch.manual_seed(1234)
cuda.manual_seed_all(1234)
np.random.seed(1234)

SEGMENTS = 20
BATCH_SIZE = 4
EPOCHS = 300
LR = 1e-3
NUM_CLASSES = 10
NUM_KFOLD = 10
NUM_REDUCE_LR_PATIENCE = 5
NUM_EARLY_STOPPING_PATIENCE = 15

DEVICE = torch.device('cuda:0')


# In[3]:


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


class DenseInception(nn.Module):
    def __init__(self, num_in_channels, num_out_channels, num_dense_modules, num_classes=10):
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
        
    def forward(self, x):
        x = self.preprocessing(x)
        x = self.features(x)
        x = self.classifier(x)
        return x


# In[4]:


def run_epoch(net, dataloader, criterion, phase):
    if phase == 'train':
        net.train()
    else:
        net.eval()

    running_samples = 0
    running_segments = 0
    running_loss = 0
    running_corrects = 0
    running_seg_corrects = 0
    with tqdm(dataloader, total=len(dataloader)) as progress:
        for inputs, labels in progress:
            num_samples, num_segments, num_freqs, num_frames = inputs.shape
            inputs = inputs.type(torch.FloatTensor).to(DEVICE)
            labels = labels.to(DEVICE)
            labels_ = labels
            inputs_ = inputs

            inputs = inputs.view(num_samples * num_segments, 1, num_freqs, num_frames)
            labels = labels.expand(num_segments, num_samples).transpose_(0, 1)
            labels = labels.contiguous().view(labels.numel())

            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                outputs = net(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_samples += num_samples
            running_segments += num_samples * num_segments

            running_loss += loss.item() * num_samples * num_segments

            preds = outputs.contiguous().view(num_samples, num_segments, NUM_CLASSES)
            preds = F.softmax(preds, dim=2)
            preds = preds.sum(dim=1)
            _, preds = torch.max(preds, 1)
            running_corrects += torch.sum(preds == labels_.data).item()

            _, seg_preds = torch.max(outputs, 1)
            running_seg_corrects += torch.sum(seg_preds == labels.data).item()

            loss = running_loss / running_segments
            acc = running_corrects / running_samples
            seg_acc = running_seg_corrects / running_segments

            progress.set_postfix(OrderedDict(
                phase=phase,
                loss='{:.4f}'.format(loss),
                acc='{:.2%}'.format(acc),
                seg_acc='{:.2%}'.format(seg_acc)))

    return loss, acc, seg_acc


criterion = nn.CrossEntropyLoss()
cv_results = []
dataset = GTZAN(phase='all', min_segments=SEGMENTS)
train_set = GTZAN(phase='all', min_segments=SEGMENTS, overlap=0.5)
test_set = GTZAN(phase='all', min_segments=SEGMENTS, overlap=0.5)

skf = StratifiedKFold(NUM_KFOLD, shuffle=True, random_state=1234)
for kfold, (train_index, test_index) in enumerate(skf.split(dataset.X, dataset.Y)):
    train_set.X, train_set.Y = dataset.X[train_index], dataset.Y[train_index]
    test_set.X, test_set.Y = dataset.X[test_index], dataset.Y[test_index]
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=9, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, num_workers=9, pin_memory=True)
    dataloaders = {
        'train': train_loader,
        'test': test_loader,
    }
    
    net = DenseInception(1, 32, 3).to(DEVICE)
    optimizer = optim.Adam(net.parameters(), lr=LR)
    
    reduce_lr = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=NUM_REDUCE_LR_PATIENCE)
    best_loss = 1e9
    patience = NUM_EARLY_STOPPING_PATIENCE
    
    info = {}
    tb_writer = SummaryWriter('runs/densenet_sequential_segment_fold_%d' % kfold)

    with tqdm(range(EPOCHS), total=EPOCHS) as epoch_progress:
        for epoch in epoch_progress:
            early_stopping = False
            train_loss = -1
            train_acc = -1
            train_seg_acc = -1
            test_loss = -1
            test_acc = -1
            test_seg_acc = -1
            for phase in ('train', 'test'):
                loss, acc, seg_acc = run_epoch(net, dataloaders[phase], criterion, phase)
                
                if phase == 'test':
                    test_loss = loss
                    test_acc = acc
                    test_seg_acc = seg_acc
                else:
                    train_loss = loss
                    train_acc = acc
                    train_seg_acc = seg_acc
                    reduce_lr.step(loss)
            info = OrderedDict(
                train_loss=train_loss,
                train_acc=train_acc,
                train_seg_acc=train_seg_acc,
                test_loss=test_loss,
                test_acc=test_acc,
                test_seg_acc=test_seg_acc)
            
            for name, val in info.items():
                tb_writer.add_scalar(name, val, epoch)         
            epoch_progress.set_postfix(**info)
            
            if train_loss > best_loss:
                patience -= 1
            else:
                patience = NUM_EARLY_STOPPING_PATIENCE
            best_loss = min(train_loss, best_loss)
            if patience == 0:
                break
    # Collect cross-validate summaries
    cv_results.append(info)
    
    with open('checkpoints/densenet_sequential_segment_fold_%d.pth' % kfold, 'wb') as f:
        torch.save(net.state_dict(), f)

# In[ ]:

print('\n')
for kfold, result in enumerate(cv_results):
    print('Fold {}, train loss: {:.4f}, train acc: {:.2%}, train seg acc: {:.2%}, '
          'test loss: {:.4f}, test acc: {:.2%}, test seg acc: {:.2%}'.format(
              kfold, result['train_loss'], result['train_acc'], result['train_seg_acc'],
              result['test_loss'], result['test_acc'], result['test_seg_acc']))
    
print('{}-fold cross-validation'.format(len(cv_results)))
print('train loss: {:.4f}'.format(sum(x['train_loss'] for x in cv_results) / len(cv_results)))
print('train acc: {:.2%}'.format(sum(x['train_acc'] for x in cv_results) / len(cv_results)))
print('train seg acc: {:.2%}'.format(sum(x['train_seg_acc'] for x in cv_results) / len(cv_results)))
print('test loss: {:.4f}'.format(sum(x['test_loss'] for x in cv_results) / len(cv_results)))
print('test acc: {:.2%}'.format(sum(x['test_acc'] for x in cv_results) / len(cv_results)))
print('test seg acc: {:.2%}'.format(sum(x['test_seg_acc'] for x in cv_results) / len(cv_results)))

