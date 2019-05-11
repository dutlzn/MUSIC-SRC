
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

SEGMENTS = 10
BATCH_SIZE = 4
EPOCHS = 300
LR = 1e-3
NUM_CLASSES = 10
NUM_KFOLD = 10
NUM_REDUCE_LR_PATIENCE = 3
NUM_EARLY_STOPPING_PATIENCE = 10

DEVICE = torch.device('cuda:0')


# In[3]:


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x
    

class NET(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = models.vgg16_bn().features
        self.classifier = nn.Sequential(
            Flatten(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes))

    def forward(self, x):
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

            inputs = inputs.view(num_samples * num_segments, 1, num_freqs, num_frames)
            inputs = inputs.expand(-1, 3, -1, -1)
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


# In[5]:


criterion = nn.CrossEntropyLoss()
cv_results = []
dataset = GTZAN(phase='all', min_segments=SEGMENTS)
train_set = GTZAN(phase='all', min_segments=SEGMENTS, randomized=True)
test_set = GTZAN(phase='all', min_segments=SEGMENTS)

skf = StratifiedKFold(NUM_KFOLD, shuffle=True, random_state=1234)
for kfold, (train_index, test_index) in enumerate(skf.split(dataset.X, dataset.Y)):
    train_set.X, train_set.Y = dataset.X[train_index], dataset.Y[train_index]
    test_set.X, test_set.Y = dataset.X[test_index], dataset.Y[test_index]
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=9)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, num_workers=9)

    dataloaders = {
        'train': train_loader,
        'test': test_loader,
    }
    
    net = NET().to(DEVICE)
    optimizer = optim.Adam(net.parameters(), lr=LR)
    
    reduce_lr = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=NUM_REDUCE_LR_PATIENCE)
    best_loss = 1e9
    patience = NUM_EARLY_STOPPING_PATIENCE
    
    info = {}
    tb_writer = SummaryWriter('runs/vgg16_randomized_segment_fold_%d' % kfold)

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
    
    with open('checkpoints/vgg16_randomized_segment_fold_%d.pth' % kfold, 'wb') as f:
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

