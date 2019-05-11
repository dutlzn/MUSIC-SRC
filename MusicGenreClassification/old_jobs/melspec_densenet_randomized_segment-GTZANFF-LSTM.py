
# coding: utf-8

# In[1]:


import os
import os.path as osp
import sys

sys.path.append(osp.abspath('..'))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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


from datasets.gtzan import GTZANFF_MELSPEC as GTZAN
from model import DenseInception
from utils import manual_seed
from train_utils import run_epoch

# In[2]:
manual_seed()
    
PRETRAIN_SEGMENTS = 10
PRETRAIN_BATCH_SIZE = 10
PRETRAIN_CHECKPOINT = 'checkpoints/gtzan_fault_filtered_cnn_pretrained.pt'
MIN_SEGMENTS = 10
SEGMENTS = 18
BATCH_SIZE = 8
CHECKPOINT = 'checkpoints/gtzan_fault_filtered_drnn.pt'
OVERLAP = 0.5
EPOCHS = 300
LR = 1e-3
FINE_TUNING_LR = 1e-4
LR_DECAY_RATE = 0.985
NUM_EARLY_STOPPING_PATIENCE = 50

DEVICE = torch.device('cuda:0')


# In[4]:


# CNN pretrain
criterion = nn.NLLLoss()
net = DenseInception(1, 32, 3, cnn_pretrain=True).to(DEVICE)
best_net = DenseInception(1, 32, 3, cnn_pretrain=True).to(DEVICE)
if not osp.exists(PRETRAIN_CHECKPOINT):
    optimizer = optim.Adam(net.parameters(), lr=LR)

    train_set = GTZAN(phase='train', min_segments=PRETRAIN_SEGMENTS,
                      randomized=True, overlap=OVERLAP, noise_rate=1e-3)
    val_set = GTZAN(phase='val', min_segments=10, overlap=OVERLAP)
    test_set = GTZAN(phase='test', min_segments=10, overlap=OVERLAP)
    manual_seed()
    train_loader = DataLoader(train_set, batch_size=PRETRAIN_BATCH_SIZE,
                              shuffle=True, num_workers=9, pin_memory=True,
                              worker_init_fn=lambda x: manual_seed(x))
    val_loader = DataLoader(val_set, batch_size=PRETRAIN_BATCH_SIZE, num_workers=9, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=PRETRAIN_BATCH_SIZE, num_workers=9, pin_memory=True)
    dataloaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
    }

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, LR_DECAY_RATE)
    scheduler.step()

    best_loss = 1e9
    patience = NUM_EARLY_STOPPING_PATIENCE

    tb_writer = SummaryWriter('runs/gtzan_fault_filtered_cnn_pretrain')

    manual_seed()
    with tqdm(range(EPOCHS), total=EPOCHS) as epoch_progress:
        for epoch in epoch_progress:
            train_loss = -1
            train_acc = -1
            val_loss = -1
            val_acc = -1
            for phase in ('train', 'val'):
                loss, acc = run_epoch(net, optimizer, dataloaders[phase],
                                      criterion, phase, pretrain=True)

                if phase == 'val':
                    val_loss = loss
                    val_acc = acc
                else:
                    train_loss = loss
                    train_acc = acc
                    scheduler.step()
            info = OrderedDict(
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc)

            for name, val in info.items():
                tb_writer.add_scalar(name, val, epoch)         
            epoch_progress.set_postfix(**info)

            if train_loss > best_loss:
                patience -= 1
            else:
                patience = NUM_EARLY_STOPPING_PATIENCE
                best_net.load_state_dict(net.state_dict())
                with open(PRETRAIN_CHECKPOINT, 'wb') as f:
                    torch.save(net.state_dict(), f)
            best_loss = min(train_loss, best_loss)
            if patience == 0:
                break
    
    manual_seed()            
    loss, acc = run_epoch(best_net, optimizer, dataloaders['test'], criterion, 'test')
    print('[CNN] Test loss: {:.3f}, Test acc: {:.2%}'.format(loss, acc))
else:
    best_net.load_state_dict(torch.load(PRETRAIN_CHECKPOINT))


    
# Final training
criterion = nn.CrossEntropyLoss()

net.cnn_pretrain = False
net.load_state_dict(best_net.state_dict())

optimizer = optim.Adam([
    {'params': net.preprocessing.parameters(), 'lr': FINE_TUNING_LR},
    {'params': net.features.parameters(), 'lr': FINE_TUNING_LR},
    {'params': net.classifier.parameters(), 'lr': FINE_TUNING_LR},
    {'params': net.aggregate.parameters()}], lr=LR)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, LR_DECAY_RATE)
scheduler.step()

tb_writer = SummaryWriter('runs/gtzan_fault_filtered')

train_set = GTZAN(phase='train', min_segments=SEGMENTS, randomized=True,
                  overlap=OVERLAP, noise_rate=1e-3)
val_set = GTZAN(phase='val', min_segments=SEGMENTS, overlap=OVERLAP)
test_set = GTZAN(phase='test', min_segments=SEGMENTS, overlap=OVERLAP)
manual_seed()
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=9, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=9, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, num_workers=9, pin_memory=True)
dataloaders = {
    'train': train_loader,
    'val': val_loader,
    'test': test_loader,
}

best_loss = 1e9
patience = NUM_EARLY_STOPPING_PATIENCE

manual_seed()
with tqdm(range(EPOCHS), total=EPOCHS) as epoch_progress:
    for epoch in epoch_progress:
        train_loss = -1
        train_acc = -1
        val_loss = -1
        val_acc = -1
        for phase in ('train', 'val'):
            loss, acc = run_epoch(net, optimizer, dataloaders[phase], criterion, phase)

            if phase == 'val':
                val_loss = loss
                val_acc = acc
            else:
                train_loss = loss
                train_acc = acc
                scheduler.step()
        info = OrderedDict(
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc)

        for name, val in info.items():
            tb_writer.add_scalar(name, val, epoch)         
        epoch_progress.set_postfix(**info)

        if val_loss > best_loss:
            patience -= 1
        else:
            patience = NUM_EARLY_STOPPING_PATIENCE
            best_net.load_state_dict(net.state_dict())
            with open(CHECKPOINT, 'wb') as f:
                torch.save(net.state_dict(), f)
        best_loss = min(val_loss, best_loss)
        if patience == 0:
            break

manual_seed()            
loss, acc = run_epoch(best_net, optimizer, dataloaders['test'], criterion, 'test')
print('Test loss: {:.3f}, Test acc: {:.2%}'.format(loss, acc))
