
from __future__ import print_function
import sys
import os
import numpy as np
import librosa

# mel-spec options
fftsize = 1024
window = 1024
hop = 512
melBin = 128

# A location where gtzan labels are located
label_path =  '../../data/GTZAN/'#'./gtzan/'

# read train / valid / test lists
with open(label_path + 'train_filtered.txt') as f:
    train_list = f.read().splitlines()
with open(label_path + 'valid_filtered.txt') as f:
    valid_list = f.read().splitlines()
with open(label_path + 'test_filtered.txt') as f:
    test_list = f.read().splitlines()

song_list = train_list+valid_list+test_list
print(len(song_list))