
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

# # read train / valid / test lists
# with open(label_path + 'train_filtered.txt') as f:
#     train_list = f.read().splitlines()
# with open(label_path + 'valid_filtered.txt') as f:
#     valid_list = f.read().splitlines()
# with open(label_path + 'test_filtered.txt') as f:
#     test_list = f.read().splitlines()

# read train / valid / test lists
with open(label_path + 'filtered_train_files.txt') as f:
    train_list = f.read().splitlines()
with open(label_path + 'filtered_val_files.txt') as f:
    valid_list = f.read().splitlines()
with open(label_path + 'filtered_train_files.txt') as f:
    test_list = f.read().splitlines()

song_list = train_list+valid_list+test_list
print(len(song_list))

# A location where gtzan dataset is located
load_path = '../../data/GTZAN/genres/'

# A location where mel-spectrogram would be saved
save_path = '../../data/GTZAN/genres/gtzan_mel/'

def mkdir(path):
    if os.path.exists(path):
        print("目录/文件已经创建")
    else:
        os.makedirs(path)
        print("成功创建目录")

def main():
    # mkdir(load_path)
    # mkdir(save_path)
    # save mel-spectrograms
    for iter in range(0,len(song_list)):
        # song_list[iter] blues/blues.00012.wav
        #因为txt文件里面的列表是wav格式，分析的时候需要把后缀改成au，保存npy文件的时候需要把后缀名改成npy
        file_name = load_path + song_list[iter].replace('.wav','.au')
        save_name = save_path + song_list[iter].replace('.wav','.npy')
        
        #os.path.dirname 是去掉文件名 返回目录
        if not os.path.exists(os.path.dirname(save_name)):
            os.makedirs(os.path.dirname(save_name))

        if os.path.isfile(save_name) == 1:
            print(iter, save_name + "_file_already_extracted!")
            continue

        # STFT
        y,sr = librosa.load(file_name,sr=22050)
        S = librosa.core.stft(y,n_fft=fftsize,hop_length=hop,win_length=window)
        X = np.abs(S)

        # mel basis
        mel_basis = librosa.filters.mel(sr,n_fft=fftsize,n_mels=melBin)

        # mel basis are multiplied to the STFT
        mel_S = np.dot(mel_basis,X)

        # log amplitude compression
        mel_S = np.log10(1+10*mel_S)
        mel_S = mel_S.astype(np.float32)

        # cut audio to have 30-second size for all files
        Nframes = int(29.9*22050.0/hop)
        if mel_S.shape[1] > Nframes:
            mel_S = mel_S[:,:Nframes]

        # save file
        print(iter,mel_S.shape,save_name)
        np.save(save_name,mel_S)



if __name__ == '__main__':
    main()
