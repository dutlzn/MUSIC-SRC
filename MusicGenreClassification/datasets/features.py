import numpy as np
import librosa as rosa
import librosa.feature as rosaf

import utils
from datasets.args import build_args


const_args = build_args()


def get_audio(fpath, sr=const_args['sample_rate'],
              n_target=const_args['num_target_samples'],
              cut_strategy='random'):
    """Read raw audio for given file path.
    
    Parameters
    ----------
    fpath: str, a string to indicate which file needs to be read.
    sr: int, sample rate for the audio.
    n_target: int, desired length of audio.
    cut_strategy: str, one of ['random', 'begin', 'middle', 'end'], which
        cutting strategy to use when the length of audio is larger than the
        desired length.
    
    Returns
    -------
    x: np.ndarray, audio time series
    """
    x, sr = rosa.load(fpath, sr=sr)
    n_src = x.shape[0]
    if n_target is None or n_src == n_target:
        return x
    if n_src < n_target:
        x = np.hstack((x, np.zeros(n_target - n_src, dtype=np.float32)))
    elif n_src > n_target:
        if cut_strategy == 'random':
            start = np.random.randint(0, n_src - n_target + 1)
        elif cut_strategy == 'begin':
            start = 0
        elif cut_strategy == 'end':
            start = n_src - n_target
        else:
            start = (n_src - n_target) // 2
        x = x[start:start+n_target]
    return x


def get_spectrogram(y, padding=0):
    """Extract spectrogram from audio data.
    
    This implementation is identical to what's described in the original paper.
    
    Parameters
    ----------
    y: np.ndarray, the input singal(audio time series).
    padding(optional): int, pad the input signal.
    
    Returns
    -------
    D: np.ndarray, STFT matrix.
    """
    y = rosa.stft(y, n_fft=const_args['n_fft'],
                  hop_length=const_args['hop_length'],
                  window=const_args['window'])
    y = np.abs(y)
    y = np.flipud(y)
    y = np.log(1e-5+y) - np.log(1e-5)
    y = y - np.mean(y)
    y = y / np.sqrt(np.mean(np.power(y, 2)))
    r, c = y.shape
    for c_ in range(c):
        y[:, c_] = utils.smooth(y[:, c_], 19)
    y = np.concatenate((np.zeros((r, padding)), y, np.zeros((r, padding))), axis=1)
    return y


def get_melspectrogram(y, **kwargs):
    kwargs.update({
        'n_fft': const_args['n_fft'],
        'hop_length': const_args['hop_length'],
    })
    if const_args['pad_signal'] is True:
        return rosa.power_to_db(rosa.feature.melspectrogram(y, **kwargs))
    S = np.abs(rosa.stft(y, center=False, **kwargs))**2
    return rosa.power_to_db(rosa.feature.melspectrogram(S=S, **kwargs))


def get_melspectrogram_delta_deltadelta(y, **kwargs):
    """Compute Mel-spectrogram, delta features and delta-delta features."""
    melspec = get_melspectrogram(y, **kwargs)
    delta = rosaf.delta(melspec)
    delta_delta = rosaf.delta(melspec, order=2)
    return np.stack([melspec, delta, delta_delta])


def get_MFCC(y, **kwargs):
    """Extract Mel-frequency cepstral coefficients.
    
    This function is a wrapper of the MFCC function provided by librosa.
    
    Parameters
    ----------
    y: np.ndarray, the input singal(audio time series).
    kwargs: dict, optinal keyword arguments for librosa.feature.mfcc().
    
    Returns
    -------
    M: np.ndarray, MFCC sequence, plus its delta and delta-delta features
    """
    kwargs.update({
        'n_fft': const_args['n_fft'],
        'hop_length': const_args['hop_length'],
        'n_mfcc': const_args['n_mfccs'],
    })
    return rosa.feature.mfcc(y, **kwargs)