import numpy as np
from torch.utils.data import Dataset as BaseDataset

from datasets.args import build_args


const_args = build_args()


class Dataset(BaseDataset):
    def __init__(self, phase='train', test_size=0.2, val_size=0,
                 root='/share', random_seed=0,
                 transforms=None, *args, **kwargs):
        super().__init__()
        
        self.phase = phase
        self.test_size = test_size
        self.val_size = val_size
        self.root = root
        self.random_seed = random_seed
        self.transforms = transforms
        

class SegmentDataset:
    """A base class for splitting audio into segments."""
    NONE = 0
    SEQUENTIAL_RANDOM = 1
    NONSEQUENTIAL_RANDOM = 2

    def __init__(self, min_segments=None, max_segments=None,
                 randomized_method=NONE, overlap=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_segments = min_segments
        if max_segments is None:
            self.max_segments = min_segments
        else:
            assert min_segments <= max_segments
            self.max_segments = max_segments
        assert 0 <= overlap < 1
        self.overlap = overlap
        self.segment_samples = const_args['num_target_segment_samples']
        self.randomized = randomized_method
        self.segments = None

    def __getitem__(self, idx):
        x, y = super().__getitem__(idx)
        if self.min_segments is None or self.min_segments == 0:
            return x[None], y
        if self.segments is None:
            self.segments = np.random.randint(self.min_segments, self.max_segments+1)
        segments = self.segments
        origin_length, = x.shape
        extra_samples = int((1 - self.overlap) * self.segment_samples)
        target_length = self.segment_samples + (segments - 1) * extra_samples
        if self.randomized == self.SEQUENTIAL_RANDOM:
            start = np.random.randint(0, 1 + origin_length - target_length)
            start_points = np.arange(start, start + extra_samples * segments, extra_samples)
        elif self.randomized == self.NONSEQUENTIAL_RANDOM:
            start_points = np.random.randint(0, origin_length - target_length, segments)
        else:
            start = 0
            start_points = np.arange(start, start + extra_samples * segments, extra_samples)
        x = np.stack([x[s:s+self.segment_samples] for s in start_points]).astype(np.float32)
        return x, y

    
class FeatureExtracterDataSet:
    """A base class to provide feature extracting logic."""
    def __init__(self, feature_extractor, transforms=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_extractor = feature_extractor
        self.transforms = transforms
        
    def __getitem__(self, idx):
        x, y = super().__getitem__(idx)
        x = np.array([self.feature_extractor(xr) for xr in x])
        x = x[:, np.newaxis, ...]
        if self.transforms:
            x = self.transforms(x)
        return x, y
    
    
class NoisedDataSet:
    """A base class to augment data by adding noises."""
    def __init__(self, noise_rate=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_rate = noise_rate
        
    def __getitem__(self, idx):
        x, y = super().__getitem__(idx)
        x = x + self.noise_rate * np.random.normal(size=x.shape)
        return x, y