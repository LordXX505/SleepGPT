import torch
import random
import numpy as np
from scipy import signal
from scipy.ndimage.interpolation import shift

class FFT_Transform:

    def __init__(self):
        pass
    def DataTransform_FD(self, sample):
        """Weak and strong augmentations in Frequency domain """
        aug_1 = self.remove_frequency(sample, pertub_ratio=0.1)
        aug_2 = self.add_frequency(sample, pertub_ratio=0.1)
        aug_F = aug_1 + aug_2
        return aug_F

    def remove_frequency(self, x, pertub_ratio=0.0):
        mask = torch.cuda.FloatTensor(x.shape).uniform_() > pertub_ratio  # maskout_ratio are False
        mask = mask.to(x.device)
        return x*mask

    def add_frequency(self, x, pertub_ratio=0.0):
        mask = torch.cuda.FloatTensor(x.shape).uniform_() > (1-pertub_ratio) # only pertub_ratio of all values are True
        mask = mask.to(x.device)
        max_amplitude = x.max()
        random_am = torch.rand(mask.shape, device=mask.device)*(max_amplitude*0.1)
        pertub_matrix = mask*random_am
        return x+pertub_matrix

    def __call__(self, x):
        return self.DataTransform_FD(x)
class TwoTransform:

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class Multi_Transform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return torch.stack([transform(x) for transform in self.transform], dim=0).squeeze().float()

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transform:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class normalize:
    def __init__(self):
        # self.mu = torch.tensor([-6.8460e-02,  1.9104e-01,  4.1165e+01,  3.8937e-01, -2.0938e+00,
        #  1.6496e-03, -2.6778e-03, -4.8439e-05,  8.1125e-04, -8.7787e-04,
        #  7.1748e-05])
        # self.std = torch.tensor([34.6887,  34.9556, 216.6215,  23.2826,  35.4035,  26.8738,  26.9540,
        #   4.9272,  25.1366,  24.5395,   3.6142])
        # self.mu = torch.tensor([-6.8460e-02,  1.9104e-01,  4.1165e+01, -2.0938e+00,
        #  1.6496e-03, -4.8439e-05,  8.1125e-04,
        #  7.1748e-05])
        # self.std4 = torch.tensor([34.6887,  34.9556, 216.6215,  35.4035,  26.8738,
        #  4.9272,  25.1366,   3.6142])
        self.mu4 = torch.tensor([-6.8460e-02,  1.9104e-01,  3.8937e-01, -2.0938e+00,
         ])
        self.std4 = torch.tensor([34.6887,  34.9556, 23.2826,  35.4035])
        self.mu = torch.tensor([-6.8460e-02,  1.9104e-01,  3.8937e-01, -2.0938e+00,
         1.6496e-03,-4.8439e-05,  8.1125e-04,
         7.1748e-05])
        self.std = torch.tensor([34.6887,  34.9556, 23.2826,  35.4035,  26.8738,
          4.9272,  25.1366,   3.6142])

    def __call__(self, x, attention_mask):
        if x.shape[0] == 4:
            return (x - self.mu4.unsqueeze(-1)) / self.std4.unsqueeze(-1)
        else:
            return (x - self.mu.unsqueeze(-1)) / self.std.unsqueeze(-1)


class Compose:

    def __init__(self, transforms, mode='full'):
        self.transforms = transforms
        self.mode = mode
        # self.normalize = normalize()

    def __call__(self, x):
        # x = self.normalize(x)
        # print(f"Using transforms: {len(self.transforms)}")
        if self.mode == 'random':
            index = random.randint(0, len(self.transforms) - 1)
            x = self.transforms[index](x)
        elif self.mode == 'full':
            for t in self.transforms:
                x = t(x)
        elif self.mode == 'shuffle':
            transforms = np.random.choice(self.transforms, len(self.transforms), replace=False)
            for t in transforms:
                x = t(x)
        else:
            raise NotImplementedError
        return x

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class default:
    def __init__(self):
        pass

    def __call__(self, x):
        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Permutation:
    def __init__(self,  patch_size=200, p=0.5):
        self.p = p
        self.patch_size = patch_size

    def __call__(self, x):
        if torch.rand(1) < self.p:
            C, L = x.shape
            n = L//self.patch_size
            x = x.reshape(C, L//self.patch_size, self.patch_size)
            noise = torch.rand(n, device=x.device)  # noise in [0, 1]
            ids_shuffle = torch.argsort(noise, dim=-1).unsqueeze(0).unsqueeze(-1)
            x = torch.gather(x, dim=1, index=ids_shuffle.repeat(C, 1, self.patch_size))
            x = x.reshape(C, L)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'

class RandomAmplitudeScale:

    def __init__(self, range=(0.5, 2.0), p=0.5):
        self.range = range
        self.p = p

    def __call__(self, x):
        if torch.rand(1) < self.p:
            scale = random.uniform(self.range[0], self.range[1])
            return x * scale
        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomDCShift:

    def __init__(self, range=(-2.5, 2.5), p=0.5):
        self.range = range
        self.p = p

    def __call__(self, x):
        if torch.rand(1) < self.p:
            shift = random.uniform(self.range[0], self.range[1])
            return x + shift
        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomTimeShift:

    def __init__(self, range=(-500, 500), mode='constant', cval=0.0, p=0.5):
        self.range = range
        self.mode = mode
        self.cval = cval
        self.p = p

    def __call__(self, x):
        if torch.rand(1) < self.p:
            t_shift = random.randint(self.range[0], self.range[1])
            x = torch.roll(x, shifts=t_shift, dims=1)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomZeroMasking:

    def __init__(self, range=(0, 500), p=0.5):
        self.range = range
        self.p = p

    def __call__(self, x):
        if torch.rand(1) < self.p:
            mask_len = random.randint(self.range[0], self.range[1])
            random_pos = random.randint(0, x.shape[1] - mask_len)
            mask = torch.concatenate(
                [torch.ones((1, random_pos)), torch.zeros((1, mask_len)), torch.ones((1, x.shape[1] - mask_len - random_pos))],
                dim=1)
            return x * mask
        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomAdditiveGaussianNoise:

    def __init__(self, range=(0.0, 2.5), p=0.5):
        self.range = range
        self.p = p

    def __call__(self, x):
        if torch.rand(1) < self.p:
            sigma = random.uniform(self.range[0], self.range[1])
            return x + torch.normal(0, sigma, x.shape)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomBandStopFilter:

    def __init__(self, range=(0.3, 35.0), band_width=2.0, sampling_rate=100.0, p=0.5):
        self.range = range
        self.band_width = band_width
        self.sampling_rate = sampling_rate
        self.p = p

    def __call__(self, x):
        if torch.rand(1) < self.p:
            low_freq = random.uniform(self.range[0], self.range[1])
            center_freq = low_freq + self.band_width / 2.0
            b, a = signal.iirnotch(center_freq, center_freq / self.band_width, fs=self.sampling_rate)
            x = torch.from_numpy(
                signal.lfilter(b, a, x))

        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'
