import torch
import random
import numpy as np
from scipy import signal
from scipy.ndimage.interpolation import shift
import torch.nn.functional as F
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
        if torch.cuda.is_available():
            mask = torch.cuda.FloatTensor(x.shape).uniform_() > pertub_ratio  # maskout_ratio are False
        else:
            mask = torch.empty(x.shape, dtype=torch.float32).uniform_() > pertub_ratio  # maskout_ratio are False
        mask = mask.to(x.device)
        return x*mask

    def add_frequency(self, x, pertub_ratio=0.0):
        if torch.cuda.is_available():
            mask = torch.cuda.FloatTensor(x.shape).uniform_() > (
                        1 - pertub_ratio)  # only pertub_ratio of all values are True
        else:
            mask = torch.empty(x.shape, dtype=torch.float32).uniform_() > (
                        1 - pertub_ratio)
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
    def __init__(self, transform, show_param):
        self.transform = transform
        self.show_param = show_param

    def __call__(self, x, label=None, ):
        if label is not None:
            labels = []
        else:
            labels = None
        res = []
        for transform in self.transform:
            item = transform(x, label, self.show_param)
            if label is not None:
                labels.append(item[1])
                res.append(item[0])
            else:
                res.append(item)
        if label is not None:
            x = torch.stack(res, dim=0).squeeze(0).float()
            label = torch.stack(labels, dim=0).squeeze(0).float()
            item = (x, label)
        else:
            x = torch.stack(res, dim=0).squeeze(0).float()
            item = x
        return item

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transform:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class normalize:
    def __init__(self):
        #[-5.77058569 -5.893796   -5.29914995 -5.2774984  -5.58912249 -6.17521508, -3.43324822 -4.57360642 -4.63527194]
        #['abd', 'airflow', 'c3', 'c4', 'ecg', 'emg', 'eog', 'f3', 'o1']
        self.mu_11 = torch.tensor([-5.77058569, -5.893796, -5.29914995,
                                   -5.2774984, -5.58912249, -6.17521508,
                                   -3.43324822, -4.57360642,  8.1125e-04, -4.63527194, 7.1748e-05])
        #[791.409925   811.93428768 845.27496042 773.86625409 803.77746945 870.667796   741.95722232 755.83128998 785.98923913]
        self.std_11 =torch.tensor([791.409925, 811.93428768,  845.27496042, 773.86625409, 803.77746945,
                                   870.667796, 741.95722232, 755.83128998, 4.9272,  785.98923913, 3.6142])
        self.mu4 = torch.tensor([-6.8460e-02,  1.9104e-01,  3.8937e-01, -2.0938e+00,
         ])
        self.std4 = torch.tensor([34.6887,  34.9556, 23.2826,  35.4035])
        self.mu = torch.tensor([-6.8460e-02,  1.9104e-01,  3.8937e-01, -2.0938e+00,
         1.6496e-03,-4.8439e-05,  8.1125e-04,
         7.1748e-05])
        self.std = torch.tensor([34.6887,  34.9556, 23.2826,  35.4035,  26.8738,
          4.9272,  25.1366,   3.6142])

    def __call__(self, x, attention_mask=None):
        if x.shape[0] == 11:
            # max_val, indices = torch.max(x, keepdim=True, dim=-1)
            # min_val, indices = torch.min(x, keepdim=True,dim=-1)
            # return (x - min_val + 1e-6) / (1e-6 + max_val - min_val)
            return x
        elif x.shape[0] == 4:
            return (x - self.mu4.unsqueeze(-1)) / self.std4.unsqueeze(-1)
        else:
            return (x - self.mu.unsqueeze(-1)) / self.std.unsqueeze(-1)

class unnormalize:
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
            return x * self.std4.unsqueeze(-1) + self.mu4.unsqueeze(-1)
        else:
            return x * self.std.unsqueeze(-1) + self.mu.unsqueeze(-1)

class Compose:

    def __init__(self, transforms, mode='full'):
        self.transforms = transforms
        self.mode = mode
        # self.normalize = normalize()

    def __call__(self, x, label=None, *args, **kwargs):
        # x = self.normalize(x)
        # print(f"Using transforms: {len(self.transforms)}")
        if self.mode == 'random':
            index = random.randint(0, len(self.transforms) - 1)
            x, label = self.transforms[index](x, label, *args, **kwargs)
        elif self.mode == 'full':
            for t in self.transforms:
                x, label = t(x, label, *args, **kwargs)
        elif self.mode == 'shuffle':
            transforms = np.random.choice(self.transforms, len(self.transforms), replace=False)
            for t in transforms:
                x, label = t(x, label, *args, **kwargs)
        else:
            raise NotImplementedError
        if label is None:
            return x
        else:
            return x, label

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

    def __call__(self, x, label=None, *args, **kwargs):
        if label is not None:
            return x, label
        else:
            return x, None

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Permutation:
    def __init__(self,  patch_size=200, p=0.5):
        self.p = p
        self.patch_size = patch_size

    def __call__(self, x, label=None, *args, **kwargs):
        if torch.rand(1) < self.p:
            C, L = x.shape
            n = L//self.patch_size
            x = x.reshape(C, L//self.patch_size, self.patch_size)
            noise = torch.rand(n, device=x.device)  # noise in [0, 1]
            ids_shuffle = torch.argsort(noise, dim=-1).unsqueeze(0).unsqueeze(-1)
            x = torch.gather(x, dim=1, index=ids_shuffle.repeat(C, 1, self.patch_size))
            if label is not None:
                if (label.shape[0]//self.patch_size) != (L//self.patch_size):
                    label_patch_nums = label.shape[0]//self.patch_size
                    need_pad_patches = L//self.patch_size - label_patch_nums
                    p1d = [0, need_pad_patches*self.patch_size]
                    label = F.pad(label, p1d, "constant", 0)
                label = label.reshape(L // self.patch_size, self.patch_size)
                label = torch.gather(label, dim=0, index=ids_shuffle.squeeze(0).repeat(1, self.patch_size))
                label = label.reshape(L)
            x = x.reshape(C, L)
            if label is not None:
                return x, label
            else:
                return x, None
        if label is not None:
            return x, label
        else:
            return x, None

    def __repr__(self):
        return self.__class__.__name__ + '()'

class RandomAmplitudeScale:

    def __init__(self, range=(0.5, 2.0), p=0.5,):
        self.range = range
        self.p = p

    def __call__(self, x, label=None, show_param=False, *args, **kwargs):
        if torch.rand(1) < self.p:
            scale = random.uniform(self.range[0], self.range[1])
            scale_tensor = torch.tensor(scale, device=x.device)
            if show_param is True:
                print(f'RandomAmplitudeScale: {scale_tensor}')
            if label is not None:
                return x * scale_tensor, label
            else:
                return x * scale_tensor, None
        if label is not None:
            return x, label
        else:
            return x, None

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomDCShift:

    def __init__(self, range=(-2.5, 2.5), p=0.5):
        self.range = range
        self.p = p

    def __call__(self, x, label=None, show_param=False, *args, **kwargs):
        if torch.rand(1) < self.p:
            shift = random.uniform(self.range[0], self.range[1])
            shift_tensor = torch.tensor(shift, device=x.device)
            if show_param is True:
                print(f'RandomDCShift: {shift}')
            if label is not None:
                return x + shift_tensor, label
            else:
                return x + shift_tensor, None
        if label is not None:
            return x, label
        else:
            return x, None

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomTimeShift:

    def __init__(self, range=(-1000, 1000), mode='constant', cval=0.0, p=0.5):
        self.range = range
        self.mode = mode
        self.cval = cval
        self.p = p

    def __call__(self, x, label=None, show_param=False, *args, **kwargs):
        # if torch.rand(1) < 1:
        if torch.rand(1) < self.p:
            t_shift = random.randint(self.range[0], self.range[1])
            if show_param is True:
                print(f'RandomTimeShift: {RandomTimeShift}')
            if label is not None:
                L = label.shape[0]
                tmp = x
                if L != x.shape[1]:
                    tmp = x[:, :L]
                tmp = torch.roll(tmp, shifts=t_shift, dims=1)
                x = torch.cat([tmp, x[:, L:]], dim=1)
                label = torch.roll(label, shifts=t_shift, dims=0)
        if label is not None:
            return x, label
        else:
            return x, None

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomZeroMasking:

    def __init__(self, range=(0, 500), p=0.5):
        self.range = range
        self.p = p

    def __call__(self, x, label=None, *args, **kwargs):
        if torch.rand(1) < self.p:
            mask_len = random.randint(self.range[0], self.range[1])
            random_pos = random.randint(0, x.shape[1] - mask_len)
            mask = torch.concatenate(
                [torch.ones((1, random_pos)), torch.zeros((1, mask_len)), torch.ones((1, x.shape[1] - mask_len - random_pos))],
                dim=1)
            if label is not None:
                return x * mask, label
            else:
                return x * mask
        if label is not None:
            return x, label
        else:
            return x, None

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomAdditiveGaussianNoise:

    def __init__(self, range=(0.0, 2.5), p=0.5):
        self.range = range
        self.p = p

    def __call__(self, x, label=None, show_param=False, *args, **kwargs):
        if torch.rand(1) < self.p:
            sigma = random.uniform(self.range[0], self.range[1])
            if show_param is True:
                print(f'RandomAdditiveGaussianNoise, sigma: {sigma}')
            if label is not None:
                return x + torch.normal(0, sigma, x.shape, device=x.device), label
            else:
                return x + torch.normal(0, sigma, x.shape, device=x.device), None
        if label is not None:
            return x, label
        else:
            return x, None

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomBandStopFilter:

    def __init__(self, range=(0.3, 35.0), band_width=2.0, sampling_rate=100.0, p=0.5):
        self.range = range
        self.band_width = band_width
        self.sampling_rate = sampling_rate
        self.p = p

    def __call__(self, x, label=None, *args, **kwargs):
        if torch.rand(1) < self.p:
            low_freq = random.uniform(self.range[0], self.range[1])
            center_freq = low_freq + self.band_width / 2.0
            b, a = signal.iirnotch(center_freq, center_freq / self.band_width, fs=self.sampling_rate)
            x = torch.from_numpy(
                signal.lfilter(b, a, x))
            if label is not None:
                return x, label
            else:
                return x, None
        if label is not None:
            return x, label
        else:
            return x, None

    def __repr__(self):
        return self.__class__.__name__ + '()'

class RandomTimeInverted:

    def __init__(self, p=0.5):
        self.range = range
        self.p = p

    def __call__(self, x, label=None, *args, **kwargs):
        if torch.rand(1) < self.p:
            if label is not None:
                L = label.shape[0]
                tmp = x
                if L != x.shape[1]:
                    tmp = x[:, :L]
                tmp = torch.flip(tmp, dims=[1])
                x = torch.cat([tmp, x[:, L:]], dim=1)
                label = torch.flip(label, dims=[0])
        if label is not None:
            return x, label
        else:
            return x, None

    def __repr__(self):
        return self.__class__.__name__ + '()'