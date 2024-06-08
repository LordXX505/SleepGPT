from .transform import Compose, RandomTimeShift, RandomDCShift, RandomZeroMasking, RandomBandStopFilter, \
    RandomAdditiveGaussianNoise, RandomAmplitudeScale, TwoTransform, default, Multi_Transform, FFT_Transform, \
    normalize, Permutation, RandomTimeInverted, unnormalize
_idx_to_transforms = [RandomAmplitudeScale(), RandomTimeShift(), RandomDCShift(), RandomAdditiveGaussianNoise(),
                      RandomBandStopFilter(), RandomZeroMasking(), Permutation(), RandomTimeInverted()]


def keys_to_transforms(keys, mode, show_param):
    res = []
    for index in range(len(keys)):
        transforms = [default()]
        for key in keys[index]:
            transforms.append(_idx_to_transforms[key])
        res.append(Compose(transforms, mode=mode[index]))
    return Multi_Transform(res, show_param=show_param)
