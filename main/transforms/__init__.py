from .transform import Compose, RandomTimeShift, RandomDCShift, RandomZeroMasking, RandomBandStopFilter, \
    RandomAdditiveGaussianNoise, RandomAmplitudeScale, TwoTransform, default, Multi_Transform, FFT_Transform, \
    normalize, Permutation
_idx_to_transforms = [RandomAmplitudeScale(), RandomTimeShift(), RandomDCShift(), RandomAdditiveGaussianNoise(),
                      RandomBandStopFilter(), RandomZeroMasking(), Permutation()]


def keys_to_transforms(keys, mode):
    res = []
    for index in range(len(keys)):
        transforms = [default()]
        for key in keys[index]:
            transforms.append(_idx_to_transforms[key])
        res.append(Compose(transforms, mode=mode[index]))
    return Multi_Transform(res)
