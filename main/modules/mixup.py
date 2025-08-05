import numpy as np
import torch
from pytorch_lightning.utilities.rank_zero import rank_zero_info

def one_hot(x, num_classes, on_value=1., off_value=0., device='cuda'):
    x = x.long().view(-1, 1)
    return torch.full((x.size()[0], num_classes), off_value, device=device).scatter_(1, x, on_value)

def mixup_target(target, num_classes, lam=1., smoothing=0.0, device='cuda'):
    target = target.long().view(-1, 1)
    off_value = smoothing / num_classes
    on_value = 1. - smoothing + off_value
    y1 = one_hot(target, num_classes, on_value=on_value, off_value=off_value, device=device)
    y2 = one_hot(target.flip(0), num_classes, on_value=on_value, off_value=off_value, device=device)
    return y1 * lam + y2 * (1. - lam)


def rand_bbox(size, lam):
    seq_len = size[-1]
    cut_rat = 1. - lam
    cut_len = np.int(seq_len * cut_rat)
    # uniform
    cx = np.random.randint(seq_len)

    bbx1 = np.clip(cx - cut_len // 2, 0, seq_len)
    bbx2 = np.clip(cx + cut_len // 2, 0, seq_len)
    return bbx1, bbx2

class Mixup:
    def __init__(self, mixup_alpha=1., prob=1.0, switch_prob=0.5,
                 mode='batch', correct_lam=True, label_smoothing=0.1, num_classes=5):
        self.mixup_alpha = mixup_alpha
        self.mix_prob = prob
        self.switch_prob = switch_prob
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.mode = mode
        self.correct_lam = correct_lam  # correct lambda based on clipped area for cutmix
        self.mixup_enabled = True  # set to false to disable mixing (intended tp be set by train loop)
    def _params_per_batch(self):
        lam = 1.
        use_cutmix = True
        if self.mixup_enabled and np.random.rand() < self.mix_prob:
            if self.mixup_alpha > 0:
                lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            else:
                assert False, "One of mixup_alpha > 0., cutmix_alpha > 0., cutmix_minmax not None should be true."
            lam = float(lam_mix)
        return lam, use_cutmix
    def _mix_batch_collate(self, batch):
        lam, use_cutmix = self._params_per_batch()
        if lam == 1.:
            return 1.
        if use_cutmix:
            bbx1, bbx2 = rand_bbox(batch.shape, lam)
            batch[..., bbx1:bbx2] = batch.flip(0)[..., bbx1:bbx2]
        return lam, (bbx1, bbx2)

    def __call__(self, batch, target, return_box=False):
        batch_size = len(batch)

        assert batch_size % 2 == 0, 'Batch size should be even when using this'
        half = 'half' in self.mode
        if half:
            batch_size //= 2
        lam, (bbx1, bbx2) = self._mix_batch_collate(batch)

        torch.set_printoptions(threshold=np.inf)
        # rank_zero_info(f'target1 {target}')
        target = mixup_target(target, self.num_classes, lam, self.label_smoothing, batch.device)
        # rank_zero_info(f'target2 {target}')
        if return_box==True:
            return batch, target, (bbx1, bbx2)
        else:
            return batch, target