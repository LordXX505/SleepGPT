import torch
from torchmetrics import Metric
from torchmetrics import ConfusionMatrix

confmat = ConfusionMatrix

class Accuracy(Metric):
    """
    - logits 形状:
        * 二分类:  (B,) 或 (B, 1)   => 使用 sigmoid+0.5 判断
        * 多分类:  (B, C) (C>=2)    => 使用 argmax(dim=-1)
    - target 形状:
        * 二分类:  (B,)      (0/1)
        * 多分类:  (B,)      (类别索引) 或 one-hot (B,C)
    """

    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total",   default=torch.tensor(0.), dist_reduce_fx="sum")

    def _binary_pred(self, logit: torch.Tensor) -> torch.Tensor:
        """sigmoid→阈值 0.5 → {0,1}"""
        return (logit.sigmoid() > 0.5).long()

    def update(self, logits: torch.Tensor, target: torch.Tensor):
        logits, target = logits.detach().to(self.correct.device), target.detach().to(self.correct.device)

        # -------- 1. 处理 target 可能是 one-hot ----------
        if target.ndim == 2 and target.size(-1) > 1:
            target = target.argmax(dim=-1)

        # -------- 2. 生成 preds ----------
        # 二分类 logit: shape (B,) or (B,1)
        if logits.ndim == 1 or logits.size(-1) == 1:
            preds = self._binary_pred(logits.view(-1))
        else:                         # 多分类
            preds = logits.argmax(dim=-1)

        # -------- 3. 累积 ----------
        self.correct += (preds == target).sum()
        self.total   += target.numel()

    def compute(self):
        return self.correct / self.total


class Scalar(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("scalar", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, scalar):
        if isinstance(scalar, torch.Tensor):
            scalar = scalar.detach().to(self.scalar.device)
        else:
            scalar = torch.tensor(scalar).float().to(self.scalar.device)

        self.scalar += scalar
        self.total += 1

    def compute(self):
        return self.scalar / self.total

class ACC(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("scalar", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, scalar):
        if isinstance(scalar, torch.Tensor):
            scalar = scalar.detach().to(self.scalar.device)
        else:
            scalar = torch.tensor(scalar).float().to(self.scalar.device)

        self.scalar += scalar
    def compute(self):
        return self.scalar


class ChannelwiseScalar(Metric):
    def __init__(self, n_channels, dist_sync_on_step=False, need_sum=True):

        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.n_channels = n_channels

        self.add_state("sums", default=torch.zeros(n_channels), dist_reduce_fx="sum")
        self.add_state("channel_counts", default=torch.zeros(n_channels), dist_reduce_fx="sum")
        self.need_sum = need_sum
    def update(self, scalar: torch.Tensor):
        if self.need_sum:
            self.sums += torch.sum(scalar, dim=0)  # along batch dim
            self.channel_counts += scalar.size(0)
        else:
            assert len(self.sums) == len(scalar), f'sums: {len(self.sums)}, scalar: {len(scalar)}'
            self.sums += scalar
            self.channel_counts += 1

    def compute(self):

        return self.sums / self.channel_counts