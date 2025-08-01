import copy
import glob
import os
import sys
import torch.nn.functional as F
import torch.nn as nn
from pytorch_lightning.utilities.rank_zero import rank_zero_info
import torch
import time
from typing import List
import datetime
from main.gadgets.my_metrics import Accuracy, Scalar, confmat, ACC, ChannelwiseScalar
from scipy.optimize import linear_sum_assignment
import numpy as np


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


def one_hot(x, num_classes, on_value=1., off_value=0., device='cuda'):
    x = x.long().view(-1, 1)
    return torch.full((x.size()[0], num_classes), off_value, device=device).scatter_(1, x, on_value)


class MultiFocalLoss(nn.Module):
    def __init__(self, alpha=0.0, gamma=2, reduction='none'):
        super(MultiFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        B = inputs.shape[0]
        inputs = inputs.float().view(-1, 1)
        targets = targets.float().view(-1, 1)
        p = inputs
        if self.alpha > 0:
            alpha_t = self.alpha
            alpha_t_0 = 1.0
        else:
            alpha_t = 1.0
            alpha_t_0 = 1.0
        ce = BCELoss_class_weighted(reduction='none', weights=torch.tensor([alpha_t_0, alpha_t]))
        ce_loss = ce(inputs, targets)
        p_t = p * targets + (1 - p) * (1 - targets)
        coef = (1 - p_t) ** self.gamma
        # rank_zero_info(f'ce_loss: {ce_loss}')
        loss = ce_loss * coef.squeeze()
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "none":
            loss = loss.reshape(B, -1)
            assert loss.shape[1] != 1
            loss = loss.mean(1)
        return loss


class Dice_loss(nn.Module):
    def __init__(self, beta=1, smooth=1e-5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, target):
        n, h = inputs.size()
        nt, ht = target.size()
        assert not ((inputs > 1).any() or (inputs < 0).any()), f'input has an error. inputs > 1: {inputs[inputs > 1]}, ' \
                                                               f'inputs < 0: {inputs[inputs < 0]}'
        if h != ht:
            inputs = F.interpolate(inputs, size=(ht), align_corners=True)
        tp = torch.sum(target * inputs, dim=1)
        fp = torch.sum(inputs, dim=1) - tp
        fn = torch.sum(target, dim=1) - tp
        score = ((1 + self.beta ** 2) * tp + self.smooth) / (
                (1 + self.beta ** 2) * tp + self.beta ** 2 * fn + fp + self.smooth)
        # index = torch.isnan(score)
        # rank_zero_info(f'tp: {tp[index]}, fp:{fp[index]}, fn: {fn[index]}, target:{target[index]}, inputs: {inputs[index]}')
        # if torch.sum(index) != 0:
        #     sys.exit(0)
        dice_loss = 1 - score
        return dice_loss


class BCELoss_class_weighted(nn.Module):
    def __init__(self, weights, reduction=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert weights.shape[0] == 2
        self.weights = weights
        self.reduction = reduction

    def loss(self, input, target):
        input = torch.clamp(input, min=1e-7, max=1 - 1e-7)
        bce = - self.weights[1] * target * torch.clamp(torch.log(input), min=-100) - \
            (1 - target) * self.weights[0] * torch.clamp(torch.log(1 - input), min=-100)
        if self.reduction == 'none':
            return torch.mean(bce, dim=-1)
        else:
            return torch.mean(bce)

    def forward(self, input, target):
        return self.loss(input, target)


class Fpfn_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        sum1 = torch.sum(target, -1)
        sum2 = torch.sum(1 - target, -1)
        loss = torch.clamp(torch.sum(output * (1 - target), -1) / (sum2 + 1e-6), max=100) + \
               torch.clamp(torch.sum((1 - output) * target, -1) / (sum1 + 1e-6), max=100)
        return loss


class Fpfn(nn.Module):

    def __init__(self, use_fpfn='Fpfn'):
        super(Fpfn, self).__init__()
        self.use_fpfn = use_fpfn

    def _get_next_version(self, root_dir) -> int:
        import os
        save_dir = root_dir

        listdir_info = os.listdir(save_dir)

        existing_versions = []
        for listing in listdir_info:
            d = listing
            bn = os.path.basename(d)
            if bn.startswith("version_"):
                dir_ver = bn.split("_")[1].replace("/", "")
                existing_versions.append(dir_ver)
        if len(existing_versions) == 0:
            return 0

        return len(existing_versions) + 1

    def _get_loss(self, output, target, weight):
        res = {}
        if self.use_fpfn == 'Fpfn':
            fpfn = Fpfn_loss()
            loss = fpfn(output, target)
            res.update({'loss': loss})
        elif self.use_fpfn == 'BCE':
            assert weight is not None, f'Using BCE loss but weight is None.'
            weight = torch.tensor([1, int(weight)], device=output.device)
            ce = BCELoss_class_weighted(weights=weight, reduction='none')
            loss = ce(output, target)
            res.update({'loss': loss})
        elif self.use_fpfn == 'Dice_BCE':
            assert weight is not None, f'Using BCE loss but weight is None.'
            weight = torch.tensor([1, int(weight)], device=output.device)
            ce = BCELoss_class_weighted(weights=weight, reduction='none')
            ce_loss = ce(output, target)
            dc_loss = Dice_loss()
            dice_loss = dc_loss(output, target)
            loss = 0.5 * ce_loss + 0.5 * dice_loss
            res.update({'loss': loss, 'dice_loss': dice_loss, 'ce_loss': {ce_loss}})
        elif self.use_fpfn == 'Dice':
            assert weight is not None, f'Using BCE loss but weight is None.'
            dc_loss = Dice_loss()
            dice_loss = dc_loss(output, target)
            loss = dice_loss
            res.update({'loss': loss})
        elif self.use_fpfn == 'Focal':
            focal = MultiFocalLoss(alpha=weight)
            focal_loss = focal(output, target)
            loss = focal_loss
            res.update({'loss': loss})
        elif self.use_fpfn == 'Focal_Fpfn':
            focal = MultiFocalLoss(alpha=weight)
            focal_loss = focal(output, target)
            fpfn = Fpfn_loss()
            fpfn_loss = fpfn(output, target)
            loss = 0.5 * focal_loss + 0.5 * fpfn_loss
            res.update({'loss': loss, 'focal_loss': {focal_loss}, 'fpfn': {fpfn_loss}})
        elif self.use_fpfn == 'BCE_Fpfn':
            weight = torch.tensor([1, int(weight)], device=output.device)
            ce = BCELoss_class_weighted(weights=weight, reduction='none')
            ce_loss = ce(output, target)
            fpfn = Fpfn_loss()
            fpfn_loss = fpfn(output, target)
            loss = 0.5 * ce_loss + 0.5 * fpfn_loss
            res.update({'loss': loss, 'ce_loss': {ce_loss}, 'fpfn': {fpfn_loss}})
        elif self.use_fpfn == 'BCE_Fpfn_Dice':
            weight = torch.tensor([1, int(weight)], device=output.device)
            ce = BCELoss_class_weighted(weights=weight, reduction='none')
            ce_loss = ce(output, target)
            fpfn = Fpfn_loss()
            fpfn_loss = fpfn(output, target)
            dc_loss = Dice_loss()
            dice_loss = dc_loss(output, target)
            loss = 0.5 * ce_loss + 0.5 * fpfn_loss + 0.5*dice_loss
            res.update({'loss': loss, 'ce_loss': {ce_loss}, 'fpfn': {fpfn_loss}, 'dice_loss': dice_loss})
        else:
            loss = None
            res.update({'loss': loss})
        return res

    def forward(self, output: torch.Tensor, target: torch.Tensor, data_idx=None, store=False,
                weight=None, stage='fit', path_name=None):
        target = target.float()
        res = self._get_loss(output, target, weight)
        loss = res['loss']
        # if store is True:
        #     rootdir = f'/home/cuizaixu_lab/huangweixuan/DATA/ver_log/{path_name}'
        #     idx = torch.where(loss > 1.1)[0]
        #     # rank_zero_info(f'idx: {idx.shape}, {idx}')
        #     if idx.shape[0] != 0:
        #         import numpy as np
        #         torch.set_printoptions(threshold=np.inf)
        #         os.makedirs(rootdir, exist_ok=True)
        #         version = self._get_next_version(rootdir)
        #         torch.save({"output": output[idx].detach().cpu(), "target": target[idx].detach().cpu(),
        #                     "idx": data_idx, 'loss': loss.detach().cpu()},
        #                    f'{rootdir}/version_{version}_{stage}_1.ckpt')
        #         rank_zero_info(f'Saving stage: {stage} result in {rootdir}/version_{version}_{stage}_1')
        #     idx = torch.where(loss < 0.5)[0]
        #     if idx.shape[0] != 0:
        #         import numpy as np
        #         torch.set_printoptions(threshold=np.inf)
        #         os.makedirs(rootdir, exist_ok=True)
        #         version = self._get_next_version(rootdir)
        #         torch.save(
        #             {"output": output[idx].detach().cpu(),
        #              "target": target[idx].detach().cpu(),
        #              "idx": data_idx, 'loss': loss.detach().cpu()},
        #             f'{rootdir}/version_{version}_{stage}_0_5.ckpt')
        #         rank_zero_info(f'Saving stage: {stage}  result in {rootdir}/version_{version}_{stage}_0_5')
        assert loss.ndim == 1
        return torch.mean(loss, dim=0), res


class Event(nn.Module):

    def __init__(self, threshold, device, freq=100, time=0.5, test=False):
        super(Event, self).__init__()
        self.threshold = threshold
        self.freq = freq
        self.time = time
        self.len_threshold = int(freq * time)
        # rank_zero_info(f'freq:{self.freq}, time:{self.time}, len_threshold:{self.len_threshold}')
        self.device = device
        self.test = test

    def get_event(self, seq, processingpost=None, prob=True):
        if isinstance(seq, list):
            seq = torch.tensor(seq)
        assert len(seq.shape) == 2
        predicted_events = self._get_event_n(seq, prob)
        if self.test:
            for item in predicted_events:
                print(f'len predicted_events: {len(item)}')
        if processingpost is not None:
            processingpost(seq, predicted_events, self.len_threshold, device=self.device)
            predicted_events = self._get_event_n(seq, prob)
            if self.test:
                for item in predicted_events:
                    print(f'after processingpost: len predicted_events: {len(item)}')
        return predicted_events
        # lf, rt, belong, group, unused = self._get_event(seq, prob)
        # if processingpost is not None:
        #     processingpost(seq, lf, rt, belong, group, self.len_threshold)
        #     lf, rt, belong, group, unused = self._get_event(seq, prob)
        # return lf, rt, belong, group, unused

    def _check(self, prob, item):
        if prob:
            return item.item() > self.threshold
        else:
            return item.item() == 1

    def _get_event_n(self, seq, prob=True):
        if prob:
            tmp = prob_to_binary(seq, self.threshold)
            predicted_events = [binary_to_array(k) for k in tmp]
            return predicted_events
        else:
            predicted_events = [binary_to_array(k) for k in seq]
            return predicted_events

    def _get_event(self, seq, prob=True):
        header = 'test'
        # start_time = time.time()

        belong = torch.zeros(seq.shape, dtype=torch.int).to(self.device)
        lf = torch.zeros(seq.shape, dtype=torch.int).to(self.device)
        rt = torch.zeros(seq.shape, dtype=torch.int).to(self.device)
        group = torch.zeros(seq.shape[0], dtype=torch.int).to(self.device)
        unused = torch.zeros(seq.shape, dtype=torch.int).to(self.device)

        # total_time = time.time() - start_time
        # total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        # print('Test: {} Get storage Total time: {} )'.format(
        #     header, total_time_str))
        # start_time = time.time()

        for batch_iter, batch in enumerate(seq):
            index = 0
            for _, item in enumerate(batch):

                if self._check(prob, item):
                    if _ == 0:
                        index = 1
                        lf[batch_iter][index] = _
                    elif belong[batch_iter][_ - 1].item() == 0:
                        index += 1
                        lf[batch_iter][index] = _
                    belong[batch_iter][_] = index
                    rt[batch_iter][index] = _
                else:
                    belong[batch_iter][_] = 0
            group[batch_iter] = index
        # total_time = time.time() - start_time
        # total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        # print('Test: {} Batch Total time: {} )'.format(
        #     header, total_time_str))
        return lf, rt, belong, group, unused


def ProcessingPost(seq, lf, rt, belong, group, len_threshold):
    for batch_iter in range(seq.shape[0]):
        for index in range(1, group[batch_iter].item() + 1):
            lf_index = lf[batch_iter][index]
            rt_index = rt[batch_iter][index]
            if (rt_index - lf_index + 1) < len_threshold:
                seq[batch_iter][lf_index:rt_index + 1] = 0


def ProcessingPostEvent(seq, predicted_events: List[torch.Tensor], len_threshold, device):
    for batchiter, batch in enumerate(seq):
        if predicted_events[batchiter].shape[0] == 0:
            continue
        index = torch.where(predicted_events[batchiter][:, 1] -
                            predicted_events[batchiter][:, 0] < torch.tensor(len_threshold, device=device))[0]
        for indexx in predicted_events[batchiter][index]:
            seq[batchiter][indexx[0]: indexx[1]] = 0


class By_Event(nn.Module):

    def __init__(self, threshold, IOU_threshold, device='cpu', **kwargs):
        super(By_Event, self).__init__()
        self.threshold = threshold
        self.IOU_threshold = IOU_threshold
        # rank_zero_info(f'threshold: {self.threshold}, IOU_threshold:{self.IOU_threshold}')
        self.get_event = Event(threshold, device, **kwargs)
        self.device = device

    def jaccard_overlap(self, output, target):
        A = output.size(0)
        B = target.size(0)
        max_min = torch.max(output[:, 0].unsqueeze(1).expand(A, B),
                            target[:, 0].unsqueeze(0).expand(A, B))
        min_max = torch.min(output[:, 1].unsqueeze(1).expand(A, B),
                            target[:, 1].unsqueeze(0).expand(A, B))
        intersection = torch.clamp((min_max - max_min), min=0)
        lentgh_a = (output[:, 1] - output[:, 0]).unsqueeze(1).expand(A, B)
        lentgh_b = (target[:, 1] - target[:, 0]).unsqueeze(0).expand(A, B)
        overlaps = intersection / (lentgh_a + lentgh_b - intersection)
        return overlaps

    def best_match(self, max_iou_col, index_col, index_row, max_iou_row):
        # print(f'max_iou_col: {max_iou_col}, index_col: {index_col}, max_iou_row: {max_iou_row}, index_row: {index_row}', )
        one = 0
        col_len = index_col.shape[0]
        row_len = index_row.shape[0]
        bestmatch = torch.zeros((row_len, col_len), device=self.device)
        index_1 = torch.where((index_row[index_col[range(col_len)]] == torch.tensor(range(col_len), device=self.device))
                              & (max_iou_col[range(col_len)] >= torch.tensor(self.IOU_threshold, device=self.device)))[
            0]
        index_2 = torch.where((index_col[index_row[range(row_len)]] == torch.tensor(range(row_len), device=self.device))
                              & (max_iou_row[range(row_len)] >= torch.tensor(self.IOU_threshold, device=self.device)))[
            0]
        # print(index_1, index_2)
        index_1 = index_1.reshape(-1).to(self.device).long()
        index_2 = index_2.reshape(-1).to(self.device).long()
        index_1_true = torch.tensor([False] * col_len, dtype=torch.bool, device=self.device)
        index_2_true = torch.tensor([False] * row_len, dtype=torch.bool, device=self.device)
        index_1_true[index_1] = True
        index_2_true[index_2] = True

        index_2_true = torch.where((index_2_true == False)
                                   &
                                   (max_iou_row[range(row_len)] >=
                                    torch.tensor(self.IOU_threshold,
                                                 device=self.device)))[0]
        index_1_true = torch.where((index_1_true == False)
                                   &
                                   (max_iou_col[range(col_len)] >=
                                    torch.tensor(self.IOU_threshold,
                                                 device=self.device)))[0]
        # print(index_2_true, index_1_true)

        bestmatch[(index_2_true, index_row[index_2_true])] = 1
        bestmatch[(index_col[index_1_true], index_1_true)] = 1
        try:
            bestmatch = bestmatch.index_fill(dim=0, index=index_2, value=0)
            # bestmatch[index_2] = 0
        except Exception as e:
            print(f"Exception : {e}")
            print(index_2, index_2_true, index_2_true.shape, bestmatch)
            sys.exit(0)
        try:
            bestmatch = bestmatch.index_fill(dim=1, index=index_row[index_2], value=0)
            # bestmatch[:, index_row[index_2]] = 0
        except Exception as e:
            print(f"Exception : {e}, index_2: {index_2}, index_2_true:{index_2_true}, bestmatch: {bestmatch}")
            sys.exit(0)
        try:
            # bestmatch.index_put((index_2, index_row[index_2]), values=torch.tensor(2, device=self.device, dtype=torch.int64))
            bestmatch[(index_2, index_row[index_2])] = 2
        except Exception as e:
            print(f"Exception : {e}, index_2: {index_2}, index_2_true:{index_2_true}, bestmatch: {bestmatch}")
            sys.exit(0)
        TP = (bestmatch == 2).sum().item()

        # print('del: ', bestmatch)
        res = torch.where(bestmatch != 1)
        one = (bestmatch == 1).sum().item()
        # print(one)
        # print(TP)

        return TP, res, one, index_2, index_row[index_2]

    def forward(self, output: torch.Tensor, target: torch.Tensor, device='cuda'):
        TP = 0.0
        FN = 0.0
        FP = 0.0

        # print(output.shape)
        len = output.shape[1]

        assert output.shape == target.shape, f'output shape:{output.shape}, target shape: {target.shape}'

        predicted_events_output = self.get_event.get_event(output, processingpost=ProcessingPostEvent, prob=True)
        predicted_events_target = self.get_event.get_event(target, prob=False)
        header = 'Test:'
        start_time = time.time()
        for i in range(output.shape[0]):
            output_item = predicted_events_output[i]
            target_item = predicted_events_target[i]
            torch.set_printoptions(threshold=np.inf)
            # print('predicted_events_output: ', output_item)
            # print('predicted_events_target: ', target_item)
            res_row = []
            res_col = []
            if target_item.shape[0] == 0:
                FP += output_item.shape[0]
                continue
            elif output_item.shape[0] == 0:
                FN += target_item.shape[0]
                continue
            iou = self.jaccard_overlap(output_item, target_item)
            max_iou_col, index_col = iou.max(0)
            max_iou_row, index_row = iou.max(1)
            true_positive, one_index, one, choose_row, choose_col = self.best_match(max_iou_col, index_col, index_row, max_iou_row)
            res_row.append(choose_row)
            res_col.append(choose_col)
            if one > 0:
                iou[one_index] = 0
                max_iou_col, index_col = iou.max(0)
                max_iou_row, index_row = iou.max(1)
                true_positive_, one_index, one, choose_row, choose_col = self.best_match(max_iou_col, index_col, index_row, max_iou_row)
                true_positive += true_positive_
                res_row.append(choose_row)
                res_col.append(choose_col)
            false_positive = output_item.shape[0] - true_positive
            false_negative = target_item.shape[0] - true_positive
            TP += true_positive
            FN += false_negative
            FP += false_positive
            res_row = torch.cat(res_row)
            res_col = torch.cat(res_col)
            # print('result output target is: ')
            # for output, target in zip(output_item[res_row], target_item[res_col]):
            #     print(output, target)
        Recall = cal_Recall(TP, FN)
        # print(Recall)
        Precision = cal_Precision(TP, FP)
        # print(Precision)
        # return Recall, Precision, cal_F1_score(Precision, Recall)
        return TP, FN, FP,

class By_Event_Bipartite(nn.Module):
    def __init__(self, threshold, IOU_threshold, device='cpu', **kwargs):
        super(By_Event_Bipartite, self).__init__()
        self.threshold = threshold
        self.IOU_threshold = IOU_threshold
        # rank_zero_info(f'threshold: {self.threshold}, IOU_threshold:{self.IOU_threshold}')
        self.get_event = Event(threshold, device, **kwargs)
        self.device = device

    def jaccard_overlap(self, output, target):
        A = output.size(0)
        B = target.size(0)
        max_min = torch.max(output[:, 0].unsqueeze(1).expand(A, B),
                            target[:, 0].unsqueeze(0).expand(A, B))
        min_max = torch.min(output[:, 1].unsqueeze(1).expand(A, B),
                            target[:, 1].unsqueeze(0).expand(A, B))
        intersection = torch.clamp((min_max - max_min), min=0)
        lentgh_a = (output[:, 1] - output[:, 0]).unsqueeze(1).expand(A, B)
        lentgh_b = (target[:, 1] - target[:, 0]).unsqueeze(0).expand(A, B)
        overlaps = intersection / (lentgh_a + lentgh_b - intersection)
        return overlaps
    def forward(self, output: torch.Tensor, target: torch.Tensor, device='cuda'):
        TP = 0.0
        FN = 0.0
        FP = 0.0
        predicted_events_output = self.get_event.get_event(output, processingpost=ProcessingPostEvent, prob=True)
        predicted_events_target = self.get_event.get_event(target, prob=False)
        for i in range(output.shape[0]):
            output_item = predicted_events_output[i]
            target_item = predicted_events_target[i]
            # print('predicted_events_output: ', output_item)
            # print('predicted_events_target: ', target_item)

            if target_item.shape[0] == 0:
                FN += output_item.shape[0]
                continue
            elif output_item.shape[0] == 0:
                FP += target_item.shape[0]
                continue
            iou = -1 * self.jaccard_overlap(output_item, target_item)
            indices = linear_sum_assignment(iou)
            print(indices)

def cal_IOU(overlap, left, right):
    return (1.0 * overlap / (right - left + 1)).item()


def cal_Recall(TP, FN):
    if TP + FN == 0:
        return 0.0
    return 1.0 * TP / (TP + FN)


def cal_Precision(TP, FP):
    if TP + FP == 0:
        return 0.0
    return 1.0 * TP / (TP + FP)


def cal_F1_score(Precision, Recall):
    if Precision + Recall == 0:
        return 0.0
    return 2.0 * Precision * Recall / (Precision + Recall)


def prob_to_binary(x, threshold):
    """ Return [0,1,0,1] from prob array
        """
    tmp = (x >= threshold).detach().clone().to(torch.int)
    return tmp


def binary_to_array(x):
    """ Return [start, duration] from binary array

    binary_to_array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1])
    [[4, 8], [11, 13]]
    """
    # tmp = torch.tensor([0] + list(x) + [0])
    device = x.device
    tmp = torch.cat((torch.tensor([0], device=device), x, torch.tensor([0], device=device)))
    return torch.where((tmp[1:] - tmp[:-1]) != 0)[0].reshape((-1, 2))


def set_metrics(pl_module, **kwargs):
    if pl_module.visual is not True or pl_module.visual_mode == 'UMAP':
        for split in ["train", "validation", "test"]:
            for k, v in pl_module.hparams.config["loss_names"].items():
                if v < 1:
                    continue
                if k == "Spnidle":
                    if split == "train":
                        setattr(pl_module, f"train_{k}_loss", Scalar())
                        setattr(pl_module, f"train_{k}_TP", ACC())
                        setattr(pl_module, f"train_{k}_FN", ACC())
                        setattr(pl_module, f"train_{k}_FP", ACC())
                    else:
                        setattr(pl_module, f"validation_{k}_loss", Scalar())
                        setattr(pl_module, f"validation_{k}_TP", ACC())
                        setattr(pl_module, f"validation_{k}_FN", ACC())
                        setattr(pl_module, f"validation_{k}_FP", ACC())
                        setattr(pl_module, f"validation_{k}_Precision", Scalar())
                        setattr(pl_module, f"validation_{k}_Recall", Scalar())
                        setattr(pl_module, f"validation_{k}_F1", Scalar())
                        setattr(pl_module, f"test_{k}_loss", Scalar())
                        setattr(pl_module, f"test_{k}_TP", ACC())
                        setattr(pl_module, f"test_{k}_FN", ACC())
                        setattr(pl_module, f"test_{k}_FP", ACC())
                        setattr(pl_module, f"test_{k}_Precision", Scalar())
                        setattr(pl_module, f"test_{k}_Recall", Scalar())
                        setattr(pl_module, f"test_{k}_F1", Scalar())
                elif k == "Apnea":
                    if split == "train":
                        setattr(pl_module, f"train_{k}_loss", Scalar())
                        setattr(pl_module, f"train_{k}_TP", ACC())
                        setattr(pl_module, f"train_{k}_FN", ACC())
                        setattr(pl_module, f"train_{k}_FP", ACC())
                    else:
                        setattr(pl_module, f"validation_{k}_loss", Scalar())
                        setattr(pl_module, f"validation_{k}_TP", ACC())
                        setattr(pl_module, f"validation_{k}_FN", ACC())
                        setattr(pl_module, f"validation_{k}_FP", ACC())
                        setattr(pl_module, f"validation_{k}_Precision", Scalar())
                        setattr(pl_module, f"validation_{k}_Recall", Scalar())
                        setattr(pl_module, f"validation_{k}_F1", Scalar())
                        setattr(pl_module, f"test_{k}_loss", Scalar())
                        setattr(pl_module, f"test_{k}_TP", ACC())
                        setattr(pl_module, f"test_{k}_FN", ACC())
                        setattr(pl_module, f"test_{k}_FP", ACC())
                        setattr(pl_module, f"test_{k}_Precision", Scalar())
                        setattr(pl_module, f"test_{k}_Recall", Scalar())
                        setattr(pl_module, f"test_{k}_F1", Scalar())
                elif k == "CrossEntropy":
                    multi_y = copy.deepcopy(pl_module.multi_y)
                    if pl_module.local_pooling:
                        multi_y.append('local')
                    if split == "train":
                        setattr(pl_module, f"train_{k}_loss", Scalar())
                        setattr(pl_module, f"train_{k}_local_loss", Scalar())
                        setattr(pl_module, f"train_{k}_local_accuracy_tf", Accuracy())
                        setattr(pl_module, f"train_{k}_local_conf_tf", confmat(task="multiclass", num_classes=kwargs['num_classes']))
                        for name in multi_y:
                            setattr(pl_module, f"train_{k}_accuracy_{name}", Accuracy())
                            setattr(pl_module, f"train_{k}_conf_{name}", confmat(task="multiclass", num_classes=kwargs['num_classes']))
                        if pl_module.spo2_ods_settings.get('inj', False):
                            setattr(pl_module, f"train_{k}_accuracy_spo2_cls_feats", Accuracy())
                            setattr(pl_module, f"train_{k}_conf_spo2_cls_feats", confmat(task="binary", num_classes=2))
                    else:
                        if 'persub' in kwargs and kwargs['persub'] is True:
                            test_names = kwargs['test_sub_names']
                            for tn in test_names.values():
                                _tn = tn.split('/')[-1]
                                setattr(pl_module, f"test_{_tn}_conf", confmat(task="multiclass", num_classes=kwargs['num_classes']))
                        if pl_module.return_alpha is True:
                            for stage in [0, 1, 2, 3, 4]:
                                if pl_module.transformer.actual_channels is not None:
                                    setattr(pl_module, f"{stage}_mapping", ChannelwiseScalar(pl_module.transformer.actual_channels.shape[0], need_sum=False))
                                else:
                                    setattr(pl_module, f"{stage}_mapping", ChannelwiseScalar(8, need_sum=False))
                        setattr(pl_module, f"test_{k}_loss", Scalar())
                        setattr(pl_module, f"validation_{k}_loss", Scalar())
                        if pl_module.spo2_ods_settings.get('inj', False):
                            setattr(pl_module, f"validation_{k}_accuracy_spo2_cls_feats", Accuracy())
                            setattr(pl_module, f"test_{k}_accuracy_spo2_cls_feats", Accuracy())
                            setattr(pl_module, f"validation_{k}_conf_spo2_cls_feats", confmat(task="binary", num_classes=2))
                            setattr(pl_module, f"test_{k}_conf_spo2_cls_feats", confmat(task="binary", num_classes=2))
                        for name in multi_y:
                            setattr(pl_module, f"validation_{k}_accuracy_{name}", Accuracy())
                            setattr(pl_module, f"test_{k}_accuracy_{name}", Accuracy())
                            setattr(pl_module, f"test_{k}_conf_{name}", confmat(task="multiclass", num_classes=kwargs['num_classes']))
                            setattr(pl_module, f"validation_{k}_conf_{name}", confmat(task="multiclass", num_classes=kwargs['num_classes']))
                elif k == "Pathology":
                    if split == "train":
                        setattr(pl_module, f"train_{k}_loss", Scalar())
                        setattr(pl_module, f"train_{k}_accuracy", Accuracy())
                        setattr(pl_module, f"train_{k}_conf", confmat(task="multiclass", num_classes=7))
                    else:
                        setattr(pl_module, f"test_{k}_loss", Scalar())
                        setattr(pl_module, f"test_{k}_accuracy", Accuracy())
                        setattr(pl_module, f"test_{k}_conf", confmat(task="multiclass", num_classes=7))
                        setattr(pl_module, f"validation_{k}_loss", Scalar())
                        setattr(pl_module, f"validation_{k}_accuracy", Accuracy())
                        setattr(pl_module, f"validation_{k}_conf", confmat(task="multiclass", num_classes=7))
                elif k == "mtm":
                    setattr(pl_module, f"{split}_{k}_loss2", Scalar())
                    setattr(pl_module, f"{split}_{k}_loss", Scalar())
                elif k == "itc":
                    if pl_module.time_only or pl_module.fft_only:
                        setattr(pl_module, f"{split}_{k}_w2s_accuracy", Accuracy())
                        setattr(pl_module, f"{split}_{k}_s2w_accuracy", Accuracy())
                        setattr(pl_module, f"{split}_{k}_w2s_mask_accuracy", Accuracy())
                        setattr(pl_module, f"{split}_{k}_s2w_mask_accuracy", Accuracy())

                        setattr(pl_module, f"{split}_{k}_loss", Scalar())
                        setattr(pl_module, f"{split}_{k}_logit_scale", Scalar())
                        setattr(pl_module, f"{split}_{k}_logit_mask_scale", Scalar())
                    else:
                        setattr(pl_module, f"{split}_{k}_f2t_accuracy", Accuracy())
                        setattr(pl_module, f"{split}_{k}_t2f_accuracy", Accuracy())
                        setattr(pl_module, f"{split}_{k}_loss", Scalar())
                        setattr(pl_module, f"{split}_{k}_logit_scale", Scalar())

                        setattr(pl_module, f"{split}_{k}_tf_f2t_accuracy", Accuracy())
                        setattr(pl_module, f"{split}_{k}_tf_t2f_accuracy", Accuracy())
                        setattr(pl_module, f"{split}_{k}_tf_logit_scale", Scalar())
                else:
                    setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                    setattr(pl_module, f"{split}_{k}_loss", Scalar())
    else:
        setattr(pl_module, f"test_loss", ChannelwiseScalar(8))
        setattr(pl_module, f"test_loss2", ChannelwiseScalar(8))
        if 'persub' in kwargs and kwargs['persub'] is True:
            test_names = kwargs['test_sub_names']
            for tn in test_names.values():
                print(f'tn : {tn}')
                _tn = tn.split('/')[-1]
                setattr(pl_module, f"test_{_tn}_loss", ChannelwiseScalar(8))
                setattr(pl_module, f"test_{_tn}_loss2", ChannelwiseScalar(8))

if __name__ == '__main__':
    import torch

    # ckpt = torch.load('../../data/case.ckpt', map_location=torch.device('cpu'))
    # cls = ckpt['cls']
    # spindle = ckpt['Spindle_label']

    # path_list = glob.glob('../../data/ver_log/*')
    # for path in path_list:
    #     ckpt = torch.load(path, map_location=torch.device('cpu'))
    #     import matplotlib.pyplot as plt
    #
    #     x = range(2000)
    #     cls = ckpt['output']
    #     spindle = ckpt['Spindle_label']
    #     for i in range(cls):
    #         plt.plot(x, spindle[i], c='r')
    #         plt.plot(x, cls[i], c='b')
    #         plt.legend()
    #         plt.show()
    cls = torch.zeros([2, 100])
    cls[0, 0:9] = 1
    cls[0, 20:30] = 1
    cls[0, 40:50] = 1
    cls[0, 55:65] = 1
    spindle = torch.zeros([2, 100])
    spindle[0, 5:15] = 1
    spindle[0, 25:45] = 1
    fpfn = Fpfn()
    loss = fpfn(cls, spindle)
    by_e = By_Event(threshold=0.55, IOU_threshold=0.2, device=cls.device, freq=10, time=1)
    TP, FN, FP = by_e(cls.detach().clone(), spindle.detach().clone())
    print(TP, FN, FP)
    Recall = cal_Recall(TP, FN)
    Precision = cal_Precision(TP, FP)
    print(Recall, Precision, cal_F1_score(Precision, Recall))
