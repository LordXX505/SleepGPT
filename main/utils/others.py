import copy

import torch.nn as nn
from pytorch_lightning.utilities.rank_zero import rank_zero_info
import torch
import time
from typing import List
import datetime
from main.gadgets.my_metrics import Accuracy, Scalar, confmat, ACC


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


class Fpfn(nn.Module):

    def __init__(self):
        super(Fpfn, self).__init__()

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        sum1 = torch.sum(target, -1)
        sum2 = torch.sum(1 - target, -1)
        # print(sum1, sum2)
        # print(torch.sum(output * (1 - target), -1), torch.sum((1 - output) * target, -1))
        weight = [1, 10]
        loss = torch.sum(output[..., 0] * (1 - target), -1) / sum2 + torch.sum(output[..., 1] * target, -1) / sum1
        # print(loss)
        # print(torch.mean(loss))
        return torch.mean(loss)


class Event(nn.Module):

    def __init__(self, threshold, device, freq=100, time=0.5):
        super(Event, self).__init__()
        self.threshold = threshold
        self.freq = freq
        self.time = time
        self.len_threshold = int(freq * time)
        self.device = device

    def get_event(self, seq, processingpost=None, prob=True):
        if isinstance(seq, list):
            seq = torch.tensor(seq)
        assert len(seq.shape) == 2
        predicted_events = self._get_event_n(seq, prob)
        if processingpost is not None:
            processingpost(seq, predicted_events, self.len_threshold, device=self.device)
            predicted_events = self._get_event_n(seq, prob)
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
                              & (max_iou_col[range(col_len)] >= torch.tensor(self.IOU_threshold, device=self.device)))[0]
        index_2 = torch.where((index_col[index_row[range(row_len)]] == torch.tensor(range(row_len), device=self.device))
                              & (max_iou_row[range(row_len)] >= torch.tensor(self.IOU_threshold, device=self.device)))[0]
        # print(index_1, index_2)
        index_1 = index_1.reshape(-1).to(self.device)
        index_2 = index_2.reshape(-1).to(self.device)
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
        # print(index_2, index_2_true, index_2_true.shape, bestmatch.shape, bestmatch.device, bestmatch[0])
        bestmatch[index_2, :] = 0
        bestmatch[:, index_row[index_2]] = 0
        bestmatch[(index_2, index_row[index_2])] = 2
        TP = (bestmatch == 2).sum().item()

        # print('del: ', bestmatch)
        res = torch.where(bestmatch != 1)
        one = (bestmatch == 1).sum().item()
        # print(one)
        # print(TP)

        return TP, res, one

    def forward(self, output: torch.Tensor, target: torch.Tensor, device='cuda'):
        TP = 0.0
        FN = 0.0
        FP = 0.0
        # print(output.shape)
        len = output.shape[1]
        output = output[..., 0]
        assert output.shape == target.shape

        predicted_events_output = self.get_event.get_event(output, processingpost=ProcessingPostEvent, prob=True)
        predicted_events_target = self.get_event.get_event(target, prob=False)
        header = 'Test:'
        start_time = time.time()
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
            iou = self.jaccard_overlap(output_item, target_item)
            max_iou_col, index_col = iou.max(0)
            max_iou_row, index_row = iou.max(1)
            true_positive, one_index, one = self.best_match(max_iou_col, index_col, index_row, max_iou_row)
            if one > 0:
                iou[one_index] = 0
                max_iou_col, index_col = iou.max(0)
                max_iou_row, index_row = iou.max(1)
                true_positive_, one_index, one = self.best_match(max_iou_col, index_col, index_row, max_iou_row)
                true_positive += true_positive_

            false_positive = output_item.shape[0] - true_positive
            false_negative = target_item.shape[0] - true_positive
            TP += true_positive
            FN += false_negative
            FP += false_positive
        # print(TP, FN, FP)
        # total_time = time.time() - start_time
        # total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        # print('Test: {} test_target Total time: {} )'.format(
        #     header, total_time_str))
        # header = 'Test:'
        # start_time = time.time()
        # lf, rt, belong, group, unused = self.get_event.get_event(target, prob=False)
        # total_time = time.time() - start_time
        # total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        # print('Test: {} test_target Total time: {} )'.format(
        #     header, total_time_str))
        #
        # start_time = time.time()
        # output_lf, output_rt, output_belong, output_group, output_unused = \
        #     self.get_event.get_event(output, processingpost=ProcessingPost, prob=True)
        # total_time = time.time() - start_time
        # total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        # print('Test: {} test_output Total time: {} )'.format(
        #     header, total_time_str))
        #
        # start_time = time.time()
        # for batch_iter, batch in enumerate(output):
        #     for indexx in range(1, output_group[batch_iter] + 1):
        #         overlap = {}
        #         # print("indexx: ", indexx)
        #         # print("output_lf: {}, output_rt:{}".format(output_lf[batch_iter][indexx],
        #         #                                            output_rt[batch_iter][indexx]))
        #         for i in range(output_lf[batch_iter][indexx], output_rt[batch_iter][indexx] + 1):
        #             belong_index = belong[batch_iter][i].item()
        #             if belong_index == 0:
        #                 continue
        #             if belong_index not in overlap:
        #                 overlap[belong_index] = 0
        #             overlap[belong_index] += 1
        #         # print(overlap)
        #         max_IOU = self.IOU_threshold
        #         max_item = None
        #         count = 0
        #         for k, v in overlap.items():
        #             IOU = cal_IOU(v, min(lf[batch_iter][k], output_lf[batch_iter][indexx]), max(rt[batch_iter][k], output_rt[batch_iter][indexx]))
        #             if IOU >= self.IOU_threshold:
        #                 count += 1
        #                 if IOU >= max_IOU:
        #                     max_IOU = IOU
        #                     max_item = (k, v, IOU)
        #         if max_item is not None:
        #             if unused[batch_iter][max_item[0]].item() == 0:
        #                 unused[batch_iter][max_item[0]] = 1
        #                 TP += 1
        #         else:
        #             FP += 1
        # total_time = time.time() - start_time
        # total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        # print('Test: {} test_all Total time: {} )'.format(
        #     header, total_time_str))
        #
        # for batch in unused:
        #     for item in batch[1:]:
        #         if item.item() == 0:
        #             FN += 1
        # print(TP, FN, FP)
        Recall = cal_Recall(TP, FN)
        # print(Recall)
        Precision = cal_Precision(TP, FP)
        # print(Precision)
        # return Recall, Precision, cal_F1_score(Precision, Recall)
        return TP, FN, FP

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


def set_metrics(pl_module, ):
    for split in ["train", "validation", "test"]:
        for k, v in pl_module.hparams.config["loss_names"].items():
            if v < 1:
                continue
            if k == "FpFn":
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
                    setattr(pl_module, f"test_{k}_loss", Scalar())
                    setattr(pl_module, f"test_{k}_TP", ACC())
                    setattr(pl_module, f"test_{k}_FN", ACC())
                    setattr(pl_module, f"test_{k}_FP", ACC())
            elif k == "CrossEntropy":
                multi_y = copy.deepcopy(pl_module.multi_y)
                if pl_module.local_pooling:
                    multi_y.append('local')
                if split == "train":
                    setattr(pl_module, f"train_{k}_loss", Scalar())
                    setattr(pl_module, f"train_{k}_local_loss", Scalar())
                    setattr(pl_module, f"train_{k}_local_accuracy_tf", Accuracy())
                    setattr(pl_module, f"train_{k}_local_conf_tf", confmat(task="multiclass", num_classes=5))

                    for name in multi_y:
                        setattr(pl_module, f"train_{k}_accuracy_{name}", Accuracy())
                        setattr(pl_module, f"train_{k}_conf_{name}", confmat(task="multiclass", num_classes=5))

                else:
                    setattr(pl_module, f"test_{k}_loss", Scalar())
                    setattr(pl_module, f"validation_{k}_loss", Scalar())
                    for name in multi_y:
                        setattr(pl_module, f"validation_{k}_accuracy_{name}", Accuracy())
                        setattr(pl_module, f"test_{k}_accuracy_{name}", Accuracy())
                        setattr(pl_module, f"test_{k}_conf_{name}", confmat(task="multiclass", num_classes=5))
                        setattr(pl_module, f"validation_{k}_conf_{name}", confmat(task="multiclass", num_classes=5))

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

