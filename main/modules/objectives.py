import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import json
import tqdm
import functools
import torch.distributed as dist
from main.utils import Fpfn, By_Event


from torch.utils.data.distributed import DistributedSampler
from einops import rearrange
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
from main.utils import PolyLoss
from main.utils.dist import all_reduce_mean


def compute_time_fft_only(pl_module, batch, stage, aggregate=True):
    """
    ret = {
    "cls_weak_feats": cls_weak_feats,
    "cls_strong_feats": cls_strong_feats,
    "batch": batch,  # epochs, ids_keep, mask, Stage_label, Spindle_label,
    "mask_feats": cls_mask_feats}
    Args:
        aggregate: Use all GPUs
        pl_module: module
        batch: batch
        stage: train, test, validation
    Returns:

    """
    if pl_module.time_only:
        res = pl_module.infer_time_only(batch)
    else:
        res = pl_module.infer_fft_only(batch)
    cls_weak_feats = res['cls_weak_feats']
    cls_strong_feats = res['cls_strong_feats']
    if pl_module.time_only:
        logit_mask_scale = pl_module.logit_mask_scale.exp().mean()
        cls_weak_mask_feats = res['cls_weak_mask_feats']

    logit_scale = pl_module.logit_scale.exp().mean()
    if aggregate and dist.is_available() and dist.is_initialized() and pl_module.hparams.config['device'] == 'cuda':
        # print('aggregate')
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # We gather tensors from all gpus to get more negatives to contrast with.
        gathered_weak_features = [
            torch.zeros_like(cls_weak_feats) for _ in range(world_size)
        ]
        gathered_strong_features = [
            torch.zeros_like(cls_strong_feats) for _ in range(world_size)
        ]
        if pl_module.time_only:
            gathered_weak_mask_features = [
                torch.zeros_like(cls_weak_mask_feats) for _ in range(world_size)
            ]
        dist.all_gather(gathered_weak_features, cls_weak_feats)
        dist.all_gather(gathered_strong_features, cls_strong_feats)
        if pl_module.time_only:
            dist.all_gather(gathered_weak_mask_features, cls_weak_mask_feats)

        all_weak_features = torch.cat(
            [cls_weak_feats]
            + gathered_weak_features[:rank]
            + gathered_weak_features[rank + 1:]
        )
        all_strong_features = torch.cat(
            [cls_strong_feats]
            + gathered_strong_features[:rank]
            + gathered_strong_features[rank + 1:]
        )
        if pl_module.time_only:
            all_weak_mask_features = torch.cat(
                [cls_weak_mask_feats]
                + gathered_weak_mask_features[:rank]
                + gathered_weak_mask_features[rank + 1:]
            )

        # this is needed to send gradients back everywhere.
        logits_per_w = logit_scale * all_weak_features @ all_strong_features.t()
        # print('logits_per_fft')
        # print(torch.isnan(logits_per_fft))
        # assert torch.isnan(logits_per_w).sum() == 0, "logits_per_w is out of break"
        logits_per_s = logits_per_w.t()

        if pl_module.time_only:
            logits_per_w_m = logit_mask_scale * all_weak_mask_features @ all_strong_features.t()
            logits_per_s_m = logits_per_w_m.t()

    else:
        logits_per_w = logit_scale * cls_weak_feats @ cls_strong_feats.t()
        logits_per_s = logit_scale * cls_strong_feats @ cls_weak_feats.t()
        if pl_module.time_only:
            logits_per_w_m = logit_mask_scale * cls_weak_mask_feats @ cls_strong_feats.t()
            logits_per_s_m = logit_mask_scale * cls_weak_feats @ cls_weak_mask_feats.t()

    ground_truth = torch.arange(len(logits_per_w)).long().to(device=logits_per_w.device)
    assert torch.isnan(logits_per_w).sum() == 0, "logits_per_w is out of break"
    assert torch.isnan(logits_per_s).sum() == 0, "logits_per_s is out of break"
    if pl_module.time_only:
        assert torch.isnan(logits_per_w_m).sum() == 0, "logits_per_w_m is out of break"
        assert torch.isnan(logits_per_s_m).sum() == 0, "logits_per_s_m is out of break"

    # print('itc:---------------------------')
    # print(logits_per_fft, logits_per_fft.shape)
    # print(logits_per_time, logits_per_time.shape)
    if pl_module.time_only:
        itc_loss = (
                           F.cross_entropy(logits_per_w.float(), ground_truth)
                           + F.cross_entropy(logits_per_s.float(), ground_truth)
                           + F.cross_entropy(logits_per_w_m.float(), ground_truth)
                           + F.cross_entropy(logits_per_s_m.float(), ground_truth)
                   ) / 4
    else:
        itc_loss = (
                F.cross_entropy(logits_per_w.float(), ground_truth)
                + F.cross_entropy(logits_per_s.float(), ground_truth))/2
    itc_total_loss = itc_loss
    ret = {
        "itc_loss": itc_total_loss,
        "itc_logit_scale": logit_scale,
        "itc_labels": ground_truth,
    }
    if pl_module.time_only:
        ret.update({ "itc_logit_mask_scale": logit_mask_scale})
        mask_feats = res['mask_feats']
        epochs = batch['epochs'][0]
        time_mask_patch = res['time_mask_patch']  # [N, L_t]
        forward_masked_loss = pl_module.forward_masked_loss(mask_feats, epochs, time_mask_patch)
        phase = stage
        loss = getattr(pl_module, f"{phase}_mtm_loss")(forward_masked_loss)
        pl_module.log(f"mtm/{phase}/loss", loss, on_step=True, sync_dist_group=True, prog_bar=True)
        ret.update({"forward_masked_loss": forward_masked_loss})

    phase = stage
    loss = getattr(pl_module, f"{phase}_itc_loss")(ret["itc_loss"])
    scale = getattr(pl_module, f"{phase}_itc_logit_scale")(ret["itc_logit_scale"])
    w2s_acc = getattr(pl_module, f"{phase}_itc_w2s_accuracy")(
        logits_per_w, ret["itc_labels"]
    )
    s2w_acc = getattr(pl_module, f"{phase}_itc_s2w_accuracy")(
        logits_per_s, ret["itc_labels"]
    )
    if pl_module.time_only:
        w2s_mask_acc = getattr(pl_module, f"{phase}_itc_w2s_mask_accuracy")(
            logits_per_w_m, ret["itc_labels"]
        )
        s2w_mask_acc = getattr(pl_module, f"{phase}_itc_s2w_mask_accuracy")(
            logits_per_s_m, ret["itc_labels"]
        )
    pl_module.log(f"itc/{phase}/loss", loss, on_step=True, sync_dist_group=True, prog_bar=True)
    pl_module.log(f"itc/{phase}/logit_scale", scale)
    pl_module.log(f"itc/{phase}/w2s_accuracy", w2s_acc)
    pl_module.log(f"itc/{phase}/s2w_accuracy", s2w_acc)
    if pl_module.time_only:
        mask_scale = getattr(pl_module, f"{phase}_itc_logit_mask_scale")(ret["itc_logit_mask_scale"])
        pl_module.log(f"itc/{phase}/logit_mask_scale", mask_scale)

        pl_module.log(f"itc/{phase}/w2s_mask_accuracy", w2s_mask_acc)
        pl_module.log(f"itc/{phase}/w2s_mask_accuracy", s2w_mask_acc)

    return ret


def compute_fft_only(pl_module, batch, stage):
    pass

def compute_mtm(pl_module, batch, stage):
    """
    The implementation of masked time-fft reconstruction refers to MAE (https://github.com/facebookresearch/mae)
    Reconstruct the masked sequences.
    time_mask=True
    Args:
        stage:
        pl_module: Model
        batch: batch
    Returns:
        ret = {
        "mtm_loss": forward_masked_loss,  # Only the masked patches
        "mtm_logits": cls_feats,  # The prediction: [N, L_t, patch_size:200].
    }
    """
    infer = pl_module.infer(batch, time_mask=True)
    time_mask_patch = infer['time_mask_patch']  # [N, L_t]
    fft_mask_patch = infer['fft_mask_patch']
    cls_feats = infer['cls_feats']  # [N, L_t, patch_size:200]
    cls_feats_fft = infer['cls_feats_fft']
    epochs = batch['epochs']
    forward_masked_loss = pl_module.forward_masked_loss(cls_feats, epochs[0], time_mask_patch)
    forward_masked_loss_2d = pl_module.forward_masked_loss_2D(cls_feats_fft, epochs[1], fft_mask_patch)
    ret = {
        "mtm_loss": forward_masked_loss,
        'mtm_loss2': forward_masked_loss_2d,
        "mtm_logits": cls_feats,
        "mtm_logits_fft": cls_feats_fft,
        "time_mask_patch": time_mask_patch,
        "fft_mask_patch": fft_mask_patch,
        "batch": batch,
    }
    phase = stage
    loss = getattr(pl_module, f"{phase}_mtm_loss")(ret["mtm_loss"])
    loss2 = getattr(pl_module, f"{phase}_mtm_loss2")(ret["mtm_loss2"])

    pl_module.log(f"mtm/{phase}/loss", loss,  on_step=True, sync_dist_group=True, prog_bar=True)
    pl_module.log(f"mtm/{phase}/loss2", loss2,  on_step=True, sync_dist_group=True, prog_bar=True)

    return ret


def compute_itm_hardneg(pl_module, batch, sim_f2t, sim_t2f, stage):
    """
    The implementation of time-fft to compute hard negative samples refers to VLMO (https://github.com/microsoft/unilm/tree/master/vlmo)
    L_t = numpatches*channels(57)
    L_f = numpatches*choose_fft_channels
    Args:
        stage:
        pl_module: model
        batch: batch
        sim_f2t: fft to time. Matrix (batch*world_size) * (batch*world_size)
        sim_t2f: time to fft
    Returns:
        ret = {
        "itm_loss": itm_loss,
        "itm_logits": itm_logits,
        "itm_labels": itm_labels,
    }
    """
    # print('sim_f2t, sim_t2f')
    # print(sim_f2t, sim_t2f)
    # print(sim_f2t.shape, sim_t2f.shape)
    # print('isnan:', torch.isnan(sim_t2f).sum(), torch.isnan(sim_f2t).sum())
    pos_len = batch["epochs"][0].size(0)
    neg_len = batch["epochs"][0].size(0)
    bsz = batch["epochs"][0].size(0)
    itm_labels = torch.cat([torch.ones(pos_len), torch.zeros(neg_len), torch.zeros(neg_len)]).to(
        pl_module.device
    )
    infer_pos = pl_module.infer(batch, time_mask=False)  # epochs, mask,
    batch = infer_pos['batch']
    batch_time = batch['epochs'][0].contiguous()  # B, L_t
    batch_fft = batch['epochs'][1].contiguous()  # B, L_f
    batch_mask = batch['mask']  # B, C
    batch_mask_fft = batch_mask[2]  # L_t + 1
    batch_mask_time = batch_mask[0]  # L_f + 1
    batch_mask_fft_cls = batch_mask[3]
    batch_mask_time_cls = batch_mask[1]
    # print(batch_time.shape, batch_fft.shape)
    with torch.no_grad():
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # We gather tensors from all gpus to get more hard negative candidates.
        gathered_time = [
            torch.zeros_like(batch_time) for _ in range(world_size)
        ]
        gathered_masks_fft = [
            torch.zeros_like(batch_mask_fft) for _ in range(world_size)
        ]
        gathered_masks_time = [
            torch.zeros_like(batch_mask_time) for _ in range(world_size)
        ]
        gathered_fft = [
            torch.zeros_like(batch_fft) for _ in range(world_size)
        ]

        dist.all_gather(gathered_time, batch_time)
        dist.all_gather(gathered_masks_fft, batch_mask_fft)
        dist.all_gather(gathered_masks_time, batch_mask_time)
        dist.all_gather(gathered_fft, batch_fft)

        all_time = torch.cat(
            [batch_time]
            + gathered_time[:rank]
            + gathered_time[rank + 1:]
        )
        all_masks_time = torch.cat(
            [batch_mask_time]
            + gathered_masks_time[:rank]
            + gathered_masks_time[rank + 1:]
        )
        all_masks_fft = torch.cat(
            [batch_mask_fft]
            + gathered_masks_fft[:rank]
            + gathered_masks_fft[rank + 1:]
        )
        all_fft = torch.cat(
            [batch_fft]
            + gathered_fft[:rank]
            + gathered_fft[rank + 1:]
        )

    with torch.no_grad():
        # print('weights')
        # print('sim_f2t: ', sim_f2t[:bsz, :].float())
        # print('sim_t2f: ', sim_t2f[:bsz, :].float())
        weights_f2t = F.softmax(sim_f2t[:bsz, :].float(), dim=1)
        weights_t2f = F.softmax(sim_t2f[:bsz, :].float(), dim=1)
        weights_f2t.fill_diagonal_(0)
        weights_t2f.fill_diagonal_(0)
        # print('weights_f2t: ', weights_f2t)
        # print('weights_t2f: ', weights_t2f)
    try:
        fft_neg = []
        fft_masks_neg = []
        for b in range(bsz):
            neg_idx = torch.multinomial(weights_t2f[b], 1).item()
            fft_neg.append(all_fft[neg_idx])
            fft_masks_neg.append(all_masks_fft[neg_idx])
        fft_neg = torch.stack(fft_neg, dim=0)
        fft_masks_neg = torch.stack(fft_masks_neg, dim=0)

        # select a negative text for each image
        time_neg = []
        time_masks_neg = []
        for b in range(bsz):
            neg_idx = torch.multinomial(weights_f2t[b], 1).item()
            time_neg.append(all_time[neg_idx])
            time_masks_neg.append(all_masks_time[neg_idx])
        time_neg = torch.stack(time_neg, dim=0)
        time_masks_neg = torch.stack(time_masks_neg, dim=0)
    except Exception as e:
        print(e)
        print("index: ", batch['index'], ' device= ', pl_module.device)
    # text_labels is not used in ITM loss

    batch_fft_neg = {'epochs': (batch_time, fft_neg),
                     'mask': [batch_mask_time_cls, batch_mask_time, batch_mask_fft_cls, fft_masks_neg]}  # epochs, mask
    # print('infer_fft_neg')
    infer_fft_neg = pl_module.infer(batch_fft_neg, time_mask=False)

    batch_time_neg = {'epochs': (time_neg, batch_fft),
                      'mask': [batch_mask_time_cls, time_masks_neg, batch_mask_fft_cls, batch_mask_fft]}  # epochs, mask
    # print('infer_time_neg')
    infer_time_neg = pl_module.infer(batch_time_neg, time_mask=False)

    all_cls_feats = torch.cat([infer_pos["cls_feats"], infer_fft_neg["cls_feats"], infer_time_neg["cls_feats"]],
                              dim=0)
    if 1:
        N = all_cls_feats.shape[0]
        noise = torch.rand(N, device=pl_module.device)
        ids_shuffle = torch.argsort(noise)
        all_cls_feats = all_cls_feats[ids_shuffle]
        itm_labels = itm_labels[ids_shuffle]
    else:
        ids_shuffle = torch.arange(N)
    itm_logits = pl_module.itm_score(all_cls_feats)

    itm_loss = F.cross_entropy(itm_logits, itm_labels.long())

    ret = {
        "itm_loss": itm_loss,
        "itm_logits": itm_logits,
        "itm_labels": itm_labels,
    }

    phase = stage
    loss = getattr(pl_module, f"{phase}_itm_loss")(ret["itm_loss"])
    acc = getattr(pl_module, f"{phase}_itm_accuracy")(
        ret["itm_logits"], ret["itm_labels"]
    )
    pl_module.log(f"itm/{phase}/loss", loss, on_step=True, sync_dist_group=True, prog_bar=True)
    pl_module.log(f"itm/{phase}/accuracy", acc, on_step=True, sync_dist_group=True,  prog_bar=True)

    return ret


# The implementation of image-text contrastive refers to open_clip (https://github.com/mlfoundations/open_clip)
def compute_itc(pl_module, batch,  stage, aggregate=True):
    """
    The implementation of time-fft contrastive refers to open_clip (https://github.com/mlfoundations/open_clip)
    Args:
        stage:
        pl_module:Model
        batch:batch
        aggregate:DDP
    Returns:
    ret = {
        "itc_loss": itc_total_loss,
        "itc_f2t_logits": logits_per_fft,  # Matrix (batch*world_size) * (batch*world_size)
        "itc_t2f_logits": logits_per_time, # Matrix (batch*world_size) * (batch*world_size)
        "itc_labels": ground_truth,  # labels [0,1,2,3,4,5......] Only diagonal is true.
        "itc_logit_scale": logit_scale,  # temperature
        "itc_logit_tf_scale": logit_tf_scale,  # tf-expert temperature
    }
    """

    infer_time = pl_module.infer_time(batch)
    infer_fft = pl_module.infer_fft(batch)

    time_features = infer_time["cls_feats"]
    fft_features = infer_fft["cls_feats"]
    logit_scale = pl_module.logit_scale.exp().mean()
    # print("cls_feats:", torch.isnan(time_features).sum())
    # print("cls_feats:", torch.isnan(fft_features).sum())

    time_tfffn_features = infer_time["cls_tfffn_feats"]
    fft_tfffn_features = infer_fft["cls_tfffn_feats"]
    logit_tf_scale = pl_module.logit_tf_scale.exp().mean()
    # print('fft_features, time_features')
    # print(fft_features, time_features)
    # print("cls_feats:", torch.isnan(time_tfffn_features).sum())
    # print("cls_feats:", torch.isnan(fft_tfffn_features).sum())
    if aggregate and dist.is_available() and dist.is_initialized():
        # print('aggregate')
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # We gather tensors from all gpus to get more negatives to contrast with.
        gathered_fft_features = [
            torch.zeros_like(fft_features) for _ in range(world_size)
        ]
        gathered_time_features = [
            torch.zeros_like(time_features) for _ in range(world_size)
        ]
        dist.all_gather(gathered_fft_features, fft_features)
        dist.all_gather(gathered_time_features, time_features)

        all_fft_features = torch.cat(
            [fft_features]
            + gathered_fft_features[:rank]
            + gathered_fft_features[rank + 1:]
        )
        all_time_features = torch.cat(
            [time_features]
            + gathered_time_features[:rank]
            + gathered_time_features[rank + 1:]
        )

        # this is needed to send gradients back everywhere.
        logits_per_fft = logit_scale * all_fft_features @ all_time_features.t()
        # print('logits_per_fft')
        # print(torch.isnan(logits_per_fft))
        logits_per_time = logits_per_fft.t()

        gathered_fft_tfffn_features = [
            torch.zeros_like(fft_tfffn_features) for _ in range(world_size)
        ]
        gathered_time_tfffn_features = [
            torch.zeros_like(time_tfffn_features) for _ in range(world_size)
        ]  # world_size * batch * embeds
        dist.all_gather(gathered_fft_tfffn_features, fft_tfffn_features)
        dist.all_gather(gathered_time_tfffn_features, time_tfffn_features)

        all_fft_tfffn_features = torch.cat(
            [fft_tfffn_features]
            + gathered_fft_tfffn_features[:rank]
            + gathered_fft_tfffn_features[rank + 1:]
        )
        all_time_tfffn_features = torch.cat(
            [time_tfffn_features]
            + gathered_time_tfffn_features[:rank]
            + gathered_time_tfffn_features[rank + 1:]
        )
        # this is needed to send gradients back everywhere.
        logits_per_tfffn_fft = logit_tf_scale * all_fft_tfffn_features @ all_time_tfffn_features.t()
        logits_per_tfffn_time = logits_per_tfffn_fft.t()

    else:
        logits_per_fft = logit_scale * fft_features @ time_features.t()
        logits_per_time = logit_scale * time_features @ fft_features.t()

    ground_truth = torch.arange(len(logits_per_fft)).long().to(device=logits_per_fft.get_device())
    assert torch.isnan(logits_per_fft).sum() == 0, "logits_per_fft is out of break"
    assert torch.isnan(logits_per_time).sum() == 0, "logits_per_time is out of break"

    # print('itc:---------------------------')
    # print(logits_per_fft, logits_per_fft.shape)
    # print(logits_per_time, logits_per_time.shape)
    itc_loss = (
                       F.cross_entropy(logits_per_fft.float(), ground_truth)
                       + F.cross_entropy(logits_per_time.float(), ground_truth)
               ) / 2
    if aggregate and dist.is_available() and dist.is_initialized():

        itc_tfffn_loss = (
                                 F.cross_entropy(logits_per_tfffn_fft.float(), ground_truth)
                                 + F.cross_entropy(logits_per_tfffn_time.float(), ground_truth)
                         ) / 2
    else:
        itc_tfffn_loss = itc_loss
    itc_total_loss = (itc_loss + itc_tfffn_loss) * 0.5

    ret = {
        "itc_loss": itc_total_loss,
        "itc_f2t_logits": logits_per_fft,
        "itc_t2f_logits": logits_per_time,
        "itc_labels": ground_truth,
        "itc_logit_scale": logit_scale,
        "itc_logit_tf_scale": logit_tf_scale,
    }

    phase = stage
    loss = getattr(pl_module, f"{phase}_itc_loss")(ret["itc_loss"])
    scale = getattr(pl_module, f"{phase}_itc_logit_scale")(ret["itc_logit_scale"])
    f2t_acc = getattr(pl_module, f"{phase}_itc_f2t_accuracy")(
        ret["itc_f2t_logits"], ret["itc_labels"]
    )
    t2f_acc = getattr(pl_module, f"{phase}_itc_t2f_accuracy")(
        ret["itc_t2f_logits"], ret["itc_labels"]
    )
    pl_module.log(f"itc/{phase}/loss", loss, on_step=True, sync_dist_group=True, prog_bar=True)
    pl_module.log(f"itc/{phase}/logit_scale", scale)
    pl_module.log(f"itc/{phase}/f2t_accuracy", f2t_acc)
    pl_module.log(f"itc/{phase}/t2f_accuracy", t2f_acc)

    tf_scale = getattr(pl_module, f"{phase}_itc_tf_logit_scale")(ret["itc_logit_tf_scale"])
    if aggregate and dist.is_available() and dist.is_initialized():

        tf_f2t_acc = getattr(pl_module, f"{phase}_itc_tf_f2t_accuracy")(
            logits_per_tfffn_fft, ret["itc_labels"]
        )
        tf_t2f_acc = getattr(pl_module, f"{phase}_itc_tf_t2f_accuracy")(
            logits_per_tfffn_fft, ret["itc_labels"]
        )
        pl_module.log(f"itc/{phase}/tf_f2t_accuracy", tf_f2t_acc)
        pl_module.log(f"itc/{phase}/tf_t2f_accuracy", tf_t2f_acc)
    pl_module.log(f"itc/{phase}/tf_logit_scale", tf_scale)
    return ret

def compute_ce(pl_module, batch,  stage):
    res = {}
    for name in pl_module.multi_y:
        if name == 'tf':
            infer = pl_module.infer(batch, time_mask=False)
            if pl_module.local_pooling:
                res.update({'local': infer['local_feats']})
        else:
            infer = getattr(pl_module, f"infer_{name}")(batch, time_mask=False)
        res.update(infer['cls_feats'])
    # d = res.shape[-1]
    index = batch['index']
    target = batch['Stage_label']
    targe2 = target.detach().clone()

    for k, v in res.items():
        if hasattr(pl_module, f'stage_pred_{k}_proj'):
            res[k] = getattr(pl_module, f'stage_pred_{k}_proj')(v).float()
        else:
            res[k] = v.float()
        preds = res[k]
        # print(preds.shape)
        if k=='tf':
            preds2 = preds.detach().clone()
        # preds = preds[target != -100]
        res[k] = preds
    # target = target[target != -100]
    # torch.set_printoptions(threshold=np.inf)
    # rank_zero_info(f"object target: {target}")
    # rank_zero_info(f"log_softmax: {F.log_softmax(res['tf'], dim=-1)}")
    # print("res2", res.shape, batch['Stage_label'].shape)
    if pl_module.mixup_fn is not None and pl_module.training:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
        total_loss = 0.0
        for k, v in res.items():
            ce_loss = criterion(v, target)
            total_loss += ce_loss
        if pl_module.first_log_gpu is not True:
            print('Using SoftTargetCrossEntropy Loss')
    elif pl_module.poly is True:
        criterion = PolyLoss()
        total_loss = 0.0
        for k, v in res.items():
            ce_loss = criterion(v, target)
            total_loss += ce_loss
        if pl_module.first_log_gpu is not True:
            print('Using PolyCrossEntropy Loss')
    elif pl_module.smoothing > 0.0 and pl_module.training:
        criterion = LabelSmoothingCrossEntropy(pl_module.smoothing)
        total_loss = 0.0
        for k, v in res.items():
            assert v.shape[0] == target.shape[0]
            ce_loss = criterion(v, target)
            total_loss += ce_loss
        # if pl_module.local_pooling:
        #     ce_loss_local = criterion(preds_local, target)
        #     ce_loss = (ce_loss+ce_loss_local)/2
        if pl_module.first_log_gpu is not True:
            print('Using LabelSmoothingCrossEntropy Loss')
    else:
        total_loss = 0.0
        for k, v in res.items():
            ce_loss = F.cross_entropy(v, target,  ignore_index=-100)
            total_loss += ce_loss
        # if pl_module.local_pooling:
        #     ce_loss_local = F.cross_entropy(preds_local, target, ignore_index=-100)
        #     ce_loss = (ce_loss + ce_loss_local) / 2
        if pl_module.first_log_gpu is not True:
            print('Using F.cross_entropy Loss')
    num = len(res)

    total_loss = total_loss/num
    index2 = index.detach().clone()
    target2 = target.detach().clone()
    # print(f" index2: {index2.shape}, target2: {target2.shape}")
    # pred2_list = [torch.zeros_like(preds2) for i in range(dist.get_world_size())]
    # index_list = [torch.zeros_like(index2) for i in range(dist.get_world_size())]
    # target_list = [torch.zeros_like(target2) for i in range(dist.get_world_size())]
    # dist.all_gather(pred2_list, preds2)
    # dist.all_gather(index_list, index2)
    # dist.all_gather(target_list, target2)
    # print(f"dist.all_gather")
    pred2_list = [torch.zeros_like(preds2) for i in range(1)]
    index_list = [torch.zeros_like(index2) for i in range(1)]
    target_list = [torch.zeros_like(target2) for i in range(1)]
    preds2 = torch.cat(pred2_list, dim=0)
    index2 = torch.cat(index_list, dim=0)
    target2 = torch.cat(target_list, dim=0)
    # print(f"2 rank: {dist.get_global_rank()}index2: {index2.shape}, target2: {target2.shape}")

    ret = {
        'ce_loss': total_loss,
        'label': target2,
        'feats': torch.argmax(preds2, dim=-1),
        'index': index2,
    }

    phase = stage
    # print(f"celoss: {ret['ce_loss']}")
    # print('loss = getattr(pl_module, f"{phase}_CrossEntropy_loss")(ret["ce_loss"])')
    loss = getattr(pl_module, f"{phase}_CrossEntropy_loss")(ret["ce_loss"])
    # print("pl_module.log")
    pl_module.log(f"ce/{phase}/loss", loss, on_step=True, sync_dist_group=True, prog_bar=True)
    for k, v in res.items():
        ce_acc = getattr(pl_module, f"{phase}_CrossEntropy_accuracy_{k}")(
            v, target
        )

        pl_module.log(f"ce/{phase}/{k}/ce_acc", ce_acc, on_step=True, sync_dist_group=True, prog_bar=True)
        confmat = getattr(pl_module, f"{phase}_CrossEntropy_conf_{k}")(v, target)
        precision, recall, kappa, sensitivity, specificity = confusion(confmat)
        for i in range(len(precision)):
            pl_module.log(f"ce/{phase}/ce_{i}_precision_{k}", precision[i])
            pl_module.log(f"ce/{phase}/ce_{i}_recall_{k}", recall[i])

    return ret

def confusion(cm:torch.Tensor):
    sum0 = cm.sum(axis=0)
    sum1 = cm.sum(axis=1)
    all_sum = cm.sum()
    p0 = torch.diag(cm).sum()/all_sum
    FP = sum0 - torch.diag(cm)
    FN = sum1 - torch.diag(cm)
    TP = torch.diag(cm)
    TN = all_sum - FP.sum() - FN.sum() - TP.sum()

    precision = TP / (TP + FP)

    recall = TP / (TP + FN)

    pe=(sum0*sum1).sum()/(all_sum**2)

    kappa = (p0 - pe) / (1 - pe)
    sensitivity = TP.sum() / (TP.sum()+ FN.sum())
    specificity = TN / (FP.sum() + TN)
    return precision, recall, kappa, sensitivity, specificity

def compute_fpfn(pl_module, prob, IOU_th, batch,  stage):
    infer = pl_module.infer(batch, time_mask=False)
    cls_feats = infer['cls_feats']
    fpfn = Fpfn()
    loss = fpfn(cls_feats, batch['Spindle_label'])
    by_e = By_Event(threshold=prob, IOU_threshold=IOU_th, device=cls_feats.device)
    TP, FN, FP = by_e(cls_feats.detach().clone(), batch['Spindle_label'].detach().clone())
    ret = {
        'loss': loss,
        'TP': TP,
        'FN': FN,
        'FP': FP,
    }
    phase = stage
    loss = getattr(pl_module, f"{phase}_FpFn_loss")(ret["loss"])
    TP = getattr(pl_module, f"{phase}_FpFn_TP")(ret["TP"])
    FN = getattr(pl_module, f"{phase}_FpFn_FN")(ret["FN"])
    FP = getattr(pl_module, f"{phase}_FpFn_FP")(ret["FP"])
    pl_module.log(f"FpFn/{phase}/loss", loss, on_step=True, sync_dist_group=True, prog_bar=True)
    return ret





