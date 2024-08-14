from . import cross_attention
from . import vit
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from einops import rearrange
import torch.nn.functional as F
import torch.distributed as dist
from main.utils import box_ops, matcher
from main.utils import others
from pytorch_lightning.utilities.rank_zero import rank_zero_info


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class SetCriterion(nn.Module):
    def __init__(self, matcher, losses):
        super().__init__()
        self.matcher = matcher
        self.losses = losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t[i] for t, (_, i) in zip(targets, indices)], dim=0)
        rank_zero_info(f'idx: {idx}, src_boxes.shape: {src_boxes.shape}, target_boxes.shape: {target_boxes.shape}')

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {'loss_bbox': loss_bbox.sum() / num_boxes}

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxw_to_x(src_boxes),
            box_ops.box_cxw_to_x(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indicesz
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):

        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / dist.get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    @torch.no_grad()
    def forward(self, outputs, target_sizes):

        out_bbox = outputs['pred_boxes']

        assert target_sizes.shape[1] == 1

        # convert to [x0, x1] format
        boxes = box_ops.box_cxw_to_x(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        eeg_len = target_sizes.unbind(1)
        # scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        # boxes = boxes * scale_fct[:, None, :]
        #
        # results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        #
        # return results


class Cross_Attn_Event_Model(nn.Module):
    def __init__(self,
                 qdim,
                 kvdim,
                 num_heads=8,
                 qkv_bias=False,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 use_cls_token=True,
                 drop_path=None,
                 num_queries=20,
                 decoder_depth=6,
                 act_layer=nn.GELU,
                 seq_len=2000,
                 ):
        super().__init__()
        self.pe = vit.PositionalEncoding(out_features=qdim)
        self.decoder = nn.ModuleList()
        if drop_path is not None:
            drop_path_rate = [x.item() for x in (torch.linspace(0.00, drop_path, decoder_depth))]
        else:
            drop_path_rate = None
        for i in range(decoder_depth):
            self.decoder.append(cross_attention.DecoderBlock(qdim, kvdim, num_heads=num_heads, qkv_bias=qkv_bias,
                                                             attn_drop=attn_drop, drop_path=drop_path_rate[i],
                                                             act_layer=act_layer),)
        self.qdim = qdim
        self.bbox_embed = MLP(qdim, qdim, 2, 3)
        losses = ['boxes']
        self.Criterion = SetCriterion(matcher=matcher.build_matcher({'set_cost_bbox': 1, 'set_cost_giou': 1}), losses=losses)

    def forward(self, memory):
        time_c3, fft_c3 = memory
        memory = torch.cat([time_c3, fft_c3], dim=1)
        B, L, C = memory.shape
        q = torch.zeros(self.num_queries, self.qdim).unsqueeze(0).repeat(B, 1, 1).to(memory.device)
        x = self.pe(q)
        for layer in self.decoder:
            x, attn = layer(x, memory)
        outputs_coord = self.bbox_embed(x).sigmoid()  # B, n_q, 2
        out = {'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_boxes': item} for item in outputs_coord[:-1]]

    def forward_loss(self, output, targets_orig, conver_target2_box=True):
        if conver_target2_box:
            Event = others.Event(threshold=0, device=targets_orig.deivce)
            events = Event.get_event(targets_orig, prob=False)
            targets = events
        else:
            targets = targets_orig
        loss = self.Criterion(output, targets)
        return loss
