import torch, torch.nn as nn, torch.nn.functional as F
# ------------------------------------------------------------------ #
# 1.  Multi-Scale SpO₂ Token Extractor  (1 Hz → T_eeg token)
# ------------------------------------------------------------------ #
class MultiScaleSpO2Extractor(nn.Module):
    def __init__(self, d_model: int = 128, k_set=(3, 9, 15)):
        """
        输出维度严格 = d_model
        k_set       : 不同卷积核大小（秒为单位）
        """
        super().__init__()

        # 1) 计算每个卷积分支应输出的通道数
        n_br   = len(k_set)
        base   = d_model // n_br            # 整除部分
        extra  = d_model %  n_br            # 余数部分
        ch_each = [base + (1 if i < extra else 0) for i in range(n_br)]
        assert sum(ch_each) == d_model      # 总和 = d_model

        # 2) 多尺度卷积分支
        self.branches = nn.ModuleList([
            nn.Conv1d(1, ch_each[i], k_set[i],
                      stride=1, padding=k_set[i] // 2, bias=False)
            for i in range(n_br)
        ])

        self.bn   = nn.BatchNorm1d(d_model)
        self.proj = nn.Conv1d(3 * d_model, d_model, kernel_size=1)

    def forward(self, spo2, tgt_len=None):
        """
        spo2 : (B, 1, T_spo2)  —— 原始 1 Hz 波形
        tgt_len : 与 EEG token 对齐的目标长度 (T_eeg)。若 None 则保持原长
        return  : (B, tgt_len, d_model)
        """
        if spo2.dim() == 2:
            spo2 = spo2.unsqueeze(1)                      # (B,1,T)

        # 多尺度卷积
        feats = torch.cat([br(spo2) for br in self.branches], dim=1)  # (B,d_model,T)
        feats = F.gelu(self.bn(feats))                    # 时序维仍 T_spo2

        # 差分 Δ / Δ²
        d1 = F.pad(feats[:, :, 1:] - feats[:, :, :-1], (1, 0))
        d2 = F.pad(d1[:, :, 1:] - d1[:, :, :-1],  (1, 0))
        feats = torch.cat([feats, d1, d2], dim=1)         # (B,3*d_model,T)
        feats = self.proj(feats)                          # (B,d_model,T)

        # # 线性插值对齐到 T_eeg
        # if tgt_len is not None and tgt_len != feats.shape[-1]:
        #     feats = F.interpolate(feats, size=tgt_len,
        #                            mode='linear', align_corners=False)

        return feats.transpose(1, 2)                      # (B,tgt_len,d_model)
# ------------------------------------------------------------------ #
# 2.  Cross-Attention Fusion Block  (带门控残差)
# ------------------------------------------------------------------ #
class GatedCrossAttn(nn.Module):
    def __init__(self, d_eeg=768, d_spo2=128, n_heads=16, p_drop=0.1):
        super().__init__()
        proj_dim = d_spo2 * 2 if d_spo2 < d_eeg//2 else d_spo2
        # 手动线性投影 Spo2 → EEG 维度
        self.proj_spo2 = nn.Sequential(
            nn.LayerNorm(d_spo2),
            nn.Linear(d_spo2, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, d_eeg)
        )
        self.norm_eeg = nn.LayerNorm(d_eeg)
        self.norm_spo2_proj = nn.LayerNorm(d_eeg)

        # Cross-Attention
        self.xattn = nn.MultiheadAttention(
            embed_dim=d_eeg,
            num_heads=n_heads,
            dropout=p_drop,
            batch_first=True
        )

        # 残差 & Gate
        self.gate_fc = nn.Sequential(
            nn.Linear(d_eeg, d_eeg),
            nn.ReLU(),
            nn.Linear(d_eeg, 1),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(p_drop)

        self.out_ln = nn.LayerNorm(d_eeg)

        self._init_weights()

    def _init_weights(self):
        # 初始化 proj_spo2 中所有 Linear 层
        for m in self.proj_spo2.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

        # 初始化 gate_fc 中所有 Linear 层
        for m in self.gate_fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        # 初始化 cross-attention 参数
        for name, param in self.xattn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    def forward(self, eeg_tok, spo2_tok):
        assert not torch.isnan(eeg_tok).any(), "❌ NaN in eeg_tok"

        # LayerNorm → Linear → Project到d_eeg
        eeg_tok = self.norm_eeg(eeg_tok)  # (B, T1, 768)
        spo2_proj = self.proj_spo2(spo2_tok)  # (B, T2, 768)
        spo2_proj = self.norm_spo2_proj(spo2_proj)
        assert not torch.isnan(spo2_proj).any(), "❌ NaN after proj_spo2"

        # Cross-attention
        attn_out, _ = self.xattn(eeg_tok, spo2_proj, spo2_proj)

        if torch.isnan(attn_out).any():
            print("⚠️ NaN detected in attn_out, replacing with 0")
            print("attn_out max:", attn_out.max().item())
            print("attn_out min:", attn_out.min().item())
            print("eeg_tok std:", eeg_tok.std().item(), "spo2_tok std:", spo2_tok.std().item())
            attn_out = torch.nan_to_num(attn_out, nan=0.0, posinf=1e4, neginf=-1e4)

        gate = self.gate_fc(attn_out.mean(dim=1))  # (B, 1)
        gated = gate.unsqueeze(1) * attn_out
        out = self.out_ln(eeg_tok + self.dropout(gated))  # ← 用上 out_ln，提升稳定性

        if torch.isnan(out).any():
            print("⚠️ NaN in GatedCrossAttn final output")
            out = torch.nan_to_num(out, nan=0.0)

        return out
# ------------------------------------------------------------------ #
# 3.  整体 Wearable Sleep Model  (分期 + ODS)
# ------------------------------------------------------------------ #
class WearableSleepModel(nn.Module):
    def __init__(self,
                 eeg_backbone: nn.Module,
                 d_model: int = 128,
                 n_heads: int = 4,
                 xattn_layer: int = 6):
        """
        eeg_backbone : 预训练 SleepGPT encoder (L 层) —— 输出 (B,T,d_model)
        xattn_layer  : 在第几层前插入 Cross-Attention (0-based)
        """
        super().__init__()
        self.eeg_enc    = eeg_backbone
        self.spo2_enc   = MultiScaleSpO2Extractor(d_model)
        self.xattn      = GatedCrossAttn(d_model, n_heads)
        self.x_layer_id = xattn_layer

        # 两个任务头
        self.stage_head = nn.Sequential(
            nn.Linear(d_model, 256), nn.ReLU(),
            nn.Linear(256, 5)                        # 5 stage logits
        )
        self.ods_head = nn.Sequential(
            nn.Linear(d_model, 128), nn.ReLU(),
            nn.Linear(128, 1)                        # 每 token 事件 logit
        )

    def forward(self, eeg_wav, spo2_1hz):
        """
        eeg_wav   : 传进 SleepGPT 的输入格式 (自己处理 patch/token)
        spo2_1hz  : (B, T_spo2) 原始 1 Hz 波形
        """
        # 1) 通过预训练 EEG encoder 得到 token
        eeg_tok = self.eeg_enc(eeg_wav)              # (B,T_eeg,d)

        # 2) SpO₂ → token 序列，与 T_eeg 对齐
        spo2_tok = self.spo2_enc(spo2_1hz, eeg_tok.size(1))

        # 3) 在 backbone 中层插入 Cross-Attention
        h = eeg_wav
        for lid, layer in enumerate(self.layers):
            if lid == self.x_layer_id:
                h = self.xattn(h, spo2_tok)  # 注入
            h = layer(h)

        # 4) 两个任务输出
        stage_logits = self.stage_head(h)            # (B,T,5)
        ods_logits   = self.ods_head(h).squeeze(-1)  # (B,T)

        return stage_logits, ods_logits
# ------------------------------------------------------------------ #
# 4.  Hybrid Loss (分期 + ODS)  —— 适配时序 logits
# ------------------------------------------------------------------ #
def hybrid_loss(stage_logits, ods_logits,
                stage_labels,  ods_labels,
                stage_mask=None, ods_mask=None,      # ← 新增
                ods_pos_w=10.0,
                alpha=0.8, num_classes=5):
    """
    stage_logits : (B,T,5)   stage_labels : (B,T)
    ods_logits   : (B,T)     ods_labels   : (B,T)  (0/1)
    stage_mask   : (B,T)     1=有效 stage   0=忽略   (可 None)
    ods_mask     : (B,T)     1=有 ODS 标签 0=缺失   (可 None)
    """

    # ------------- Stage loss -------------
    if stage_mask is None:
        stage_loss = F.cross_entropy(
            stage_logits.reshape(-1, num_classes),
            stage_labels.reshape(-1),
        )
    else:
        stage_mask = stage_mask.reshape(-1).float()
        stage_loss = F.cross_entropy(
            stage_logits.reshape(-1, num_classes),
            stage_labels.reshape(-1),
            reduction='none'
        )
        stage_loss = (stage_loss * stage_mask).sum() / stage_mask.sum().clamp_min(1)

    # ------------- ODS loss ---------------
    if ods_mask is None:
        ods_loss = F.binary_cross_entropy_with_logits(
            ods_logits.reshape(-1),
            ods_labels.reshape(-1).float(),
            pos_weight=torch.tensor([ods_pos_w], device=stage_logits.device)
        )
    else:
        ods_mask = ods_mask.reshape(-1).float()
        ods_loss_raw = F.binary_cross_entropy_with_logits(
            ods_logits.reshape(-1),
            ods_labels.reshape(-1).float(),
            pos_weight=torch.tensor([ods_pos_w], device=stage_logits.device),
            reduction='none'
        )
        ods_loss = (ods_loss_raw * ods_mask).sum() / ods_mask.sum().clamp_min(1)

    # ------------- Total ------------------
    return alpha * stage_loss + (1 - alpha) * ods_loss


class OdsTemporalEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1,
                 model_type: str = "lstm", nhead: int = 4, dropout: float = 0.1,
                 bidirectional: bool = True):
        """
        参数:
            input_dim: 输入特征维度 (比如 crossattn 输出 d_eeg)
            hidden_dim: RNN隐藏维度 (单向的隐藏维度)
            num_layers: 层数 (RNN/Transformer)
            model_type: "lstm" | "gru" | "transformer"
            nhead: 如果是 transformer 的多头注意力头数
            dropout: dropout 概率
            bidirectional: 是否使用双向RNN (仅LSTM/GRU有效)
        """
        super().__init__()
        self.model_type = model_type.lower()
        self.bidirectional = bidirectional

        if self.model_type == "lstm":
            self.encoder = nn.LSTM(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional
            )
            self.output_dim = hidden_dim * (2 if bidirectional else 1)

        elif self.model_type == "gru":
            self.encoder = nn.GRU(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional
            )
            self.output_dim = hidden_dim * (2 if bidirectional else 1)

        elif self.model_type == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=nhead,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.output_dim = input_dim  # Transformer 维度不变

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def forward(self, x, lengths=None):
        """
        x: (B, T, D)
        lengths: (B,) 可选，用于 pack_padded_sequence (RNN 时使用)
        return: 编码后的时序特征 (B, T, output_dim)
        """
        if self.model_type in ["lstm", "gru"]:
            # 这里可以根据需要使用 pack_padded_sequence，但一般可以直接用
            out, _ = self.encoder(x)  # (B,T,H)  H=hidden_dim * (2 if bidirectional else 1)
            return out, _

        elif self.model_type == "transformer":
            return (self.encoder(x), )  # (B,T,D)

        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

