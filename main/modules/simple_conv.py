import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
import torchmetrics
# ===========================
# 基础卷积块
# ===========================
class DoubleConv1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv1D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

# ===========================
# UNet backbone
# ===========================
class UNet1D(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, base_ch=32):
        super(UNet1D, self).__init__()
        self.enc1 = DoubleConv1D(in_channels, base_ch)
        self.pool1 = nn.MaxPool1d(2)
        self.enc2 = DoubleConv1D(base_ch, base_ch*2)
        self.pool2 = nn.MaxPool1d(2)
        self.enc3 = DoubleConv1D(base_ch*2, base_ch*4)
        self.pool3 = nn.MaxPool1d(2)
        self.bottleneck = DoubleConv1D(base_ch*4, base_ch*8)
        self.up3 = nn.ConvTranspose1d(base_ch*8, base_ch*4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv1D(base_ch*8, base_ch*4)
        self.up2 = nn.ConvTranspose1d(base_ch*4, base_ch*2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv1D(base_ch*4, base_ch*2)
        self.up1 = nn.ConvTranspose1d(base_ch*2, base_ch, kernel_size=2, stride=2)
        self.dec1 = DoubleConv1D(base_ch*2, base_ch)
        self.out_conv = nn.Conv1d(base_ch, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        x4 = self.bottleneck(self.pool3(x3))
        x = self.up3(x4)
        x = self.dec3(torch.cat([x, x3], dim=1))
        x = self.up2(x)
        x = self.dec2(torch.cat([x, x2], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([x, x1], dim=1))
        return self.out_conv(x)

# ===========================
# mask 函数
# ===========================
def apply_random_mask(x, mask_ratio=0.3):
    mask = (torch.rand_like(x) > mask_ratio).float()
    x_masked = x * mask
    return x_masked, mask

# ===========================
# LightningModule
# ===========================
class LightningUNet(LightningModule):
    def __init__(self, in_channels=4, out_channels=4, lr=1e-3, mask_ratio=0.3,
                 mode='pretrain', num_classes=5):
        """
        mode='pretrain'：自编码模式 (mask)
        mode='finetune'：分类模式
        """
        super().__init__()
        self.save_hyperparameters()
        self.mode = mode
        self.model = UNet1D(in_channels, out_channels)
        self.lr = lr
        self.mask_ratio = mask_ratio

        if self.mode == 'finetune':

            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(out_channels, num_classes)
            )
            # 初始化 torchmetrics 混淆矩阵
            self.cm_metric = torchmetrics.ConfusionMatrix(
                task="multiclass",
                num_classes=num_classes,
                normalize=None  # 不归一化，得到计数
            )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        if self.mode == 'pretrain':
            x = batch['epochs'][0]
            x_masked, mask = apply_random_mask(x, mask_ratio=self.mask_ratio)
            out = self.forward(x_masked)
            masked_area = (1 - mask)
            loss = (((out - x) * masked_area) ** 2).sum() / (masked_area.sum() + 1e-6)
            the_metric = 5 - loss

            self.log('train/loss', loss, prog_bar=True)
            self.log('train/the_metric', the_metric, prog_bar=True)

            return loss
        else:  # finetune
            # batch = (x, y)
            batch['Stage_label'] = torch.stack(batch['Stage_label'], dim=0).squeeze(-1)
            batch['Stage_label'] = batch['Stage_label'].reshape(-1)
            x, y = batch['epochs'][0], batch['Stage_label']
            feat = self.forward(x)        # B,C,T
            logits = self.classifier(feat)  # B,num_classes
            loss = F.cross_entropy(logits, y)
            acc = (logits.argmax(dim=1) == y).float().mean()
            self.log('CrossEntropy/train/loss', loss, prog_bar=True)
            self.log('CrossEntropy/train/max_accuracy_epoch', acc, prog_bar=True)
            preds = logits.argmax(dim=1)
            # 用 torchmetrics 累积
            self.cm_metric.update(preds, y)
            return loss

    def validation_step(self, batch, batch_idx):
        if self.mode == 'pretrain':
            x = batch['epochs'][0]
            x_masked, mask = apply_random_mask(x, mask_ratio=self.mask_ratio)
            out = self.forward(x_masked)
            masked_area = (1 - mask)
            loss = (((out - x) * masked_area) ** 2).sum() / (masked_area.sum() + 1e-6)
            self.log('validation/loss', loss, prog_bar=True)
            the_metric = 5 - loss
            self.log('validation/the_metric', the_metric, prog_bar=True)

        else:
            batch['Stage_label'] = torch.stack(batch['Stage_label'], dim=0).squeeze(-1)
            batch['Stage_label'] = batch['Stage_label'].reshape(-1)
            x, y = batch['epochs'][0], batch['Stage_label']
            feat = self.forward(x)
            logits = self.classifier(feat)
            loss = F.cross_entropy(logits, y)
            acc = (logits.argmax(dim=1) == y).float().mean()
            self.log('CrossEntropy/validation/loss', loss, prog_bar=True)
            self.log('CrossEntropy/validation/max_accuracy_epoch', acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        if self.mode == 'pretrain':
            x = batch['epochs'][0]
            x_masked, mask = apply_random_mask(x, mask_ratio=self.mask_ratio)
            out = self.forward(x_masked)
            masked_area = (1 - mask)
            loss = (((out - x) * masked_area) ** 2).sum() / (masked_area.sum() + 1e-6)
            self.log('validation/loss', loss, prog_bar=True)
            the_metric = 5 - loss
            self.log('validation/the_metric', the_metric, prog_bar=True)

        else:
            batch['Stage_label'] = torch.stack(batch['Stage_label'], dim=0).squeeze(-1)
            batch['Stage_label'] = batch['Stage_label'].reshape(-1)
            x, y = batch['epochs'][0], batch['Stage_label']
            feat = self.forward(x)
            logits = self.classifier(feat)
            loss = F.cross_entropy(logits, y)
            acc = (logits.argmax(dim=1) == y).float().mean()

            preds = logits.argmax(dim=1)
            # 用 torchmetrics 累积
            self.cm_metric.update(preds, y)

            self.log('CrossEntropy/test/loss', loss, prog_bar=True)
            self.log('CrossEntropy/test/max_accuracy_epoch', acc, prog_bar=True)

    def on_test_epoch_end(self):
        if self.mode != 'pretrain':
            cm = self.cm_metric.compute().cpu().numpy()
            print("\n===== Confusion Matrix (torchmetrics) =====")
            print(cm)
            # 如果需要保存
            # np.save("confusion_matrix.npy", cm)
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)