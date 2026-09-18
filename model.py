import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import timm

from monai.networks.nets import DenseNet121

# ===================================================================
# 消融实验：Monkey Patch 替换 SwinTransformerBlock._attn，取消 SW-MSA
# ===================================================================
def patched_attn(self, x):
    """
    替换原 SwinTransformerBlock._attn，强制取消 cyclic shift（即取消 SW-MSA）。
    关键改动：不做 torch.roll，直接使用原始窗口；不做 reverse cyclic shift。
    """
    B, H, W, C = x.shape
    # 强制跳过 cyclic shift，直接使用原始窗口
    shifted_x = x
    # pad for resolution not divisible by window size
    pad_h = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
    pad_w = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
    shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    _, Hp, Wp, _ = shifted_x.shape
    # partition windows
    x_windows = timm.models.swin_transformer.window_partition(shifted_x, self.window_size)
    x_windows = x_windows.view(-1, self.window_area, C)
    # W-MSA（因为没有 roll，attn_mask 实际不生效）
    if getattr(self, 'dynamic_mask', False):
        attn_mask = self.get_attn_mask(shifted_x)
    else:
        attn_mask = self.attn_mask
    attn_windows = self.attn(x_windows, mask=attn_mask)
    # merge windows
    attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    shifted_x = timm.models.swin_transformer.window_reverse(attn_windows, self.window_size, Hp, Wp)
    shifted_x = shifted_x[:, :H, :W, :].contiguous()
    # 强制跳过 reverse cyclic shift
    x = shifted_x
    return x


def apply_no_shift_patch(model):
    """
    遍历模型，将所有 SwinTransformerBlock 的 _attn 替换为 patched_attn。
    调用后模型即取消 SW-MSA，仅保留 W-MSA。
    """
    for module in model.modules():
        if module.__class__.__name__ == 'SwinTransformerBlock':
            module._attn = patched_attn.__get__(module, module.__class__)
    return model


# ====================== 主模型：Swin Transformer Tiny ======================
class OCT_Model(nn.Module):
    """
    主分类模型：Swin Transformer Tiny（论文最终选用）。
    
    Args:
        num_classes: 分类数，固定为9
        img_size: 输入图像尺寸 (H, W)，默认 (256, 448)
        use_shift: True=标准Swin-T（含SW-MSA），False=取消SW-MSA（消融实验）
    """
    def __init__(self, num_classes=9, img_size=(256, 448), use_shift=True):
        super(OCT_Model, self).__init__()
        self.use_shift = use_shift
        self.model = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=True, num_classes=num_classes,
            in_chans=1, img_size=img_size,
            drop_rate=0.2, drop_path_rate=0.1, attn_drop_rate=0.05
        )
        # 消融实验：use_shift=False 时取消 SW-MSA
        if not use_shift:
            apply_no_shift_patch(self.model)

    def forward(self, x):
        return self.model(x)

class SwinModel(nn.Module):
    """Swin-T（独立定义，与OCT_Model参数一致，便于消融对比）"""
    def __init__(self, num_classes=9, img_size=(256, 448), use_shift=True):
        super().__init__()
        self.use_shift = use_shift
        self.model = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=True, num_classes=num_classes,
            in_chans=1, img_size=img_size,
            drop_rate=0.2, drop_path_rate=0.1, attn_drop_rate=0.05
        )
        if not use_shift:
            apply_no_shift_patch(self.model)
    def forward(self, x):
        return self.model(x)

# ====================== 基线模型 ======================
class DenseNetModel(nn.Module):
    """
    DenseNet-121基线模型（MONAI实现）。
    分类头替换为 Dropout(0.5) + Linear，增强抗过拟合能力。
    """
    def __init__(self, num_classes=9):
        super().__init__()
        self.model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=num_classes, pretrained=True)
        in_features = self.model.class_layers.out.in_features
        self.model.class_layers.out = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes)
        )
    def forward(self, x):
        return self.model(x)


class ResNetModel(nn.Module):
    """ResNet-50基线模型（timm实现）"""
    def __init__(self, num_classes=9):
        super().__init__()
        self.model = timm.create_model(
            'resnet50', pretrained=True,
            in_chans=1, num_classes=num_classes, drop_rate=0.5
        )
    def forward(self, x):
        return self.model(x)


class ConvNeXtModel(nn.Module):
    """ConvNeXt-Tiny基线模型（timm实现）"""
    def __init__(self, num_classes=9):
        super().__init__()
        self.model = timm.create_model(
            'convnext_tiny', pretrained=True,
            in_chans=1, num_classes=num_classes, drop_rate=0.5
        )
    def forward(self, x):
        return self.model(x)


class vitModel(nn.Module):
    """ViT-Small基线模型（timm实现，参数量~22M）"""
    def __init__(self, num_classes=9, img_size=(256, 448)):
        super().__init__()
        self.model = timm.create_model(
            'vit_small_patch16_224',
            pretrained=True, in_chans=1,
            num_classes=num_classes, img_size=img_size,
            drop_rate=0.2, attn_drop_rate=0.05, drop_path_rate=0.1
        )
    def forward(self, x):
        return self.model(x)
