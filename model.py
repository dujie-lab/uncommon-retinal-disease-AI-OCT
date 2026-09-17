import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# 限制单GPU，防止多卡抢占
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import tensorflow as tf
# TF强制使用CPU，不初始化CUDA
tf.config.set_visible_devices([], 'GPU')

# 后续原有导入
from monai.networks.nets import SwinUNETR, DenseNet121
from monai.networks.layers import Pool
import torch.nn.functional as F
from monai.networks.blocks import UnetResBlock, UnetUpBlock
from ds import OCT_dataset
import timm


class OCT_Model(nn.Module):
    def __init__(self, num_classes, img_size=(256, 448)):
        super(OCT_Model, self).__init__()

        self.model = timm.create_model(
            'swin_tiny_patch4_window7_224', 
            pretrained=True,  # 使用 ImageNet 预训练权重
            num_classes=num_classes,
            in_chans=1,
            img_size=img_size,
            drop_rate=0.2, #0.5
            drop_path_rate=0.1,  #0.1
            attn_drop_rate=0.05
        )

    def forward(self, x):
        # x 的形状应该是 (Batch_Size, 1, 256, 448)，确保输入正确  
        output = self.model(x)
        return output 



# class OCT_Model(nn.Module):
    
#     def __init__(self, num_classes):
#         super(OCT_Model, self).__init__()
        
#         # 1. 加载预训练的 DenseNet121
#         self.model = DenseNet121(
#             spatial_dims=2,
#             in_channels=1,
#             out_channels=num_classes,
#             pretrained=True
#         )
        
#         # 2. 【核心修改】对抗过拟合：重构分类头 (Classifier Head)
#         # MONAI 的 DenseNet121 最后输出层名为 class_layers.out
#         # 我们获取它的输入特征数 (通常是 1024)
#         in_features = self.model.class_layers.out.in_features
        
#         # 用包含 Dropout 的新结构替换原本简单的 Linear 层
#         self.model.class_layers.out = nn.Sequential(
#             nn.Dropout(p=0.5),                # 50% 概率丢弃神经元，强力抑制过拟合
#             nn.Linear(in_features, num_classes)
#         )

#     def forward(self, x):
#         # 移除 Assert，提高训练效率并允许 CPU 调试
#         # 如果数据有问题，PyTorch 底层计算会自动报错，无需手动检查
#         output = self.model(x)
#         return output
    
# import timm
# import torch.nn as nn

# class OCT_Model(nn.Module):
#     def __init__(self, num_classes):
#         super(OCT_Model, self).__init__()
        
#         # 使用 timm 的 ResNet50
#         self.model = timm.create_model(
#             'resnet50',               # 模型名称
#             pretrained=True,          # ImageNet 预训练权重
#             in_chans=1,               # 单通道输入（灰度图）
#             num_classes=num_classes,  # 你的分类数
#             drop_rate=0.5            # 分类头前的 dropout
#         )
        
#     def forward(self, x):
#         return self.model(x)

# class OCT_Model(nn.Module):
#     def __init__(self, num_classes=7):
#         super(OCT_Model, self).__init__()
        
#         # 使用 timm 的 ConvNeXt
#         self.model = timm.create_model(
#             'convnext_tiny',           # 模型名称，可选择 convnext_tiny/base/large
#             pretrained=True,            # ImageNet 预训练权重
#             in_chans=1,                 # 单通道输入（灰度图）
#             num_classes=num_classes,    # 你的分类数
#             drop_rate=0.5,              # 分类头前的 dropout
#         )
#     def forward(self, x):
#         return self.model(x)
    
if __name__ == '__main__':
    # 创建模型
    model = OCT_Model(num_classes=7) # 6分类+OTHER=7
    # 测试输入
    batch_size = 2
    input_tensor = torch.randn(batch_size, 1, 256, 448)
    # 前向传播
    output = model(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print("Output shape:", output.shape)  # 应该是 (2, 7）
