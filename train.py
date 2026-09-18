import os
import argparse
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torchmetrics import Accuracy, Recall
from torchmetrics.classification import MulticlassJaccardIndex

from ds import OCT_dataset
from wrapper import OCTWrapper

# ====================== 训练配置 ======================
# 九分类类别名称（固定顺序，与ds.py中LABEL_MAP一致）
CLASS_NAMES = ['NORMAL', 'AMD', 'DR', 'ERM', 'MH', 'RRD', 'RVO', 'RAO', 'CSC']
NUM_CLASSES = len(CLASS_NAMES)

# cuDNN配置
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision('high')


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='OCT九分类模型训练')
    parser.add_argument('--no_shift', action='store_true',
                        help='取消SW-MSA（消融实验），默认使用标准Swin-T（含SW-MSA）')
    # 数据路径（必填）
    parser.add_argument('--xlsx', type=str, required=True,
                        help='训练数据Excel路径（需包含process_img_path, label_dict, datatag列）')
    # 训练超参数
    parser.add_argument('--batch_size', type=int, default=64, help='训练/验证批大小')
    parser.add_argument('--lr', type=float, default=5e-5, help='初始学习率（实际由wrapper中warmup调度）')
    parser.add_argument('--min_epochs', type=int, default=100, help='最小训练轮数')
    parser.add_argument('--max_epochs', type=int, default=200, help='最大训练轮数')
    parser.add_argument('--patience', type=int, default=50, help='早停耐心轮数')
    # 数据加载
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader进程数')
    parser.add_argument('--use_cache', action='store_true', help='启用图像缓存（默认关闭）')
    # 测试
    parser.add_argument('--test_checkpoint', type=str, default='last',
                        choices=['best', 'last'], help='测试时使用的检查点（best=验证loss最低，last=最后一轮）')
    # 输出
    parser.add_argument('--log_dir', type=str, default='./logs', help='日志保存目录')
    parser.add_argument('--model_dir', type=str, default='./Model', help='模型检查点保存目录')
    return parser.parse_args()


def main():
    args = parse_args()
    current_root = os.getcwd()

    # CUDA设置
    torch.set_float32_matmul_precision('high')
    torch.cuda.empty_cache()
    import resource
    resource.setrlimit(resource.RLIMIT_NOFILE, (65536, 65536))
    print(f"CUDA可用: {torch.cuda.is_available()}")
    print(f"配置: 分类数={NUM_CLASSES}, batch_size={args.batch_size}, lr={args.lr}, epochs={args.min_epochs}-{args.max_epochs}")

    # ====================== 1. 数据加载 ======================
    ds_train = OCT_dataset(args.xlsx, config={
        'mode': 'train', 'size': [256, 448],
        'is_debug': False, 'use_cache': args.use_cache,
        'cache_dir': './img_cache_train'
    })
    ds_valid = OCT_dataset(args.xlsx, config={
        'mode': 'valid', 'size': [256, 448],
        'is_debug': False, 'use_cache': args.use_cache,
        'cache_dir': './img_cache_valid'
    })
    ds_test = OCT_dataset(args.xlsx, config={
        'mode': 'test', 'size': [256, 448],
        'is_debug': False, 'test_samples': -1,
        'use_cache': args.use_cache, 'cache_dir': './img_cache_test'
    })

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          pin_memory=False, persistent_workers=True, num_workers=args.num_workers)
    dl_valid = DataLoader(ds_valid, batch_size=args.batch_size, shuffle=False,
                          pin_memory=False, persistent_workers=True, num_workers=args.num_workers)
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False,
                         pin_memory=False, persistent_workers=True, num_workers=args.num_workers)

    # ====================== 2. 模型初始化 ======================
    # 损失函数：标准交叉熵（类别不均衡通过加权采样处理，而非loss加权）
    loss_function = nn.CrossEntropyLoss()

    model = OCTWrapper(
        num_classes=NUM_CLASSES,
        class_names=CLASS_NAMES,
        metrics_dict={
            "acc": Accuracy(task='multiclass', num_classes=NUM_CLASSES),
            "recall": Recall(task='multiclass', num_classes=NUM_CLASSES, average='macro'),
            "jaccard": MulticlassJaccardIndex(num_classes=NUM_CLASSES),
        },
        loss_fn=loss_function,
        lr=args.lr,
        use_shift=not args.no_shift  # 默认True，加--no_shift则为False
    )

    # ====================== 3. 回调函数 ======================
    model_tag = 'noshift' if args.no_shift else 'standard'
    model_ckpt = ModelCheckpoint(
        dirpath=args.model_dir,
        filename=f'OCT_9cls_{model_tag}_model-{{epoch:02d}}-{{val_loss:.3f}}',
        monitor='val_loss',
        save_top_k=1,          # 只保存验证loss最低的1个模型
        mode='min',
        save_last=True,         # 额外保存最后一轮模型（方便断点续传）
        verbose=True,
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=args.patience,
        mode='min',
        verbose=True,
        check_on_train_epoch_end=False,
    )

    # 日志记录器（TensorBoard + CSV）
    tb_logger = TensorBoardLogger(save_dir=args.log_dir, name="oct_tensorboard")
    csv_logger = CSVLogger(save_dir=args.log_dir, name="oct_csv")

    # ====================== 4. 训练器 ======================
    trainer = pl.Trainer(
        logger=[tb_logger, csv_logger],
        precision="16-mixed",           # 混合精度训练
        enable_checkpointing=True,
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        accelerator='gpu',
        devices=[0],
        callbacks=[model_ckpt, early_stopping],
    )

    # ====================== 5. 启动训练 ======================
    print(f'开始训练 | 训练样本: {len(ds_train)} | 验证样本: {len(ds_valid)}')
    trainer.fit(model, dl_train, dl_valid)
    print('训练完成')

    # ====================== 6. 训练结束后自动测试 ======================
    print("\n" + "=" * 30)
    if args.test_checkpoint == 'best':
        target_ckpt = model_ckpt.best_model_path
        print(f"测试策略: BEST Model（验证Loss最低）")
    else:
        target_ckpt = model_ckpt.last_model_path
        if not target_ckpt and os.path.exists(os.path.join(model_ckpt.dirpath, 'last.ckpt')):
            target_ckpt = os.path.join(model_ckpt.dirpath, 'last.ckpt')
        print(f"测试策略: LAST Model（最后Epoch）")

    if target_ckpt and os.path.exists(target_ckpt):
        print(f"加载模型: {target_ckpt}")
        trainer.test(model, dataloaders=dl_test, ckpt_path=target_ckpt, weights_only=False)
    else:
        print(f"错误: 找不到检查点文件: {target_ckpt}")
    print("=" * 30 + "\n")


if __name__ == '__main__':
    main()
