import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
current_root = os.getcwd()
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
# import sys
# sys.path.append(os.getcwd())
from ds import OCT_dataset
from torch.utils.data import DataLoader
from monai.losses import FocalLoss
from monai.metrics import DiceMetric, MeanIoU
from wrapper import OCTWrapper
from pytorch_lightning.loggers import CSVLogger
import pytorch_lightning as pl 
from torchmetrics import Accuracy,Recall
from torchmetrics.classification import BinaryJaccardIndex, JaccardIndex
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping
import torch.nn as nn
from torchmetrics.classification import MulticlassJaccardIndex
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from monai.data import DataLoader as MONAI_DataLoader

torch.backends.cudnn.benchmark = True  # 禁用cudnn自动优化（避免部分版本兼容问题）
torch.backends.cudnn.deterministic = False  # 确保结果可复现
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

torch.set_float32_matmul_precision('high')  # 使用高精度

train_batch_size = 64
valid_batch_size = 64
lr = 5e-5 #1e-5

# 跑9分类，只需把这里改为 9
RAW_DISEASE_NUM = 9 

# === 动态定义类别名称 ===
if RAW_DISEASE_NUM == 6:
    # 6分类 + OTHERS
    CLASS_NAMES = ['NORMAL', 'CNV', 'DME', 'DRUSEN', 'MH', 'DR', 'OTHERS']
elif RAW_DISEASE_NUM == 9:
    # 9分类 + OTHERS (假设的9分类顺序，请根据实际情况调整)
    CLASS_NAMES = ['NORMAL', 'AMD', 'DR', 'ERM', 'MH', 'RRD', 'RVO', 'RAO', 'CSC']
elif RAW_DISEASE_NUM == 13:
    CLASS_NAMES = ['NORMAL', 'CNV', 'DME', 'DRUSEN', 'MH', 'DR', 'AMD', 'CSC', 'ERM', 'RVO', 'VID', 'RRD', 'RAO']
elif RAW_DISEASE_NUM == 11:
    # 11分类，假设排除 AMD 和 VID，其他11类不变
    CLASS_NAMES = ['NORMAL', 'CNV', 'DME', 'DRUSEN', 'MH', 'DR', 
                   'CSC', 'ERM', 'RVO', 'RRD', 'RAO']
elif RAW_DISEASE_NUM == 12: 
    # 12分类，假设排除 AMD，其他12类不变
    CLASS_NAMES = ['NORMAL', 'DRUSEN', 'GA', 'CNV', 'DME', 'DR', 
                'ERM', 'MH', 'RRD', 'RVO', 'RAO', 'CSC']
else:
    raise ValueError("不支持的分类数量，请检查 RAW_DISEASE_NUM")

NUM_CLASSES = len(CLASS_NAMES)

# 可选: 'best' (验证集Loss最低) 或 'last' (最后一个Epoch)
TEST_CHECKPOINT_MODE = 'last' 

print(f"Config: Raw Num={RAW_DISEASE_NUM}, Model Output Classes={NUM_CLASSES}")
num_classes = NUM_CLASSES
print("train_batch_size:",train_batch_size)
print("valid_batch_size:",valid_batch_size)
print("lr:", lr) 
print("num_classes:", num_classes)

if __name__ == '__main__':
    # === [修改1] CUDA 设置移入主函数 ===
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0' 修改1
    torch.set_float32_matmul_precision('high')
    torch.cuda.empty_cache()
    import resource
    resource.setrlimit(resource.RLIMIT_NOFILE, (65536, 65536))
    print("cuda:",torch.cuda.is_available())

    # 1. 数据预处理（包括数据增广）
    train_xlsx = '/data/chenxinfen/OCT1/new9/视网膜疾病分类_train_splitV7.xlsx'
    log_dir = os.path.join(current_root, "logs")

    # 实例化 Train / Valid / Test 数据集
    ds_train = OCT_dataset(train_xlsx, config={'mode':'train','num':RAW_DISEASE_NUM,'size':[256,448],'is_debug':False,'use_cache':False, 'cache_dir':'./ima_cache_train'})
    ds_valid = OCT_dataset(train_xlsx, config={'mode':'valid','num':RAW_DISEASE_NUM,'size':[256,448],'is_debug':False,'use_cache':False, 'cache_dir':'./ima_cache_valid'})
    ds_test = OCT_dataset(train_xlsx, config={'mode':'test','num':RAW_DISEASE_NUM,'size':[256,448],'is_debug':False,'test_samples': -1,'use_cache':False, 'cache_dir':'./ima_cache_test'})
    # 2. 将预处理的数据预加载到dataloader中，无需修改
    dl_train = DataLoader(ds_train,
                          batch_size=train_batch_size,
                          shuffle=True,  # 随机打乱
                          pin_memory=False,  # 修改 2 关闭 pin_memory，避免部分系统报错
                          persistent_workers=True,  # 保持worker进程存活
                          num_workers=4)   # 修改 3 16→4
    dl_valid = DataLoader(ds_valid,
                          batch_size=valid_batch_size,
                          shuffle=False,  # 验证集不打乱
                          pin_memory=False,  # 修改 2 关闭 pin_memory，避免部分系统报错
                          persistent_workers=True,  # 保持worker进程存活
                          num_workers=4)   # 修改 3 16→4
    dl_test = DataLoader(ds_test,
                          batch_size=valid_batch_size,
                          shuffle=False,  # 测试集不打乱
                          pin_memory=False,  # 修改 2 关闭 pin_memory，避免部分系统报错
                          persistent_workers=True,  # 保持worker进程存活
                          num_workers=4)  # 修改 3 16→4

   # 3. 初始化模型
    loss_function = nn.CrossEntropyLoss()

    # if RAW_DISEASE_NUM == 13:
    #         print("检测到 13 分类模式，正在应用类别平衡权重...")
    #         # 权重计算公式: Total / (Num_Classes * Count)
    #         # 防止 RAO(22张) 权重过大(500+)，这里进行了适度平滑处理或直接使用
    #         weights_tensor = torch.tensor([
    #             0.21,   # NORMAL (57963)
    #             0.28,   # CNV    (40455)
    #             0.69,   # DME    (16982)
    #             1.00,   # DRUSEN (11866)
    #             1.21,   # MH     (9619)
    #             3.88,   # DR     (3107)
    #             1.25,   # AMD    (9534)
    #             3.89,   # CSC    (3102)
    #             68.33,  # ERM    (155)  -> 重点关注
    #             125.43, # RVO    (101)  -> 重点关注
    #             144.90, # VID    (76)   -> 重点关注
    #             186.77, # RRD    (69)   -> 重点关注
    #             525.30  # RAO    (22)   -> 极度稀缺，赋予最大权重
    #         ], dtype=torch.float32)
    #         loss_function = nn.CrossEntropyLoss(weight=weights_tensor)

    model = OCTWrapper(
        num_classes=num_classes,
        class_names=CLASS_NAMES,
        metrics_dict={"acc":Accuracy(task='multiclass', num_classes=num_classes),
                      "recall": Recall(task='multiclass', num_classes=num_classes, average='macro'),
                      "jaccard":MulticlassJaccardIndex(num_classes=num_classes),
                      },
        loss_fn=loss_function,  # 使用 PyTorch 的 CrossEntropyLoss
        # loss_fn=FocalLoss(include_background=True, to_onehot_y=True, gamma=2.0) # 分类类型
        )

    # 4. 设置回调函数
    model_ckpt = ModelCheckpoint(
        dirpath=os.path.join(current_root, "Model"),
        filename=f'OCT_{RAW_DISEASE_NUM}cls_model-{{epoch:02d}}-{{val_loss:.3f}}', # 'OCT_classification_model',
        monitor='val_loss',
        save_top_k=1, # 只保存最好的1个模型
        mode='min',  # loss越小越好
        save_last=True,   # 额外保存最后一次迭代的模型(方便断点续传)
        verbose=True,
    )

    early_stopping = EarlyStopping(
        monitor = 'val_loss',
        patience = 50, # 100个epoch不下降则停止
        mode = 'min',              # 损失越小越好
        verbose=True,
        check_on_train_epoch_end=False,  # 默认在验证后检查
    )
    # 定义日志记录器
    # logger = CSVLogger(save_dir=log_dir, name="oct_logs")
    #定义 Logger (同时使用 TensorBoard 和 CSV)
    tb_logger = TensorBoardLogger(save_dir=log_dir, name="oct_tensorboard")
    csv_logger = CSVLogger(save_dir=log_dir, name="oct_csv")
    # 5. 设置训练参数
    trainer = pl.Trainer( 
                        # limit_train_batches=5,    # 每个 epoch 只跑 5 个 batch
                        # limit_val_batches=5,      # 每个 epoch 只验证 5 个 batch
                        # max_epochs=1,             # 只跑 1 个 epoch
                        logger=[tb_logger, csv_logger], # 开启日志
                        precision="16-mixed", # 修改四 16-mixed → 32-true
                        enable_checkpointing=True,  # 开启模型保存
                        min_epochs=100, max_epochs=200, # 设定最大训练轮数，例如 100 或 200
                        accelerator='gpu', # 修改五 gpu → cpu
                        devices=[0], # 修改五 [0] → 1
                        callbacks = [model_ckpt,early_stopping], #以此启用回调,early_stopping
                        # logger = CSVLogger('/data/personal/chenxinfei/Model/logs'),
                        # enable_progress_bar =False,
                        ) 

    # resume_ckpt = '/home/chenxinfei/ModelTraining_monai/Model/last-v44.ckpt'  # 修改六 指定断点续传的模型路径
    # trainer.fit(model, dl_train, dl_valid, ckpt_path=resume_ckpt)

    # 6. 启动训练循环
    print('start training')
    print(f'Training samples: {len(ds_train)}') # 每个 epoch 实际训练样本数
    print(f'Validation samples: {len(ds_valid)}') # 每个 epoch 实际验证样本数
    trainer.fit(model,dl_train,dl_valid) # ,ckpt_path="/data/personal/chenxinfei/Model/OCT_model-epoch=00-val_loss=1.37.ckpt"
    print('done training') 

    # === [修改5] 训练结束后自动进行测试 ===
    print("\n" + "="*30)
    
    target_ckpt = ""
    if TEST_CHECKPOINT_MODE == 'best':
        target_ckpt = model_ckpt.best_model_path
        print(f"测试策略: BEST Model (Loss最小)")
    else:
        # last model 通常命名为 last.ckpt，或者通过 model_ckpt.last_model_path 获取
        target_ckpt = model_ckpt.last_model_path
        if not target_ckpt and os.path.exists(os.path.join(model_ckpt.dirpath, 'last.ckpt')):
             target_ckpt = os.path.join(model_ckpt.dirpath, 'last.ckpt')
        print(f"测试策略: LAST Model (最后Epoch)")

    if target_ckpt and os.path.exists(target_ckpt):
        print(f"加载模型路径: {target_ckpt}")
        trainer.test(model, dataloaders=dl_test, ckpt_path=target_ckpt, weights_only=False)
    else:
        print(f"错误: 找不到 Checkpoint 文件: {target_ckpt}")
    
    print("="*30 + "\n")