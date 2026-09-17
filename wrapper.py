import os
from ds import OCT_dataset
from model import OCT_Model
import torch.nn as nn
from torch.optim import lr_scheduler
from monai.losses import FocalLoss
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks import one_hot
import pytorch_lightning as pl
from copy import deepcopy
import pandas as pd
import sys
import numpy as np
import datetime
from torchmetrics.classification import MulticlassJaccardIndex
from torchmetrics import Accuracy,Recall
import matplotlib.pyplot as plt
from torchmetrics import Accuracy, Recall, Precision, Specificity, F1Score, AUROC, ConfusionMatrix, ROC
import seaborn as sns # 用于画混淆矩阵
from sklearn.calibration import calibration_curve
from sklearn.metrics import det_curve
from torch.optim.lr_scheduler import LinearLR

try:
    trapz_func = np.trapezoid
except AttributeError:
    trapz_func = np.trapz

class OCTWrapper(pl.LightningModule):
    
    def __init__(self, num_classes=7, class_names=None, metrics_dict=None, loss_fn=None, lr=5e-5, test_output_dir='./testing_plots'):
        super(OCTWrapper, self).__init__()
        self.model = OCT_Model(num_classes=num_classes)
        if loss_fn is not None:# 允许外部传入损失函数（如带权重的交叉熵）20260514 修改
            self.loss_fn = loss_fn
        else:
            self.loss_fn = nn.CrossEntropyLoss()
        # self.loss_fn = nn.CrossEntropyLoss() 
        self.num_classes = num_classes
        self.lr = lr
        self.test_output_dir = test_output_dir

        # === 动态获取类别名称 ===
        if class_names is not None:
            self.class_names = class_names
        else:
            # 默认 fallback
            self.class_names = ['NORMAL', 'CNV', 'DME', 'DRUSEN', 'MH', 'DR', 'OTHERS']

        # 检查 num_classes 是否匹配
        if len(self.class_names) != self.num_classes:
            print(f"Warning: num_classes ({self.num_classes}) does not match len(class_names) ({len(self.class_names)})")

        # 建立名称到索引的映射
        self.name_to_idx = {name: i for i, name in enumerate(self.class_names)}

        # === 动态生成二分类 Pair ===
        self.binary_pairs = [] 
        # 1. 所有病种 vs NORMAL
        if 'NORMAL' in self.class_names:
            for name in self.class_names:
                if name != 'NORMAL':
                    self.binary_pairs.append((name, 'NORMAL'))
        
        # 2. 增加指定的额外对比
        extra_pairs = [('DRUSEN', 'CNV'), ('GA', 'CNV'), ('RVO', 'DME')]
        for pos, neg in extra_pairs:
            if pos in self.class_names and neg in self.class_names:
                self.binary_pairs.append((pos, neg))
        
        # 2. 定义分通过别指标
        self.val_class_acc = Accuracy(task="multiclass", num_classes=num_classes, average=None)
        self.val_class_recall = Recall(task="multiclass", num_classes=num_classes, average=None) 
        self.val_class_spec = Specificity(task="multiclass", num_classes=num_classes, average=None) 
        self.val_class_prec = Precision(task="multiclass", num_classes=num_classes, average=None) 
        self.val_class_f1 = F1Score(task="multiclass", num_classes=num_classes, average=None)
        self.val_auroc = AUROC(task="multiclass", num_classes=num_classes)
        self.val_conf_mat = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        self.val_roc = ROC(task="multiclass", num_classes=num_classes)
        self.val_global_roc = ROC(task="binary") 
        self.train_auroc = AUROC(task="multiclass", num_classes=num_classes)
        self.binary_auroc_metric = AUROC(task="binary") 
        self.best_val_loss = float('inf') 
        self.best_metrics_cache = None 
        self.error_cache = [] 

        # 3. 初始化绘图历史数据容器
        self.history_plot = {
            'train_loss': [], 'val_loss': [],
            'train_acc_global': [], 'train_auc_global': [],
            'val_acc_global': [], 'val_auc_global': [],
            'val_sens_global': [], 'val_spec_global': [], 
            'val_prec_global': [], 'val_f1_global': []
        }
        for name in self.class_names:
            self.history_plot[f'loss_{name}'] = []
            self.history_plot[f'acc_{name}'] = []
            self.history_plot[f'recall_{name}'] = []
            self.history_plot[f'spec_{name}'] = []
            self.history_plot[f'prec_{name}'] = []
            self.history_plot[f'f1_{name}'] = []

        for pos_name, neg_name in self.binary_pairs:
            pair_key = f"{pos_name}_vs_{neg_name}"
            for m in ['binary_loss', 'binary_acc', 'binary_auc']:
                self.history_plot[f'{m}_{pair_key}'] = []
            for m in ['train_binary_loss', 'train_binary_acc', 'train_binary_auc']:
                self.history_plot[f'{m}_{pair_key}'] = []

        self.class_loss_sum = {i: 0.0 for i in range(num_classes)}
        self.class_loss_count = {i: 0 for i in range(num_classes)}

        self.train_metrics = nn.ModuleDict(metrics_dict) if metrics_dict else nn.ModuleDict()
        self.val_metrics = nn.ModuleDict({k: deepcopy(v) for k, v in self.train_metrics.items()})
        self.test_metrics = nn.ModuleDict({k: deepcopy(v) for k, v in self.train_metrics.items()})
        
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []  
        self.save_hyperparameters(ignore=['loss_fn'])
        self.history = {}

    def configure_optimizers(self):
        base_lr = 1e-5  # 原5e-5 → 下调至1e-5
        warmup_epochs = 10
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=base_lr, weight_decay=2e-4)
        
        # 前10轮warmup线性升lr，避免预训练权重剧烈震荡
        warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
        # 原衰减调度
        reduce_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
        )
        # Lightning 标准返回格式：多调度器列表
        return [optimizer], [
            {
                "scheduler": warmup_scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": None  # 预热不需要监控loss，按epoch走
            },
            {
                "scheduler": reduce_scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
                "strict": False
            }
        ]
        # # 组合调度：先预热，再监控loss衰减
        # scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, reduce_scheduler], milestones=[warmup_epochs])
    
        # # optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        # # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        # #     optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-5
        # # )
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss", "interval": "epoch"} 
        # }

    def forward(self, x):    
        return self.model(x)   # 不强制转换
        # logits = self.model(x)   
        # return logits.float()   # 强制转为 float32

    def shared_step(self, batch, batch_idx):
        x, y = batch
        preds = self.model(x)          # 保持 float16（混合精度自动处理）
        loss = self.loss_fn(preds.float(), y)  # 仅损失计算时转 float32
        return {'loss': loss, 'preds': preds, 'y': y}   # 存储的 preds 仍为 half，节省显存
        # x, y = batch
        # preds = self.forward(x)   # 或 self.model(x).float()
        # loss = self.loss_fn(preds, y)
        # return {'loss': loss, 'preds': preds, 'y': y}  
    
    def training_step(self, batch, batch_idx):
        outputs = self.shared_step(batch, batch_idx)
        self.training_step_outputs.append(outputs)
        for name, metric in self.train_metrics.items():
            metric_value = metric(outputs['preds'], outputs['y'])
            self.log(f"train_{name}", metric_value, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_loss", outputs["loss"], prog_bar=True, on_step=False, on_epoch=True)
        return outputs
    
    def validation_step(self, batch, batch_idx):
        outputs = self.shared_step(batch, batch_idx)
        self.validation_step_outputs.append(outputs)
        logits = outputs['preds']
        target = outputs['y']
        preds = torch.argmax(logits, dim=1)  
        probs = torch.softmax(logits, dim=1)

        self.val_class_acc.update(preds, target)
        self.val_class_recall.update(preds, target)
        self.val_class_spec.update(preds, target)
        self.val_class_prec.update(preds, target)
        self.val_class_f1.update(preds, target)
        self.val_auroc.update(probs, target)
        self.val_conf_mat.update(preds, target)
        self.val_roc.update(probs, target)

        raw_losses = F.cross_entropy(logits, target, reduction='none')
        for i in range(self.num_classes):
            mask = (target == i)
            if mask.sum() > 0:
                self.class_loss_sum[i] += raw_losses[mask].sum().item()
                self.class_loss_count[i] += mask.sum().item()

        if len(self.error_cache) < 50:
            wrong_mask = preds != target
            if wrong_mask.any():
                idxs = torch.where(wrong_mask)[0]
                for idx in idxs:
                    if len(self.error_cache) >= 50: break
                    self.error_cache.append({
                        'img': batch[0][idx].detach().cpu(), 
                        'true': target[idx].item(),
                        'pred': preds[idx].item(),
                        'prob': probs[idx][preds[idx]].item()
                    })
        self.log("val_loss", outputs["loss"], prog_bar=True, on_step=False, on_epoch=True)
        return outputs
    
    # === test_step ===
    def test_step(self, batch, batch_idx):
        outputs = self.shared_step(batch, batch_idx)
        self.test_step_outputs.append(outputs)
        self.log("test_loss", outputs["loss"], prog_bar=True, on_step=False, on_epoch=True)
        return outputs
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if isinstance(batch, list) and len(batch) == 2: return self(batch[0])
        else: return self(batch)
    
    def on_train_epoch_end(self):
        if not self.training_step_outputs: return
        avg_loss = torch.stack([x['loss'] for x in self.training_step_outputs]).mean().item()
        self.history_plot['train_loss'].append(avg_loss)

        with torch.no_grad():
            all_preds = torch.cat([x['preds'].detach().cpu() for x in self.training_step_outputs])
            all_targets = torch.cat([x['y'].detach().cpu() for x in self.training_step_outputs])
            all_probs = torch.softmax(all_preds, dim=1)

            train_preds = torch.argmax(all_preds, dim=1)
            train_acc = (train_preds == all_targets).float().mean().item()
            self.history_plot['train_acc_global'].append(train_acc)

            try: train_auc = self.train_auroc(all_probs, all_targets).item()
            except: train_auc = 0.5
            self.train_auroc.reset()
            self.history_plot['train_auc_global'].append(train_auc)

            for pos_name, neg_name in self.binary_pairs:
                pos_idx = self.name_to_idx[pos_name]; neg_idx = self.name_to_idx[neg_name]
                pair_key = f"{pos_name}_vs_{neg_name}"
                mask = (all_targets == pos_idx) | (all_targets == neg_idx)
                if mask.sum() > 0:
                    sub_targets = (all_targets[mask] == pos_idx).long()
                    sub_logits = all_preds[mask][:, [neg_idx, pos_idx]]
                    loss_val = F.cross_entropy(sub_logits.float(), sub_targets).item()
                    sub_preds_cls = torch.argmax(sub_logits, dim=1)
                    acc_val = (sub_preds_cls == sub_targets).float().mean().item()
                    sub_probs = torch.softmax(sub_logits, dim=1)[:, 1]
                    try: auc_val = self.binary_auroc_metric(sub_probs, sub_targets).item()
                    except: auc_val = 0.5
                    self.binary_auroc_metric.reset()
                    self.history_plot[f'train_binary_loss_{pair_key}'].append(loss_val)
                    self.history_plot[f'train_binary_acc_{pair_key}'].append(acc_val)
                    self.history_plot[f'train_binary_auc_{pair_key}'].append(auc_val)
                else:
                    self.history_plot[f'train_binary_loss_{pair_key}'].append(0.0)
                    self.history_plot[f'train_binary_acc_{pair_key}'].append(0.0)
                    self.history_plot[f'train_binary_auc_{pair_key}'].append(0.0)
        self.training_step_outputs.clear() 

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs: return
        avg_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean().item()
        self.history_plot['val_loss'].append(avg_loss)
        
        accs = self.val_class_acc.compute().cpu().numpy()
        recalls = self.val_class_recall.compute().cpu().numpy()
        specs = self.val_class_spec.compute().cpu().numpy()
        precs = self.val_class_prec.compute().cpu().numpy()
        f1s = self.val_class_f1.compute().cpu().numpy()
        auc_score = self.val_auroc.compute().item()
        self.history_plot['val_auc_global'].append(auc_score)
        
        self.history_plot['val_acc_global'].append(accs.mean())
        self.history_plot['val_sens_global'].append(recalls.mean())
        self.history_plot['val_spec_global'].append(specs.mean())
        self.history_plot['val_prec_global'].append(precs.mean())
        self.history_plot['val_f1_global'].append(f1s.mean())
        self.val_auroc.reset()

        for i, name in enumerate(self.class_names):
            self.history_plot[f'acc_{name}'].append(accs[i])
            self.history_plot[f'recall_{name}'].append(recalls[i])
            self.history_plot[f'spec_{name}'].append(specs[i])
            self.history_plot[f'prec_{name}'].append(precs[i])
            self.history_plot[f'f1_{name}'].append(f1s[i])
            count = self.class_loss_count[i]
            val = self.class_loss_sum[i] / count if count > 0 else 0.0
            self.history_plot[f'loss_{name}'].append(val)
            self.log(f"Class_Recall/{name}", recalls[i], prog_bar=False)
            # self.log(f"Class_Acc/{name}", accs[i], prog_bar=False)
        
        all_logits = torch.cat([x['preds'] for x in self.validation_step_outputs]).cpu()
        all_targets = torch.cat([x['y'] for x in self.validation_step_outputs]).cpu()

        for pos_name, neg_name in self.binary_pairs:
            pos_idx = self.name_to_idx[pos_name]; neg_idx = self.name_to_idx[neg_name]
            pair_key = f"{pos_name}_vs_{neg_name}"
            mask = (all_targets == pos_idx) | (all_targets == neg_idx)
            if mask.sum() > 0:
                sub_targets = (all_targets[mask] == pos_idx).long()
                sub_logits = all_logits[mask][:, [neg_idx, pos_idx]]
                bin_loss = F.cross_entropy(sub_logits.float(), sub_targets).item()
                sub_probs = torch.softmax(sub_logits, dim=1)[:, 1]
                sub_preds = torch.argmax(sub_logits, dim=1)
                bin_acc = (sub_preds == sub_targets).float().mean().item()
                try: bin_auc = self.binary_auroc_metric(sub_probs, sub_targets).item()
                except: bin_auc = 0.5
                self.binary_auroc_metric.reset()
                self.history_plot[f'binary_loss_{pair_key}'].append(bin_loss)
                self.history_plot[f'binary_acc_{pair_key}'].append(bin_acc)
                self.history_plot[f'binary_auc_{pair_key}'].append(bin_auc)
            else:
                self.history_plot[f'binary_loss_{pair_key}'].append(0.0)
                self.history_plot[f'binary_acc_{pair_key}'].append(0.0)
                self.history_plot[f'binary_auc_{pair_key}'].append(0.0)

        current_conf_mat = self.val_conf_mat.compute().cpu().numpy()
        current_fprs, current_tprs, _ = self.val_roc.compute()
        
        per_class_auc = []
        for i in range(self.num_classes):
            _fpr = current_fprs[i].cpu().numpy()
            _tpr = current_tprs[i].cpu().numpy()
            per_class_auc.append(trapz_func(_tpr, _fpr))

        all_probs = torch.softmax(all_logits, dim=1)
        y_onehot_flat = F.one_hot(all_targets, num_classes=self.num_classes).view(-1)
        preds_flat = all_probs.view(-1)
        global_fpr, global_tpr, _ = self.val_global_roc(preds_flat, y_onehot_flat)
        global_roc_data = (global_fpr, global_tpr)
        self.val_global_roc.reset()

        current_metrics_pack = {
            'acc': accs,'recall': recalls, 'spec': specs, 'prec': precs, 'f1': f1s, 
            'auc_global': auc_score, 'auc_per_class': per_class_auc
        }

        self.save_confusion_matrix(current_conf_mat, file_prefix="08")
        self.save_roc_curve(current_fprs, current_tprs, global_roc_data=global_roc_data, file_prefix="09")
        self.save_calibration_curve(all_probs, all_targets, file_prefix="13")
        self.save_det_curve(all_targets, all_probs, file_prefix="14")
        self.save_error_analysis(current_conf_mat, file_prefix="15")

        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.best_metrics_cache = deepcopy(current_metrics_pack)
            print(f"New best model found at epoch {self.current_epoch} with loss {avg_loss:.4f}. Saving best plots.")
            self.save_confusion_matrix(current_conf_mat, file_prefix="Best")
            self.save_roc_curve(current_fprs, current_tprs, global_roc_data=global_roc_data, file_prefix="Best")
            self.save_calibration_curve(all_probs, all_targets, file_prefix="Best")
            self.save_det_curve(all_targets, all_probs, file_prefix="Best")
            self.save_error_analysis(current_conf_mat, file_prefix="Best")
            self.save_binary_roc_curves(all_logits, all_targets, file_prefix="Best")

        self.save_performance_table(current_metrics_pack)
        
        self.val_class_acc.reset(); self.val_class_recall.reset(); self.val_class_spec.reset()
        self.val_class_prec.reset(); self.val_class_f1.reset()
        self.val_conf_mat.reset(); self.val_roc.reset()
        self.class_loss_sum = {i: 0.0 for i in range(self.num_classes)}
        self.class_loss_count = {i: 0 for i in range(self.num_classes)}
        self.error_cache = [] 

        try: self.save_learning_curves()
        except Exception as e: print(f"绘图失败: {e}")
        
        self.validation_step_outputs.clear()
        self.print_bar()
        self.print(f"Epoch {self.current_epoch} Val Loss: {avg_loss:.4f}, AUC: {auc_score:.4f}")
    
    def on_test_epoch_end(self):
        """测试结束时调用：生成全套图表"""
        if not self.test_step_outputs: return
        
        print("\n正在生成测试结果图表...")
        
        # 1. 确定保存路径
        test_save_dir = self.test_output_dir
        os.makedirs(test_save_dir, exist_ok=True)
        print(f"测试结果将保存至: {test_save_dir}")
        
        # 2. 聚合数据 (保持在 GPU 上以计算指标)
        all_logits = torch.cat([x['preds'] for x in self.test_step_outputs]) # GPU
        all_targets = torch.cat([x['y'] for x in self.test_step_outputs])    # GPU
        all_probs = torch.softmax(all_logits, dim=1)
        all_preds = torch.argmax(all_logits, dim=1)
        
        # 3. 计算指标 (复用 val 的指标对象，先重置)
        self.val_class_acc.reset(); self.val_class_recall.reset()
        self.val_class_spec.reset(); self.val_class_prec.reset(); self.val_class_f1.reset()
        self.val_conf_mat.reset(); self.val_roc.reset(); self.val_global_roc.reset()
        self.val_auroc.reset()

        # Update
        self.val_class_acc.update(all_preds, all_targets)
        self.val_class_recall.update(all_preds, all_targets)
        self.val_class_spec.update(all_preds, all_targets)
        self.val_class_prec.update(all_preds, all_targets)
        self.val_class_f1.update(all_preds, all_targets)
        self.val_conf_mat.update(all_preds, all_targets)
        self.val_roc.update(all_probs, all_targets)
        self.val_auroc.update(all_probs, all_targets)

        # Compute (转 CPU Numpy)
        accs = self.val_class_acc.compute().cpu().numpy()
        recalls = self.val_class_recall.compute().cpu().numpy()
        specs = self.val_class_spec.compute().cpu().numpy()
        precs = self.val_class_prec.compute().cpu().numpy()
        f1s = self.val_class_f1.compute().cpu().numpy()
        auc_global = self.val_auroc.compute().item()
        conf_mat = self.val_conf_mat.compute().cpu().numpy()
        fprs, tprs, _ = self.val_roc.compute()
        
        # Global ROC data
        y_onehot = F.one_hot(all_targets, num_classes=self.num_classes).view(-1)
        probs_flat = all_probs.view(-1)
        g_fpr, g_tpr, _ = self.val_global_roc(probs_flat, y_onehot)
        global_roc_data = (g_fpr.cpu(), g_tpr.cpu())
        
        # Per-class AUC
        per_class_auc = [trapz_func(tprs[i].cpu().numpy(), fprs[i].cpu().numpy()) for i in range(self.num_classes)]

        # 创建 CPU 版本数据供绘图
        all_logits_cpu = all_logits.cpu()
        all_targets_cpu = all_targets.cpu()
        all_probs_cpu = all_probs.cpu()

        metrics_pack = {
            'acc': accs, 'recall': recalls, 'spec': specs, 'prec': precs, 'f1': f1s, 
            'auc_global': auc_global, 'auc_per_class': per_class_auc
        }

        # 4. 绘图 (传入 save_dir)
        self.save_confusion_matrix(conf_mat, file_prefix="Test", save_dir=test_save_dir)
        self.save_roc_curve(fprs, tprs, global_roc_data=global_roc_data, file_prefix="Test", save_dir=test_save_dir)
        self.save_binary_roc_curves(all_logits_cpu, all_targets_cpu, file_prefix="Test", save_dir=test_save_dir)
        self.save_det_curve(all_targets_cpu, all_probs_cpu, file_prefix="Test", save_dir=test_save_dir)
        self.save_calibration_curve(all_probs_cpu, all_targets_cpu, file_prefix="Test", save_dir=test_save_dir)
        # self.save_error_analysis(conf_mat, file_prefix="Test", save_dir=test_save_dir) # 需先在step中收集图片才有用，暂时注释
        
        # 5. 生成 CSV 表格
        self.save_test_performance_table(metrics_pack, save_dir=test_save_dir)

        self.save_class_metrics_bar_chart(metrics_pack, file_prefix="Test", save_dir=test_save_dir)

        print(f"测试完成! Global ACC: {accs.mean():.4f}, Global AUC: {auc_global:.4f}")
        self.test_step_outputs.clear()
            
    def shared_epoch_end(self, outputs, stage="train"):
        metrics = self.train_metrics if stage == "train" else (
            self.val_metrics if stage == "val" else self.test_metrics)
        epoch = self.current_epoch
        stage_loss = torch.stack([x['loss'] for x in outputs]).mean().item()
        dic = {"epoch": epoch, stage + "_loss": stage_loss}
        for name in metrics:
            epoch_metric = metrics[name].compute().item() 
            metrics[name].reset()
            dic[stage + "_" + name] = epoch_metric 
        if stage != 'test':
            self.history[epoch] = dict(self.history.get(epoch, {}), **dic)    
        return dic 
    
    # === 绘图函数增加 save_dir 参数 ===
    def save_confusion_matrix(self, cm, file_prefix="08", save_dir=None):
        if save_dir is None: save_dir = os.path.join(os.getcwd(), 'training_plots')
        os.makedirs(save_dir, exist_ok=True)
        style = self._get_plot_styles()
        plt.figure(figsize=(16, 14))
        sns.set_theme(font_scale=1.2)
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
        annot_labels = np.empty_like(cm, dtype=object)
        rows, cols = cm.shape
        for i in range(rows):
            for j in range(cols):
                count = cm[i, j]
                pct = cm_norm[i, j] * 100
                annot_labels[i, j] = f"{count}\n({pct:.1f}%)"
        ax = sns.heatmap(cm_norm, annot=annot_labels, fmt='', cmap='Blues', 
                         vmin=0.0, vmax=1.0,
                         xticklabels=self.class_names, yticklabels=self.class_names,
                         annot_kws={"size": 13})
        ax.set_xlabel('Predicted', fontsize=style['label_size']); ax.set_ylabel('True (Color based on Recall %)', fontsize=style['label_size'])
        ax.set_title(f'Confusion Matrix ({file_prefix})', fontsize=style['title_size'], fontweight='bold')
        plt.xticks(fontsize=style['tick_size'], rotation=45); plt.yticks(fontsize=style['tick_size'], rotation=0)
        # 调整 colorbar 的标签格式为百分比
        cbar = ax.collections[0].colorbar
        cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        cbar.set_ticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])

        plt.tight_layout()
        filename = '08_Best_Confusion_Matrix.png' if file_prefix == "Best" else f'08_{file_prefix}_Confusion_Matrix.png'
        if "Test" in file_prefix: filename = f'{file_prefix}_Confusion_Matrix.png'
        plt.savefig(os.path.join(save_dir, filename), dpi=style['dpi']); plt.close()

    def save_roc_curve(self, fprs, tprs, global_roc_data=None, file_prefix="09", save_dir=None):
        if save_dir is None: save_dir = os.path.join(os.getcwd(), 'training_plots')
        os.makedirs(save_dir, exist_ok=True)
        style = self._get_plot_styles()
        plt.figure(figsize=(14, 12))
        line_style = '--'
        for i, name in enumerate(self.class_names):
            fpr = fprs[i].cpu().numpy(); tpr = tprs[i].cpu().numpy()
            plt.plot(fpr, tpr, linestyle=line_style, label=f'{name} (AUC = {trapz_func(tpr, fpr):.2f})', linewidth=2)
        if global_roc_data is not None:
            g_fpr, g_tpr = global_roc_data
            plt.plot(g_fpr.numpy(), g_tpr.numpy(), color='black', linestyle='-', linewidth=2.5, label=f'GLOBAL (AUC = {trapz_func(g_tpr.numpy(), g_fpr.numpy()):.2f})')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=2)
        plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=style['label_size']); plt.ylabel('True Positive Rate', fontsize=style['label_size'])
        plt.title(f'ROC Curve ({file_prefix})', fontsize=style['title_size'], fontweight='bold')
        plt.xticks(fontsize=style['tick_size']); plt.yticks(fontsize=style['tick_size'])
        plt.legend(loc="lower right", fontsize=style['legend_size']); plt.grid(True, alpha=0.3)
        filename = '09_Best_ROC_Curve.png' if file_prefix == "Best" else f'09_{file_prefix}_ROC_Curve.png'
        if "Test" in file_prefix: filename = f'{file_prefix}_ROC_Curve.png'
        plt.savefig(os.path.join(save_dir, filename), dpi=style['dpi']); plt.close()
    
    def save_binary_roc_curves(self, all_logits, all_targets, file_prefix="12_Best", save_dir=None):
        if save_dir is None: save_dir = os.path.join(os.getcwd(), 'training_plots')
        os.makedirs(save_dir, exist_ok=True)
        style = self._get_plot_styles()
        batch_size = 6
        for page_idx, i in enumerate(range(0, len(self.binary_pairs), batch_size)):
            batch_pairs = self.binary_pairs[i : i + batch_size]
            fig, axes = plt.subplots(2, 3, figsize=(24, 16))
            fig.suptitle(f'Binary ROC Curves ({file_prefix}) - Page {page_idx + 1}', fontsize=style['title_size']+4, fontweight='bold')
            axes = axes.flatten()
            for j, (pos_name, neg_name) in enumerate(batch_pairs):
                ax = axes[j]; pos_idx = self.name_to_idx[pos_name]; neg_idx = self.name_to_idx[neg_name]
                mask = (all_targets == pos_idx) | (all_targets == neg_idx)
                if mask.sum() > 0:
                    sub_targets = (all_targets[mask] == pos_idx).long().numpy()
                    sub_logits = all_logits[mask][:, [neg_idx, pos_idx]]
                    sub_probs = torch.softmax(sub_logits, dim=1)[:, 1].numpy()
                    from sklearn.metrics import roc_curve, auc
                    fpr, tpr, _ = roc_curve(sub_targets, sub_probs)
                    ax.plot(fpr, tpr, color='darkorange', lw=3, label=f'AUC = {auc(fpr, tpr):.3f}')
                    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
                    ax.set_xlabel('False Positive Rate', fontsize=style['label_size']); ax.set_ylabel('True Positive Rate', fontsize=style['label_size'])
                    ax.set_title(f'{pos_name} vs {neg_name}', fontsize=style['title_size'])
                    ax.tick_params(labelsize=style['tick_size']); ax.legend(loc="lower right", fontsize=style['legend_size']); ax.grid(True)
                else: ax.text(0.5, 0.5, "No samples", ha='center', fontsize=style['title_size'])
            for k in range(len(batch_pairs), 6): axes[k].axis('off')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(os.path.join(save_dir, f'12_{file_prefix}_Binary_ROC_Page{page_idx + 1}.png'), dpi=style['dpi']); plt.close()

    def save_performance_table(self, current_metrics, save_dir=None):
        if save_dir is None: save_dir = os.path.join(os.getcwd(), 'training_plots')
        os.makedirs(save_dir, exist_ok=True)
        columns = ['Metric', 'Global (Macro)', 'Best Model (Macro)'] + self.class_names
        def fmt(val): return f"{val:.4f}"
        best_metrics = self.best_metrics_cache if self.best_metrics_cache else current_metrics
        rows = []
        metric_names = [ ("准确率 (Acc)", 'acc'), ("敏感性 (Sen)", 'recall'), ("特异性 (Spe)", 'spec'), ("精确率 (Pre)", 'prec'), ("F1-Score", 'f1')]
        for display_name, key in metric_names:
            curr_vec = current_metrics[key]; best_vec = best_metrics[key]
            row = [display_name, fmt(curr_vec.mean()), fmt(best_vec.mean())]
            for val in curr_vec: row.append(fmt(val))
            rows.append(row)
        curr_auc_cls = current_metrics['auc_per_class']
        row_auc = ["AUC", fmt(current_metrics['auc_global']), fmt(best_metrics['auc_global'])]
        for val in curr_auc_cls: row_auc.append(fmt(val))
        rows.append(row_auc)
        pd.DataFrame(rows, columns=columns).to_csv(os.path.join(save_dir, 'Performance_Summary_Table.csv'), index=False, encoding='utf-8-sig')

    def save_test_performance_table(self, metrics, save_dir=None):
        """生成测试集专用的性能表格"""
        if save_dir is None: save_dir = os.path.join(os.getcwd(), 'training_plots')
        os.makedirs(save_dir, exist_ok=True)
        columns = ['Metric', 'Test Global (Macro)'] + self.class_names
        def fmt(val): return f"{val:.4f}"
        rows = []
        metric_names = [ ("准确率 (Acc)", 'acc'), ("敏感性 (Sen)", 'recall'), ("特异性 (Spe)", 'spec'), ("精确率 (Pre)", 'prec'), ("F1-Score", 'f1')]
        for display_name, key in metric_names:
            vec = metrics[key]
            row = [display_name, fmt(vec.mean())]
            for val in vec: row.append(fmt(val))
            rows.append(row)
        row_auc = ["AUC", fmt(metrics['auc_global'])]
        for val in metrics['auc_per_class']: row_auc.append(fmt(val))
        rows.append(row_auc)
        pd.DataFrame(rows, columns=columns).to_csv(os.path.join(save_dir, 'Test_Performance_Table.csv'), index=False, encoding='utf-8-sig')

    def save_calibration_curve(self, probs, targets, file_prefix="15", save_dir=None):
        if save_dir is None: save_dir = os.path.join(os.getcwd(), 'training_plots')
        os.makedirs(save_dir, exist_ok=True)
        style = self._get_plot_styles()
        plt.figure(figsize=(12, 12))
        preds_flat = probs.view(-1).cpu().numpy()
        y_onehot_flat = F.one_hot(targets, num_classes=self.num_classes).view(-1).cpu().numpy()
        prob_true, prob_pred = calibration_curve(y_onehot_flat, preds_flat, n_bins=10, strategy='uniform')
        # ece = np.mean(np.abs(prob_true - prob_pred))
        ece = self.compute_ece(preds_flat, y_onehot_flat)
        plt.plot(prob_pred, prob_true, marker='o', linewidth=2.5, label=f'Global (ECE = {ece:.4f})')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated', linewidth=2)
        plt.xlabel('Mean Predicted Probability', fontsize=style['label_size']); plt.ylabel('Fraction of Positives', fontsize=style['label_size'])
        plt.title(f'Calibration Curve (Reliability Diagram) - {file_prefix}', fontsize=style['title_size'], fontweight='bold')
        plt.xticks(fontsize=style['tick_size']); plt.yticks(fontsize=style['tick_size']); plt.legend(fontsize=style['legend_size']); plt.grid(True)
        filename = '15_Best_Calibration_Curve.png' if file_prefix == "Best" else f'15_{file_prefix}_Calibration_Curve.png'
        if "Test" in file_prefix: filename = f'{file_prefix}_Calibration_Curve.png'
        plt.savefig(os.path.join(save_dir, filename), dpi=style['dpi']); plt.close()
        
    def save_det_curve(self, targets, probs, file_prefix="16", save_dir=None):
        if save_dir is None: save_dir = os.path.join(os.getcwd(), 'training_plots')
        os.makedirs(save_dir, exist_ok=True)
        style = self._get_plot_styles()
        plt.figure(figsize=(14, 12))
        line_style = '--' if file_prefix == "Best" else '-'
        for i, name in enumerate(self.class_names):
            y_true_binary = (targets == i).numpy()
            y_score_binary = probs[:, i].numpy()
            if len(np.unique(y_true_binary)) < 2:
                # print(f"Skipping DET curve for class {name}: Only one class present.")
                continue # 跳过该类别，不报错          
            try:
                fpr, fnr, _ = det_curve(y_true_binary, y_score_binary)
                plt.plot(fpr, fnr, linestyle=line_style, label=f'{name}', linewidth=2)
            except Exception as e:
                print(f"Error calculating DET for {name}: {e}")
                continue
        plt.xscale('log'); plt.yscale('log')
        plt.xlabel('False Positive Rate (False Alarm Rate)', fontsize=style['label_size'])
        plt.ylabel('False Negative Rate (Miss Rate)', fontsize=style['label_size'])
        plt.title(f'DET Curve ({file_prefix})', fontsize=style['title_size'], fontweight='bold')
        plt.xticks(fontsize=style['tick_size']); plt.yticks(fontsize=style['tick_size'])
        plt.grid(True, which="both", ls="-", alpha=0.5)
        
        # 防止所有类别都被跳过导致图例报错
        handles, labels = plt.gca().get_legend_handles_labels()
        if handles:
            plt.legend(fontsize=style['legend_size'], ncol=2)
        
        filename = '16_Best_DET_Curve.png' if file_prefix == "Best" else f'16_{file_prefix}_DET_Curve.png'
        if "Test" in file_prefix: filename = f'{file_prefix}_DET_Curve.png'
        plt.savefig(os.path.join(save_dir, filename), dpi=style['dpi'])
        plt.close()
        
    def save_error_analysis(self, conf_mat, file_prefix="17", save_dir=None):
        if not self.error_cache: return
        if save_dir is None: save_dir = os.path.join(os.getcwd(), 'training_plots')
        os.makedirs(save_dir, exist_ok=True)
        cm_no_diag = conf_mat.copy(); np.fill_diagonal(cm_no_diag, 0)
        top_k_indices = np.argsort(cm_no_diag.ravel())[-3:][::-1]
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle(f"Error Analysis: Top 3 Misclassified Pairs ({file_prefix})", fontsize=16)
        for rank, idx_flat in enumerate(top_k_indices):
            true_idx, pred_idx = np.unravel_index(idx_flat, conf_mat.shape)
            count = cm_no_diag[true_idx, pred_idx]
            true_name = self.class_names[true_idx]; pred_name = self.class_names[pred_idx]
            samples = [x for x in self.error_cache if x['true'] == true_idx and x['pred'] == pred_idx]
            for col in range(3):
                ax = axes[rank, col]
                if col < len(samples):
                    s = samples[col]
                    img = s['img'][0].numpy()
                    img = (img - img.min()) / (img.max() - img.min() + 1e-5)
                    ax.imshow(img, cmap='gray')
                    ax.set_title(f"T: {true_name} -> P: {pred_name}\nProb: {s['prob']:.2f}", fontsize=10, color='red')
                else: ax.text(0.5, 0.5, "No Sample", ha='center')
                ax.axis('off')
                if col == 0: ax.text(-0.2, 0.5, f"Rank {rank+1}\n{true_name}\nvs\n{pred_name}\n(Count: {count})", transform=ax.transAxes, va='center', ha='right', fontsize=12, fontweight='bold')
        plt.tight_layout(rect=[0.05, 0, 1, 0.95])
        filename = '17_Best_Error_Analysis.png' if file_prefix == "Best" else f'17_{file_prefix}_Error_Analysis.png'
        if "Test" in file_prefix: filename = f'{file_prefix}_Error_Analysis.png'
        plt.savefig(os.path.join(save_dir, filename)); plt.close()

    def _get_plot_styles(self):
        return {
            'title_size': 22, 'label_size': 20, 'tick_size': 16, 
            'legend_size': 16, 'dpi': 600, 'linewidth': 2.5
        }
    
    def save_learning_curves(self):
        save_dir = os.path.join(os.getcwd(), 'training_plots')
        os.makedirs(save_dir, exist_ok=True)
        style = self._get_plot_styles()
        
        def safe_plot_ax(data_key, label_name, ax, color=None):
            data = self.history_plot[data_key]
            if len(data) > 0:
                x = range(1, len(data) + 1)
                ax.plot(x, data, color, label=label_name, linewidth=style['linewidth']) if color else ax.plot(x, data, label=label_name, linewidth=style['linewidth'])

        def safe_plot_plt(data_key, label_name, color=None):
            data = self.history_plot[data_key]
            if len(data) > 0:
                x = range(1, len(data) + 1)
                plt.plot(x, data, color, label=label_name, linewidth=style['linewidth']) if color else plt.plot(x, data, label=label_name, linewidth=style['linewidth'])

        # === 1. Global Curves ===
        fig, axes = plt.subplots(3, 3, figsize=(26, 20))
        fig.suptitle('Global Performance Metrics', fontsize=style['title_size']+6, fontweight='bold')
        ax = axes.flatten()
        safe_plot_ax('train_loss', 'Train', ax[0], 'b-'); safe_plot_ax('val_loss', 'Valid', ax[0], 'r-')
        ax[0].set_title('Global Loss', fontsize=style['title_size']); ax[0].set_ylabel('Loss', fontsize=style['label_size'])
        safe_plot_ax('train_acc_global', 'Train', ax[1], 'b-'); safe_plot_ax('val_acc_global', 'Valid', ax[1], 'g-')
        ax[1].set_title('Global Accuracy', fontsize=style['title_size']); ax[1].set_ylabel('Accuracy', fontsize=style['label_size'])
        safe_plot_ax('train_auc_global', 'Train', ax[2], 'b-'); safe_plot_ax('val_auc_global', 'Valid', ax[2], 'm-')
        ax[2].set_title('Global AUC', fontsize=style['title_size']); ax[2].set_ylabel('AUC', fontsize=style['label_size'])
        safe_plot_ax('val_sens_global', 'Valid Sens', ax[3], 'c-')
        ax[3].set_title('Global Sensitivity (Macro)', fontsize=style['title_size']); ax[3].set_ylabel('Sensitivity', fontsize=style['label_size'])
        safe_plot_ax('val_spec_global', 'Valid Spec', ax[4], 'y-')
        ax[4].set_title('Global Specificity (Macro)', fontsize=style['title_size']); ax[4].set_ylabel('Specificity', fontsize=style['label_size'])
        safe_plot_ax('val_prec_global', 'Valid Prec', ax[5], 'k-')
        ax[5].set_title('Global Precision (Macro)', fontsize=style['title_size']); ax[5].set_ylabel('Precision', fontsize=style['label_size'])
        safe_plot_ax('val_f1_global', 'Valid F1', ax[6], 'purple')
        ax[6].set_title('Global F1-Score (Macro)', fontsize=style['title_size']); ax[6].set_ylabel('F1-Score', fontsize=style['label_size'])
        ax[7].axis('off'); ax[8].axis('off')
        for i in range(7):
            ax[i].set_xlabel('Epoch', fontsize=style['label_size']); ax[i].tick_params(labelsize=style['tick_size']); ax[i].grid(True); ax[i].legend(loc='best', fontsize=style['legend_size'])
        plt.tight_layout(rect=[0, 0.03, 1, 0.96]); plt.savefig(os.path.join(save_dir, '01_Global_Curves_9Grid.png'), dpi=style['dpi']); plt.close()

        # === 2. Per Class Metrics ===
        metrics_config = [
            ('02_Class_Loss.png', 'loss', 'Loss'), ('03_Class_Accuracy.png', 'acc', 'Accuracy'),
            ('04_Class_Sensitivity.png', 'recall', 'Sensitivity'), ('05_Class_Specificity.png', 'spec', 'Specificity'),
            ('06_Class_Precision.png', 'prec', 'Precision'), ('07_Class_F1_Score.png', 'f1', 'F1 Score')
        ]
        for fname, key_prefix, ylabel in metrics_config:
            plt.figure(figsize=(14, 10))
            for name in self.class_names: safe_plot_plt(f'{key_prefix}_{name}', name)
            plt.title(f'{ylabel} per Class', fontsize=style['title_size'], fontweight='bold')
            plt.xlabel('Epoch', fontsize=style['label_size']); plt.ylabel(ylabel, fontsize=style['label_size'])
            plt.xticks(fontsize=style['tick_size']); plt.yticks(fontsize=style['tick_size'])
            plt.legend(loc='best', fontsize=style['legend_size'], ncol=2); plt.grid(True); plt.tight_layout()
            plt.savefig(os.path.join(save_dir, fname), dpi=style['dpi']); plt.close()

        # === 3. Binary Curves ===      
        grid_configs = [
            ('binary_loss', 'Validation Loss', 'Binary_Val_Loss'), ('binary_acc', 'Validation Accuracy', 'Binary_Val_Acc'), ('binary_auc', 'Validation AUC', 'Binary_Val_AUC'),
            ('train_binary_loss', 'Training Loss', 'Binary_Train_Loss'), ('train_binary_acc', 'Training Accuracy', 'Binary_Train_Acc'), ('train_binary_auc', 'Training AUC', 'Binary_Train_AUC'),
        ]
        batch_size = 6
        for prefix, ylabel, file_base in grid_configs:
            for page_idx, i in enumerate(range(0, len(self.binary_pairs), batch_size)):
                batch_pairs = self.binary_pairs[i : i + batch_size]
                fig, axes = plt.subplots(2, 3, figsize=(24, 14))
                fig.suptitle(f'{ylabel} Comparison - Page {page_idx + 1}', fontsize=style['title_size']+4, fontweight='bold')
                axes = axes.flatten()
                for j, (pos, neg) in enumerate(batch_pairs):
                    ax = axes[j]; pair_key = f"{pos}_vs_{neg}"; full_key = f'{prefix}_{pair_key}'
                    data = self.history_plot[full_key]
                    if len(data) > 0:
                        ax.plot(range(1, len(data)+1), data, marker='.', label=f'{pos} vs {neg}', linewidth=2)
                        ax.set_title(f'{pos} vs {neg}', fontsize=style['title_size'])
                        ax.set_xlabel('Epoch', fontsize=style['label_size']); ax.set_ylabel(ylabel, fontsize=style['label_size'])
                        ax.tick_params(axis='both', which='major', labelsize=style['tick_size']); ax.grid(True); ax.legend(loc='best', fontsize=style['legend_size'])
                for k in range(len(batch_pairs), 6): axes[k].axis('off')
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.savefig(os.path.join(save_dir, f'12_{file_base}_Page{page_idx + 1}.png'), dpi=style['dpi']); plt.close()
        
        def save_composite(pairs, fname, title):
            fig, ax = plt.subplots(1, 3, figsize=(26, 8))
            for pos, neg in pairs: safe_plot_ax(f'binary_loss_{pos}_vs_{neg}', f'{pos}v{neg}', ax[0])
            ax[0].set_title(f'Loss ({title})', fontsize=style['title_size']); ax[0].set_xlabel('Epoch', fontsize=style['label_size']); ax[0].set_ylabel('Loss', fontsize=style['label_size']); ax[0].tick_params(labelsize=style['tick_size']); ax[0].grid(True); ax[0].legend(loc='best', fontsize=style['legend_size'])
            for pos, neg in pairs: safe_plot_ax(f'binary_acc_{pos}_vs_{neg}', f'{pos}v{neg}', ax[1])
            ax[1].set_title(f'Acc ({title})', fontsize=style['title_size']); ax[1].set_xlabel('Epoch', fontsize=style['label_size']); ax[1].set_ylabel('Acc', fontsize=style['label_size']); ax[1].tick_params(labelsize=style['tick_size']); ax[1].grid(True)
            for pos, neg in pairs: safe_plot_ax(f'binary_auc_{pos}_vs_{neg}', f'{pos}v{neg}', ax[2])
            ax[2].set_title(f'AUC ({title})', fontsize=style['title_size']); ax[2].set_xlabel('Epoch', fontsize=style['label_size']); ax[2].set_ylabel('AUC', fontsize=style['label_size']); ax[2].tick_params(labelsize=style['tick_size']); ax[2].grid(True)
            plt.tight_layout(); plt.savefig(os.path.join(save_dir, fname), dpi=style['dpi']); plt.close()
            
        save_composite([p for p in self.binary_pairs if p[1]=='NORMAL'], '10_Binary_Vs_Normal.png', 'Vs Normal')
        save_composite([p for p in self.binary_pairs if p[1]!='NORMAL'], '11_Binary_Others.png', 'Others')
    
    def save_class_metrics_bar_chart(self, metrics, file_prefix="Test", save_dir=None):
        """
        绘制各疾病的详细指标柱状图 (Acc, Sens, Spec, Prec, F1)
        """
        if save_dir is None: save_dir = os.path.join(os.getcwd(), 'training_plots')
        os.makedirs(save_dir, exist_ok=True)
        style = self._get_plot_styles()

        # 准备数据
        categories = self.class_names
        metric_names = ['Sensitivity', 'Specificity', 'F1-Score'] # Acc 通常是全局的，Per-Class Acc 意义不大，这里选最核心的
        
        # 提取数据 (假设 metrics 里的数据已经是 numpy 数组)
        # 注意：Recall = Sensitivity
        sens_vals = metrics['recall'] 
        spec_vals = metrics['spec']
        f1_vals = metrics['f1']
        
        x = np.arange(len(categories))  # 标签位置
        width = 0.25  # 柱子宽度

        plt.figure(figsize=(18, 10))
        
        # 绘制三组柱子
        rects1 = plt.bar(x - width, sens_vals, width, label='Sensitivity (Recall)', color='#1f77b4') # 蓝
        rects2 = plt.bar(x, spec_vals, width, label='Specificity', color='#ff7f0e') # 橙
        rects3 = plt.bar(x + width, f1_vals, width, label='F1-Score', color='#2ca02c') # 绿

        # 添加文本标签、标题等
        plt.ylabel('Score', fontsize=style['label_size'])
        plt.title(f'Per-Class Metrics Analysis ({file_prefix})', fontsize=style['title_size'], fontweight='bold')
        plt.xticks(x, categories, fontsize=style['tick_size'], rotation=45)
        plt.yticks(np.arange(0, 1.1, 0.1), fontsize=style['tick_size'])
        plt.legend(fontsize=style['legend_size'], loc='lower right')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.ylim(0, 1.05) # Y轴范围

        # 辅助函数：在柱子上方显示数值
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                plt.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=10, rotation=90)

        autolabel(rects1)
        autolabel(rects2)
        autolabel(rects3)

        plt.tight_layout()
        
        filename = f'{file_prefix}_PerClass_Metrics_Bar.png'
        plt.savefig(os.path.join(save_dir, filename), dpi=style['dpi'])
        plt.close()

    def get_history(self): return pd.DataFrame.from_dict(self.history, orient='index').sort_index()
    def print_bar(self): self.print("\n" + "=" * 80 + f" {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    def compute_ece(self, probs, labels, n_bins=10):
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probs > bin_lower) & (probs <= bin_upper)
            prop_in_bin = in_bin.mean()
            if prop_in_bin > 0:
                accuracy_in_bin = labels[in_bin].mean()
                avg_confidence_in_bin = probs[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece