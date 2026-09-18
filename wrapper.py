import os
import sys
import copy
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR
import pytorch_lightning as pl

from torchmetrics import Accuracy, Recall, Precision, Specificity, F1Score, AUROC, ConfusionMatrix, ROC
from sklearn.calibration import calibration_curve
from sklearn.metrics import det_curve

from model import OCT_Model

# numpy版本兼容：trapezoid函数在新版本中改名为np.trapezoid
try:
    trapz_func = np.trapezoid
except AttributeError:
    trapz_func = np.trapz


class OCTWrapper(pl.LightningModule):
    """
    PyTorch Lightning模型包装器，封装训练/验证/测试逻辑与可视化。
    
    Args:
        num_classes: 分类数，固定为9
        class_names: 类别名称列表
        metrics_dict: 训练时监控的指标字典
        loss_fn: 损失函数
        lr: 初始学习率
        test_output_dir: 测试结果图表保存目录
    """
    
    def __init__(self, num_classes=9, class_names=None, metrics_dict=None,
                 loss_fn=None, lr=5e-5, test_output_dir='./testing_plots', use_shift=True):
        super(OCTWrapper, self).__init__()
        self.model = OCT_Model(num_classes=num_classes, use_shift=use_shift)
        
        # 损失函数
        self.loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()
        
        self.num_classes = num_classes
        self.lr = lr
        self.test_output_dir = test_output_dir

        # 类别名称（默认九分类）
        if class_names is not None:
            self.class_names = class_names
        else:
            self.class_names = ['NORMAL', 'AMD', 'DR', 'ERM', 'MH', 'RRD', 'RVO', 'RAO', 'CSC']

        if len(self.class_names) != self.num_classes:
            print(f"Warning: num_classes ({self.num_classes}) != len(class_names) ({len(self.class_names)})")

        # ====================== 验证/测试指标 ======================
        self.val_class_acc = Accuracy(task="multiclass", num_classes=num_classes, average=None)
        self.val_class_recall = Recall(task="multiclass", num_classes=num_classes, average=None)
        self.val_class_spec = Specificity(task="multiclass", num_classes=num_classes, average=None)
        self.val_class_prec = Precision(task="multiclass", num_classes=num_classes, average=None)
        self.val_class_f1 = F1Score(task="multiclass", num_classes=num_classes, average=None)
        self.val_auroc = AUROC(task="multiclass", num_classes=num_classes)
        self.val_conf_mat = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        self.val_roc = ROC(task="multiclass", num_classes=num_classes)
        self.train_auroc = AUROC(task="multiclass", num_classes=num_classes)
        
        self.best_val_loss = float('inf')
        self.best_metrics_cache = None

        # ====================== 学习曲线历史数据 ======================
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

        # 各类别损失累计（用于计算每类平均损失）
        self.class_loss_sum = {i: 0.0 for i in range(num_classes)}
        self.class_loss_count = {i: 0 for i in range(num_classes)}

        # 训练/验证/测试指标（从外部传入）
        self.train_metrics = nn.ModuleDict(metrics_dict) if metrics_dict else nn.ModuleDict()
        self.val_metrics = nn.ModuleDict({k: copy.deepcopy(v) for k, v in self.train_metrics.items()})
        self.test_metrics = nn.ModuleDict({k: copy.deepcopy(v) for k, v in self.train_metrics.items()})
        
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.save_hyperparameters(ignore=['loss_fn'])
        self.history = {}

    # ====================== 优化器配置 ======================
    def configure_optimizers(self):
        """
        优化器配置：AdamW + 双学习率调度
        - 前10轮：线性warmup（从0.01x升到1x），避免预训练权重剧烈震荡
        - 之后：ReduceLROnPlateau（验证loss不下降时减半）
        """
        base_lr = 1e-5
        warmup_epochs = 10
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=base_lr, weight_decay=2e-4)
        
        # Warmup调度器
        warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
        # 衰减调度器
        reduce_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
        )
        return [optimizer], [
            {"scheduler": warmup_scheduler, "interval": "epoch", "frequency": 1, "monitor": None},
            {"scheduler": reduce_scheduler, "monitor": "val_loss", "interval": "epoch", "frequency": 1, "strict": False}
        ]

    def forward(self, x):
        return self.model(x)

    # ====================== 训练/验证/测试步骤 ======================
    def shared_step(self, batch, batch_idx):
        """共享的前向传播步骤"""
        x, y = batch
        preds = self.model(x)
        loss = self.loss_fn(preds.float(), y)  # 损失计算时转float32
        return {'loss': loss, 'preds': preds, 'y': y}

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

        # 更新各类别指标
        self.val_class_acc.update(preds, target)
        self.val_class_recall.update(preds, target)
        self.val_class_spec.update(preds, target)
        self.val_class_prec.update(preds, target)
        self.val_class_f1.update(preds, target)
        self.val_auroc.update(probs, target)
        self.val_conf_mat.update(preds, target)
        self.val_roc.update(probs, target)

        # 累计各类别损失
        raw_losses = F.cross_entropy(logits, target, reduction='none')
        for i in range(self.num_classes):
            mask = (target == i)
            if mask.sum() > 0:
                self.class_loss_sum[i] += raw_losses[mask].sum().item()
                self.class_loss_count[i] += mask.sum().item()

        self.log("val_loss", outputs["loss"], prog_bar=True, on_step=False, on_epoch=True)
        return outputs

    def test_step(self, batch, batch_idx):
        outputs = self.shared_step(batch, batch_idx)
        self.test_step_outputs.append(outputs)
        self.log("test_loss", outputs["loss"], prog_bar=True, on_step=False, on_epoch=True)
        return outputs

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if isinstance(batch, list) and len(batch) == 2:
            return self(batch[0])
        return self(batch)

    # ====================== Epoch结束回调 ======================
    def on_train_epoch_end(self):
        """训练轮结束：计算全局指标并记录历史"""
        if not self.training_step_outputs:
            return
        avg_loss = torch.stack([x['loss'] for x in self.training_step_outputs]).mean().item()
        self.history_plot['train_loss'].append(avg_loss)

        with torch.no_grad():
            all_preds = torch.cat([x['preds'].detach().cpu() for x in self.training_step_outputs])
            all_targets = torch.cat([x['y'].detach().cpu() for x in self.training_step_outputs])
            all_probs = torch.softmax(all_preds, dim=1)

            train_preds = torch.argmax(all_preds, dim=1)
            train_acc = (train_preds == all_targets).float().mean().item()
            self.history_plot['train_acc_global'].append(train_acc)

            try:
                train_auc = self.train_auroc(all_probs, all_targets).item()
            except Exception:
                train_auc = 0.5
            self.train_auroc.reset()
            self.history_plot['train_auc_global'].append(train_auc)

        self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        """
        验证轮结束：
        1. 计算各类别指标（Acc/Recall/Spec/Prec/F1/AUC）
        2. 生成可视化图表（混淆矩阵、ROC、校准曲线、DET曲线）
        3. 保存最佳模型的图表
        4. 记录学习曲线历史
        """
        if not self.validation_step_outputs:
            return
        avg_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean().item()
        self.history_plot['val_loss'].append(avg_loss)
        
        # 计算各类别指标
        accs = self.val_class_acc.compute().cpu().numpy()
        recalls = self.val_class_recall.compute().cpu().numpy()
        specs = self.val_class_spec.compute().cpu().numpy()
        precs = self.val_class_prec.compute().cpu().numpy()
        f1s = self.val_class_f1.compute().cpu().numpy()
        auc_score = self.val_auroc.compute().item()
        self.history_plot['val_auc_global'].append(auc_score)
        
        # 全局宏平均指标
        self.history_plot['val_acc_global'].append(accs.mean())
        self.history_plot['val_sens_global'].append(recalls.mean())
        self.history_plot['val_spec_global'].append(specs.mean())
        self.history_plot['val_prec_global'].append(precs.mean())
        self.history_plot['val_f1_global'].append(f1s.mean())
        self.val_auroc.reset()

        # 记录各类别指标历史
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
        
        # 聚合所有验证数据用于绘图
        all_logits = torch.cat([x['preds'] for x in self.validation_step_outputs]).cpu()
        all_targets = torch.cat([x['y'] for x in self.validation_step_outputs]).cpu()
        all_probs = torch.softmax(all_logits, dim=1)

        # 计算每类AUC（从ROC曲线积分）
        current_conf_mat = self.val_conf_mat.compute().cpu().numpy()
        current_fprs, current_tprs, _ = self.val_roc.compute()
        per_class_auc = []
        for i in range(self.num_classes):
            _fpr = current_fprs[i].cpu().numpy()
            _tpr = current_tprs[i].cpu().numpy()
            per_class_auc.append(trapz_func(_tpr, _fpr))

        current_metrics_pack = {
            'acc': accs, 'recall': recalls, 'spec': specs, 'prec': precs, 'f1': f1s,
            'auc_global': auc_score, 'auc_per_class': per_class_auc
        }

        # 生成当前轮图表
        self.save_confusion_matrix(current_conf_mat, file_prefix="08")
        self.save_roc_curve(current_fprs, current_tprs, file_prefix="09")
        self.save_calibration_curve(all_probs, all_targets, file_prefix="13")
        self.save_det_curve(all_targets, all_probs, file_prefix="14")

        # 保存最佳模型的图表
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.best_metrics_cache = copy.deepcopy(current_metrics_pack)
            print(f"New best model at epoch {self.current_epoch}, loss {avg_loss:.4f}. Saving best plots.")
            self.save_confusion_matrix(current_conf_mat, file_prefix="Best")
            self.save_roc_curve(current_fprs, current_tprs, file_prefix="Best")
            self.save_calibration_curve(all_probs, all_targets, file_prefix="Best")
            self.save_det_curve(all_targets, all_probs, file_prefix="Best")

        self.save_performance_table(current_metrics_pack)
        
        # 重置指标
        self.val_class_acc.reset(); self.val_class_recall.reset(); self.val_class_spec.reset()
        self.val_class_prec.reset(); self.val_class_f1.reset()
        self.val_conf_mat.reset(); self.val_roc.reset()
        self.class_loss_sum = {i: 0.0 for i in range(self.num_classes)}
        self.class_loss_count = {i: 0 for i in range(self.num_classes)}

        try:
            self.save_learning_curves()
        except Exception as e:
            print(f"绘图失败: {e}")
        
        self.validation_step_outputs.clear()
        self.print_bar()
        self.print(f"Epoch {self.current_epoch} Val Loss: {avg_loss:.4f}, AUC: {auc_score:.4f}")

    def on_test_epoch_end(self):
        """
        测试轮结束：
        1. 计算全套指标
        2. 生成测试结果图表（混淆矩阵、ROC、校准曲线、DET曲线）
        3. 保存性能表格CSV
        4. 保存类别指标柱状图
        """
        if not self.test_step_outputs:
            return
        
        print("\n正在生成测试结果图表...")
        test_save_dir = self.test_output_dir
        os.makedirs(test_save_dir, exist_ok=True)
        print(f"测试结果保存至: {test_save_dir}")
        
        # 聚合数据
        all_logits = torch.cat([x['preds'] for x in self.test_step_outputs])
        all_targets = torch.cat([x['y'] for x in self.test_step_outputs])
        all_probs = torch.softmax(all_logits, dim=1)
        all_preds = torch.argmax(all_logits, dim=1)
        
        # 重置并更新指标
        self.val_class_acc.reset(); self.val_class_recall.reset()
        self.val_class_spec.reset(); self.val_class_prec.reset(); self.val_class_f1.reset()
        self.val_conf_mat.reset(); self.val_roc.reset(); self.val_auroc.reset()

        self.val_class_acc.update(all_preds, all_targets)
        self.val_class_recall.update(all_preds, all_targets)
        self.val_class_spec.update(all_preds, all_targets)
        self.val_class_prec.update(all_preds, all_targets)
        self.val_class_f1.update(all_preds, all_targets)
        self.val_conf_mat.update(all_preds, all_targets)
        self.val_roc.update(all_probs, all_targets)
        self.val_auroc.update(all_probs, all_targets)

        # 计算指标
        accs = self.val_class_acc.compute().cpu().numpy()
        recalls = self.val_class_recall.compute().cpu().numpy()
        specs = self.val_class_spec.compute().cpu().numpy()
        precs = self.val_class_prec.compute().cpu().numpy()
        f1s = self.val_class_f1.compute().cpu().numpy()
        auc_global = self.val_auroc.compute().item()
        conf_mat = self.val_conf_mat.compute().cpu().numpy()
        fprs, tprs, _ = self.val_roc.compute()
        
        # 每类AUC
        per_class_auc = [trapz_func(tprs[i].cpu().numpy(), fprs[i].cpu().numpy()) for i in range(self.num_classes)]

        # CPU版本数据供绘图
        all_logits_cpu = all_logits.cpu()
        all_targets_cpu = all_targets.cpu()
        all_probs_cpu = all_probs.cpu()

        metrics_pack = {
            'acc': accs, 'recall': recalls, 'spec': specs, 'prec': precs, 'f1': f1s,
            'auc_global': auc_global, 'auc_per_class': per_class_auc
        }

        # 生成图表
        self.save_confusion_matrix(conf_mat, file_prefix="Test", save_dir=test_save_dir)
        self.save_roc_curve(fprs, tprs, file_prefix="Test", save_dir=test_save_dir)
        self.save_det_curve(all_targets_cpu, all_probs_cpu, file_prefix="Test", save_dir=test_save_dir)
        self.save_calibration_curve(all_probs_cpu, all_targets_cpu, file_prefix="Test", save_dir=test_save_dir)
        
        # 保存CSV表格和柱状图
        self.save_test_performance_table(metrics_pack, save_dir=test_save_dir)
        self.save_class_metrics_bar_chart(metrics_pack, file_prefix="Test", save_dir=test_save_dir)

        print(f"测试完成! Global ACC: {accs.mean():.4f}, Global AUC: {auc_global:.4f}")
        self.test_step_outputs.clear()

    # ====================== 绘图函数 ======================
    def _get_plot_styles(self):
        """统一绘图样式配置"""
        return {
            'title_size': 22, 'label_size': 20, 'tick_size': 16,
            'legend_size': 16, 'dpi': 600, 'linewidth': 2.5
        }

    def save_confusion_matrix(self, cm, file_prefix="08", save_dir=None):
        """保存混淆矩阵热力图（含样本数和百分比）"""
        if save_dir is None:
            save_dir = os.path.join(os.getcwd(), 'training_plots')
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
        ax.set_xlabel('Predicted', fontsize=style['label_size'])
        ax.set_ylabel('True (Color based on Recall %)', fontsize=style['label_size'])
        ax.set_title(f'Confusion Matrix ({file_prefix})', fontsize=style['title_size'], fontweight='bold')
        plt.xticks(fontsize=style['tick_size'], rotation=45)
        plt.yticks(fontsize=style['tick_size'], rotation=0)
        cbar = ax.collections[0].colorbar
        cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        cbar.set_ticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
        plt.tight_layout()
        filename = '08_Best_Confusion_Matrix.png' if file_prefix == "Best" else f'08_{file_prefix}_Confusion_Matrix.png'
        if "Test" in file_prefix:
            filename = f'{file_prefix}_Confusion_Matrix.png'
        plt.savefig(os.path.join(save_dir, filename), dpi=style['dpi'])
        plt.close()

    def save_roc_curve(self, fprs, tprs, file_prefix="09", save_dir=None):
        """保存各类别ROC曲线"""
        if save_dir is None:
            save_dir = os.path.join(os.getcwd(), 'training_plots')
        os.makedirs(save_dir, exist_ok=True)
        style = self._get_plot_styles()
        plt.figure(figsize=(14, 12))
        for i, name in enumerate(self.class_names):
            fpr = fprs[i].cpu().numpy()
            tpr = tprs[i].cpu().numpy()
            plt.plot(fpr, tpr, linestyle='--', label=f'{name} (AUC = {trapz_func(tpr, fpr):.2f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=2)
        plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=style['label_size'])
        plt.ylabel('True Positive Rate', fontsize=style['label_size'])
        plt.title(f'ROC Curve ({file_prefix})', fontsize=style['title_size'], fontweight='bold')
        plt.xticks(fontsize=style['tick_size']); plt.yticks(fontsize=style['tick_size'])
        plt.legend(loc="lower right", fontsize=style['legend_size']); plt.grid(True, alpha=0.3)
        filename = '09_Best_ROC_Curve.png' if file_prefix == "Best" else f'09_{file_prefix}_ROC_Curve.png'
        if "Test" in file_prefix:
            filename = f'{file_prefix}_ROC_Curve.png'
        plt.savefig(os.path.join(save_dir, filename), dpi=style['dpi'])
        plt.close()

    def save_calibration_curve(self, probs, targets, file_prefix="15", save_dir=None):
        """
        保存校准曲线（可靠性图），并计算ECE。
        ECE（Expected Calibration Error）：期望校准误差，衡量预测置信度与实际正确率的一致性。
        """
        if save_dir is None:
            save_dir = os.path.join(os.getcwd(), 'training_plots')
        os.makedirs(save_dir, exist_ok=True)
        style = self._get_plot_styles()
        plt.figure(figsize=(12, 12))
        preds_flat = probs.view(-1).cpu().numpy()
        y_onehot_flat = F.one_hot(targets, num_classes=self.num_classes).view(-1).cpu().numpy()
        prob_true, prob_pred = calibration_curve(y_onehot_flat, preds_flat, n_bins=10, strategy='uniform')
        ece = self.compute_ece(preds_flat, y_onehot_flat)
        plt.plot(prob_pred, prob_true, marker='o', linewidth=2.5, label=f'Global (ECE = {ece:.4f})')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated', linewidth=2)
        plt.xlabel('Mean Predicted Probability', fontsize=style['label_size'])
        plt.ylabel('Fraction of Positives', fontsize=style['label_size'])
        plt.title(f'Calibration Curve (Reliability Diagram) - {file_prefix}', fontsize=style['title_size'], fontweight='bold')
        plt.xticks(fontsize=style['tick_size']); plt.yticks(fontsize=style['tick_size'])
        plt.legend(fontsize=style['legend_size']); plt.grid(True)
        filename = '15_Best_Calibration_Curve.png' if file_prefix == "Best" else f'15_{file_prefix}_Calibration_Curve.png'
        if "Test" in file_prefix:
            filename = f'{file_prefix}_Calibration_Curve.png'
        plt.savefig(os.path.join(save_dir, filename), dpi=style['dpi'])
        plt.close()

    def save_det_curve(self, targets, probs, file_prefix="16", save_dir=None):
        """保存DET曲线（检测错误权衡图，对数坐标）"""
        if save_dir is None:
            save_dir = os.path.join(os.getcwd(), 'training_plots')
        os.makedirs(save_dir, exist_ok=True)
        style = self._get_plot_styles()
        plt.figure(figsize=(14, 12))
        line_style = '--' if file_prefix == "Best" else '-'
        for i, name in enumerate(self.class_names):
            y_true_binary = (targets == i).numpy()
            y_score_binary = probs[:, i].numpy()
            if len(np.unique(y_true_binary)) < 2:
                continue  # 只有一个类别时跳过
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
        handles, labels = plt.gca().get_legend_handles_labels()
        if handles:
            plt.legend(fontsize=style['legend_size'], ncol=2)
        filename = '16_Best_DET_Curve.png' if file_prefix == "Best" else f'16_{file_prefix}_DET_Curve.png'
        if "Test" in file_prefix:
            filename = f'{file_prefix}_DET_Curve.png'
        plt.savefig(os.path.join(save_dir, filename), dpi=style['dpi'])
        plt.close()

    def save_performance_table(self, current_metrics, save_dir=None):
        """保存验证集性能汇总表（当前 vs 最佳模型）"""
        if save_dir is None:
            save_dir = os.path.join(os.getcwd(), 'training_plots')
        os.makedirs(save_dir, exist_ok=True)
        columns = ['Metric', 'Global (Macro)', 'Best Model (Macro)'] + self.class_names
        def fmt(val): return f"{val:.4f}"
        best_metrics = self.best_metrics_cache if self.best_metrics_cache else current_metrics
        rows = []
        metric_names = [("准确率 (Acc)", 'acc'), ("敏感性 (Sen)", 'recall'),
                        ("特异性 (Spe)", 'spec'), ("精确率 (Pre)", 'prec'), ("F1-Score", 'f1')]
        for display_name, key in metric_names:
            curr_vec = current_metrics[key]; best_vec = best_metrics[key]
            row = [display_name, fmt(curr_vec.mean()), fmt(best_vec.mean())]
            for val in curr_vec: row.append(fmt(val))
            rows.append(row)
        curr_auc_cls = current_metrics['auc_per_class']
        row_auc = ["AUC", fmt(current_metrics['auc_global']), fmt(best_metrics['auc_global'])]
        for val in curr_auc_cls: row_auc.append(fmt(val))
        rows.append(row_auc)
        pd.DataFrame(rows, columns=columns).to_csv(
            os.path.join(save_dir, 'Performance_Summary_Table.csv'), index=False, encoding='utf-8-sig')

    def save_test_performance_table(self, metrics, save_dir=None):
        """保存测试集性能表"""
        if save_dir is None:
            save_dir = os.path.join(os.getcwd(), 'training_plots')
        os.makedirs(save_dir, exist_ok=True)
        columns = ['Metric', 'Test Global (Macro)'] + self.class_names
        def fmt(val): return f"{val:.4f}"
        rows = []
        metric_names = [("准确率 (Acc)", 'acc'), ("敏感性 (Sen)", 'recall'),
                        ("特异性 (Spe)", 'spec'), ("精确率 (Pre)", 'prec'), ("F1-Score", 'f1')]
        for display_name, key in metric_names:
            vec = metrics[key]
            row = [display_name, fmt(vec.mean())]
            for val in vec: row.append(fmt(val))
            rows.append(row)
        row_auc = ["AUC", fmt(metrics['auc_global'])]
        for val in metrics['auc_per_class']: row_auc.append(fmt(val))
        rows.append(row_auc)
        pd.DataFrame(rows, columns=columns).to_csv(
            os.path.join(save_dir, 'Test_Performance_Table.csv'), index=False, encoding='utf-8-sig')

    def save_class_metrics_bar_chart(self, metrics, file_prefix="Test", save_dir=None):
        """保存各类别Sensitivity/Specificity/F1柱状图"""
        if save_dir is None:
            save_dir = os.path.join(os.getcwd(), 'training_plots')
        os.makedirs(save_dir, exist_ok=True)
        style = self._get_plot_styles()
        categories = self.class_names
        sens_vals = metrics['recall']
        spec_vals = metrics['spec']
        f1_vals = metrics['f1']
        x = np.arange(len(categories))
        width = 0.25
        plt.figure(figsize=(18, 10))
        rects1 = plt.bar(x - width, sens_vals, width, label='Sensitivity (Recall)', color='#1f77b4')
        rects2 = plt.bar(x, spec_vals, width, label='Specificity', color='#ff7f0e')
        rects3 = plt.bar(x + width, f1_vals, width, label='F1-Score', color='#2ca02c')
        plt.ylabel('Score', fontsize=style['label_size'])
        plt.title(f'Per-Class Metrics Analysis ({file_prefix})', fontsize=style['title_size'], fontweight='bold')
        plt.xticks(x, categories, fontsize=style['tick_size'], rotation=45)
        plt.yticks(np.arange(0, 1.1, 0.1), fontsize=style['tick_size'])
        plt.legend(fontsize=style['legend_size'], loc='lower right')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.ylim(0, 1.05)
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                plt.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10, rotation=90)
        autolabel(rects1); autolabel(rects2); autolabel(rects3)
        plt.tight_layout()
        filename = f'{file_prefix}_PerClass_Metrics_Bar.png'
        plt.savefig(os.path.join(save_dir, filename), dpi=style['dpi'])
        plt.close()

    def save_learning_curves(self):
        """保存学习曲线（全局指标 + 各类别指标）"""
        save_dir = os.path.join(os.getcwd(), 'training_plots')
        os.makedirs(save_dir, exist_ok=True)
        style = self._get_plot_styles()
        
        def safe_plot_ax(data_key, label_name, ax, color=None):
            data = self.history_plot[data_key]
            if len(data) > 0:
                x = range(1, len(data) + 1)
                if color:
                    ax.plot(x, data, color, label=label_name, linewidth=style['linewidth'])
                else:
                    ax.plot(x, data, label=label_name, linewidth=style['linewidth'])

        def safe_plot_plt(data_key, label_name, color=None):
            data = self.history_plot[data_key]
            if len(data) > 0:
                x = range(1, len(data) + 1)
                if color:
                    plt.plot(x, data, color, label=label_name, linewidth=style['linewidth'])
                else:
                    plt.plot(x, data, label=label_name, linewidth=style['linewidth'])

        # === 全局指标（3x3网格） ===
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
            ax[i].set_xlabel('Epoch', fontsize=style['label_size'])
            ax[i].tick_params(labelsize=style['tick_size'])
            ax[i].grid(True); ax[i].legend(loc='best', fontsize=style['legend_size'])
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.savefig(os.path.join(save_dir, '01_Global_Curves_9Grid.png'), dpi=style['dpi'])
        plt.close()

        # === 各类别指标 ===
        metrics_config = [
            ('02_Class_Loss.png', 'loss', 'Loss'), ('03_Class_Accuracy.png', 'acc', 'Accuracy'),
            ('04_Class_Sensitivity.png', 'recall', 'Sensitivity'), ('05_Class_Specificity.png', 'spec', 'Specificity'),
            ('06_Class_Precision.png', 'prec', 'Precision'), ('07_Class_F1_Score.png', 'f1', 'F1 Score')
        ]
        for fname, key_prefix, ylabel in metrics_config:
            plt.figure(figsize=(14, 10))
            for name in self.class_names:
                safe_plot_plt(f'{key_prefix}_{name}', name)
            plt.title(f'{ylabel} per Class', fontsize=style['title_size'], fontweight='bold')
            plt.xlabel('Epoch', fontsize=style['label_size']); plt.ylabel(ylabel, fontsize=style['label_size'])
            plt.xticks(fontsize=style['tick_size']); plt.yticks(fontsize=style['tick_size'])
            plt.legend(loc='best', fontsize=style['legend_size'], ncol=2); plt.grid(True); plt.tight_layout()
            plt.savefig(os.path.join(save_dir, fname), dpi=style['dpi'])
            plt.close()

    # ====================== 工具方法 ======================
    def compute_ece(self, probs, labels, n_bins=10):
        """
        计算期望校准误差（Expected Calibration Error, ECE）。
        ECE = sum( |bin_acc - bin_conf| * bin_prop )
        
        Args:
            probs: 预测概率（展平后）
            labels: 真实标签（one-hot展平后）
            n_bins: 分箱数
        """
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

    def get_history(self):
        return pd.DataFrame.from_dict(self.history, orient='index').sort_index()

    def print_bar(self):
        self.print("\n" + "=" * 80 + f" {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
