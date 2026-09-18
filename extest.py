import os
import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from monai.transforms import Compose, ScaleIntensityd, ToTensord
from SimpleITK import ReadImage, GetArrayFromImage

from wrapper import OCTWrapper
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# ==================== 九分类定义 ====================
CLASS_NAMES = ['NORMAL', 'AMD', 'DR', 'ERM', 'MH', 'RRD', 'RVO', 'RAO', 'CSC']
NUM_CLASSES = len(CLASS_NAMES)
NAME_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}


# ==================== 图像预处理函数 ====================
def detect_orientation(image: np.ndarray) -> np.ndarray:
    """
    检测图像方向并统一为水平分层（视网膜层理水平）。
    通过比较行/列像素和的方差判断方向，若垂直方向方差更大则旋转90度。
    """
    h, w = image.shape
    if w / h > 2.0:
        return image

    # 中心裁剪60%区域进行判断，避免边缘干扰
    h_start, h_end = int(h * 0.2), int(h * 0.8)
    w_start, w_end = int(w * 0.2), int(w * 0.8)
    if h_end - h_start < 10 or w_end - w_start < 10:
        center_img = image
    else:
        center_img = image[h_start:h_end, w_start:w_end]

    img_float = center_img.astype(np.float32)
    min_val, max_val = img_float.min(), img_float.max()
    if max_val > min_val:
        img_float = (img_float - min_val) / (max_val - min_val)

    row_profile_var = np.var(np.sum(img_float, axis=1))
    col_profile_var = np.var(np.sum(img_float, axis=0))

    # 列方差显著大于行方差（>1.2倍），说明层理垂直，需要旋转
    if col_profile_var > row_profile_var * 1.2:
        image = np.rot90(image, k=-1)
    return image


def adaptive_resize(image, target_h=256, target_w=448, fill_value=0):
    """
    根据输入图像宽高比与目标宽高比的关系，自适应裁剪或填充，再缩放到目标尺寸。
    - 图像更扁（宽高比 > 目标）：裁剪左右
    - 图像更窄（宽高比 <= 目标）：填充左右黑边
    """
    h, w = image.shape
    target_aspect = target_w / target_h
    img_aspect = w / h

    if img_aspect > target_aspect:
        # 图像更扁：裁剪左右
        new_w = int(target_aspect * h)
        left = (w - new_w) // 2
        right = w - new_w - left
        cropped = image[:, left:w-right]
        resized = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    else:
        # 图像更窄或相等：填充左右黑边
        new_w = int(h * target_aspect)
        pad_w = new_w - w
        left = pad_w // 2
        right = pad_w - left
        padded = cv2.copyMakeBorder(image, 0, 0, left, right,
                                    cv2.BORDER_CONSTANT, value=fill_value)
        resized = cv2.resize(padded, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return resized


# ==================== 外部测试数据集 ====================
class ExternalTestDataset(Dataset):
    """
    外部测试数据集：从Excel读取图像路径和标签，执行预处理流水线。
    
    Args:
        df: 包含 process_img_path 和 label_dict 列的DataFrame
        name_to_idx: 标签名称到索引的映射
        img_size: 目标图像尺寸 (H, W)，默认 (256, 448)
    """
    def __init__(self, df, name_to_idx, img_size=(256, 448)):
        self.df = df.reset_index(drop=True)
        self.name_to_idx = name_to_idx
        self.img_size = img_size
        self.transform = Compose([
            ScaleIntensityd(keys=['image'], minv=0.0, maxv=1.0),
            ToTensord(keys=['image']),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['process_img_path']
        label_name = row['label_dict']
        label = self.name_to_idx[label_name]
        case_id = row.get('case_id', str(idx))

        # 读取图像
        img = ReadImage(img_path)
        img_array = GetArrayFromImage(img)
        if len(img_array.shape) == 3:
            img_array = img_array[0]

        # 预处理流水线：方向校正 -> 自适应缩放
        img_array = detect_orientation(img_array)
        img_array = adaptive_resize(img_array, self.img_size[0], self.img_size[1])

        # 增加通道维度并标准化
        img_array = np.expand_dims(img_array, axis=0)
        data_dict = {'image': img_array}
        data_dict = self.transform(data_dict)
        img_tensor = data_dict['image']

        return img_tensor, torch.tensor(label, dtype=torch.long), img_path, case_id


# ==================== 指标计算 ====================
def compute_metrics(all_labels, all_preds, all_probs, class_names):
    """
    计算每类的 Accuracy, Precision, Recall, F1, AUC。
    混淆矩阵指定labels为全部9类，确保矩阵为9x9（未出现的类别补零）。
    """
    n_classes = len(class_names)
    all_labels = all_labels.astype(int)
    all_preds = all_preds.astype(int)

    # 混淆矩阵（9x9）
    cm = confusion_matrix(all_labels, all_preds, labels=range(n_classes))

    # 分类报告
    report = classification_report(
        all_labels, all_preds,
        labels=range(n_classes),
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    # 每类AUC（one-hot编码）
    n_samples = len(all_labels)
    all_labels_onehot = np.zeros((n_samples, n_classes))
    all_labels_onehot[np.arange(n_samples), all_labels] = 1

    auc_scores = {}
    for i, name in enumerate(class_names):
        if np.sum(all_labels_onehot[:, i]) == 0:
            auc_scores[name] = np.nan
        else:
            try:
                fpr, tpr, _ = roc_curve(all_labels_onehot[:, i], all_probs[:, i])
                auc_scores[name] = auc(fpr, tpr)
            except Exception:
                auc_scores[name] = np.nan

    metrics = {}
    for name in class_names:
        metrics[name] = {
            'Accuracy': report[name]['precision'],
            'Precision': report[name]['precision'],
            'Recall': report[name]['recall'],
            'F1': report[name]['f1-score'],
            'AUC': auc_scores[name]
        }

    # 宏平均
    macro = report['macro avg']
    metrics['Macro Avg'] = {
        'Accuracy': macro['precision'],
        'Precision': macro['precision'],
        'Recall': macro['recall'],
        'F1': macro['f1-score'],
        'AUC': np.nanmean(list(auc_scores.values()))
    }
    # 总体准确率
    overall_acc = np.mean(all_labels == all_preds)
    metrics['Overall'] = {
        'Accuracy': overall_acc,
        'Precision': np.nan, 'Recall': np.nan, 'F1': np.nan, 'AUC': np.nan
    }
    return cm, metrics


# ==================== 可视化函数 ====================
def plot_confusion_matrix(cm, class_names, save_path, present_indices=None, rectangular=True):
    """
    绘制混淆矩阵。
    - rectangular=True: 只显示真实存在的类别行，保留全部9类预测列（n×9长方形）
    - rectangular=False: 只显示真实存在的类别的正方形子集
    - present_indices=None: 显示完整9x9矩阵
    """
    if present_indices is not None and rectangular:
        present_indices = list(present_indices)
        # 行切片：只提取真实存在的类别行
        cm = cm[present_indices, :]
        yticklabels = [class_names[i] for i in present_indices]
        # 列重排：真实存在的类别排到前面，其余在后
        total_classes = len(class_names)
        all_indices = list(range(total_classes))
        new_col_order = present_indices + [i for i in all_indices if i not in present_indices]
        cm = cm[:, new_col_order]
        xticklabels = [class_names[i] for i in new_col_order]
    elif present_indices is not None and not rectangular:
        cm = cm[np.ix_(present_indices, present_indices)]
        yticklabels = [class_names[i] for i in present_indices]
        xticklabels = yticklabels
    else:
        yticklabels = class_names
        xticklabels = class_names

    rows, cols = cm.shape
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)

    # 构造标注：计数\n(百分比%)
    annot = np.empty_like(cm, dtype=object)
    for i in range(rows):
        for j in range(cols):
            count = cm[i, j]
            pct = cm_norm[i, j] * 100
            annot[i, j] = f"{count}\n({pct:.1f}%)"

    # 动态计算宽高比，保持格子为正方形
    base_cell_size = 1.2
    fig_width = max(8, cols * base_cell_size)
    fig_height = max(6, rows * base_cell_size)

    plt.figure(figsize=(fig_width, fig_height))
    sns.heatmap(cm_norm, annot=annot, fmt='', cmap='Blues',
                xticklabels=xticklabels, yticklabels=yticklabels,
                annot_kws={'size': 10}, square=True, cbar_kws={'shrink': 0.6})
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('Confusion Matrix (Counts & Percentages)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_per_class_bar(metrics, class_names, save_path, present_indices=None):
    """绘制每类Precision/Recall/F1柱状图"""
    if present_indices is not None:
        class_names = [class_names[i] for i in present_indices]

    prec_vals, rec_vals, f1_vals = [], [], []
    for name in class_names:
        if name not in metrics:
            prec_vals.append(0.0); rec_vals.append(0.0); f1_vals.append(0.0)
        else:
            p = metrics[name].get('Precision', 0.0)
            r = metrics[name].get('Recall', 0.0)
            f = metrics[name].get('F1', 0.0)
            prec_vals.append(0.0 if np.isnan(p) else p)
            rec_vals.append(0.0 if np.isnan(r) else r)
            f1_vals.append(0.0 if np.isnan(f) else f)

    fig_width = max(12, len(class_names) * 0.6)
    plt.figure(figsize=(fig_width, 6))
    x = np.arange(len(class_names))
    width = 0.25
    plt.bar(x - width, prec_vals, width, label='Precision', color='#1f77b4')
    plt.bar(x, rec_vals, width, label='Recall', color='#ff7f0e')
    plt.bar(x + width, f1_vals, width, label='F1-Score', color='#2ca02c')
    plt.xticks(x, class_names, rotation=45, ha='right', fontsize=10)
    plt.ylabel('Score', fontsize=12)
    plt.ylim(0, 1.05)
    plt.legend(loc='lower right')
    plt.title('Per-Class Performance', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_performance_table(metrics, class_names, output_path):
    """保存性能指标表格CSV"""
    metric_keys = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    metric_display = ['Acc', 'Prec', 'Rec', 'F1', 'AUC']
    data = {name: [metrics[name][k] for k in metric_keys] for name in class_names}
    df = pd.DataFrame(data, index=metric_display)
    df.to_csv(output_path, float_format='%.4f')


def save_visualization(img_tensor, label_idx, pred_idx, probs, class_names, case_id, save_dir):
    """保存单张图像的预测可视化（GT/Pred/Top-3标注）"""
    os.makedirs(os.path.join(save_dir, class_names[label_idx]), exist_ok=True)
    img_np = img_tensor.squeeze().cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)
    img_color = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

    # Top-3预测
    k = min(3, len(class_names))
    top_probs, top_indices = torch.topk(probs, k)
    top_probs = top_probs.cpu().numpy()
    top_indices = top_indices.cpu().numpy()

    pred_name = class_names[pred_idx]
    true_name = class_names[label_idx]
    status = "CORRECT" if pred_idx == label_idx else "WRONG"
    color = (0, 200, 0) if status == "CORRECT" else (0, 0, 255)

    # 标注文本
    cv2.putText(img_color, f"GT: {true_name}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 1)
    cv2.putText(img_color, f"ID: {case_id}", (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    cv2.putText(img_color, f"Pred: {pred_name} ({probs[pred_idx]:.1%})", (8, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    y_base = img_color.shape[0] - 10
    for i, (idx, prob) in enumerate(zip(top_indices, top_probs)):
        text = f"{i+1}. {class_names[idx]}: {prob:.1%}"
        cv2.putText(img_color, text, (8, y_base - (2 - i) * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 0), 1)

    filename = f"{status}_{case_id}_{pred_name}.jpg"
    cv2.imwrite(os.path.join(save_dir, true_name, filename), img_color)


# ==================== CAM可解释性 ====================
def build_cam(method, model, target_layers, reshape_transform):
    """根据方法名称构建CAM对象（gradcam/gradcam++/layercam）"""
    method = method.lower()
    cam_classes = {
        "gradcam": GradCAM,
        "gradcam++": GradCAMPlusPlus,
        "layercam": LayerCAM,
    }
    if method not in cam_classes:
        raise ValueError(f"不支持的CAM方法：{method}，可选：{list(cam_classes.keys())}")
    return cam_classes[method](
        model=model, target_layers=target_layers, reshape_transform=reshape_transform
    )


def create_swin_reshape_transform(feature_h: int, feature_w: int):
    """为Swin Transformer创建reshape_transform，支持矩形特征图"""
    def reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 3:
            b, n, c = tensor.shape
            expected_tokens = feature_h * feature_w
            if n == expected_tokens + 1:
                tensor = tensor[:, 1:, :]
                n -= 1
            if n != expected_tokens:
                raise ValueError(f"Token数量不匹配：期望{expected_tokens}，实际{n}")
            return tensor.reshape(b, feature_h, feature_w, c).permute(0, 3, 1, 2).contiguous()
        if tensor.ndim == 4:
            if tensor.shape[1] == feature_h and tensor.shape[2] == feature_w:
                return tensor.permute(0, 3, 1, 2).contiguous()
            if tensor.shape[2] == feature_h and tensor.shape[3] == feature_w:
                return tensor.contiguous()
            raise ValueError(f"无法识别四维特征格式：{tuple(tensor.shape)}")
        raise ValueError(f"不支持的输出格式：{tensor.ndim}维")
    return reshape_transform


def denormalize_image(img_tensor):
    """img_tensor: [C, H, W] -> RGB float [0,1] [H,W,C]"""
    img = img_tensor.detach().cpu().float().clamp(0, 1)
    if img.ndim == 3:
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        img = img.permute(1, 2, 0).numpy()
    elif img.ndim == 2:
        img = np.stack([img.numpy()] * 3, axis=-1)
    return img


def overlay_cam_on_image(rgb_img, grayscale_cam, alpha=0.45):
    """将CAM热力图叠加到原图上"""
    rgb_uint8 = np.uint8(255 * rgb_img)
    heatmap = np.uint8(255 * grayscale_cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    img_bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
    if heatmap.shape[:2] != img_bgr.shape[:2]:
        heatmap = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    overlay = cv2.addWeighted(img_bgr, 1 - alpha, heatmap, alpha, 0)
    return overlay, heatmap, img_bgr


def generate_gradcam(
    model, dataset, class_names, save_dir, device,
    target_layer, feature_size,
    max_correct_per_class=10, max_wrong_per_class=10,
    cam_method="layercam",
):
    """
    生成CAM可解释性热力图。
    正确/错误样本分开保存，每类最多保存指定数量。
    输出：overlay叠加图、heatmap热力图、original原图、metadata CSV。
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    correct_dir = os.path.join(save_dir, "Correct")
    wrong_dir = os.path.join(save_dir, "Wrong")
    os.makedirs(correct_dir, exist_ok=True)
    os.makedirs(wrong_dir, exist_ok=True)

    n_classes = len(class_names)
    correct_counts = {i: 0 for i in range(n_classes)}
    wrong_counts = {i: 0 for i in range(n_classes)}
    log_records = []

    feature_h, feature_w = feature_size
    reshape_transform = create_swin_reshape_transform(feature_h, feature_w)
    cam = build_cam(method=cam_method, model=model, target_layers=[target_layer],
                    reshape_transform=reshape_transform)

    try:
        for idx, sample in enumerate(dataset):
            img_tensor, label = sample[0], sample[1]
            label = int(label.detach().cpu().item()) if isinstance(label, torch.Tensor) else int(label)
            if not 0 <= label < n_classes:
                continue

            # 检查该类别是否已达上限
            if (correct_counts[label] >= max_correct_per_class and
                    wrong_counts[label] >= max_wrong_per_class):
                continue
            all_done = all(
                correct_counts[i] >= max_correct_per_class and wrong_counts[i] >= max_wrong_per_class
                for i in range(n_classes)
            )
            if all_done:
                break

            input_tensor = img_tensor.unsqueeze(0).to(device)
            with torch.enable_grad():
                output = model(input_tensor)
                if isinstance(output, (tuple, list)):
                    output = output[0]
                probs = torch.softmax(output, dim=1)
                pred = int(output.argmax(dim=1).item())
                confidence = float(probs[0, pred].item())

            is_correct = (pred == label)
            if is_correct:
                if correct_counts[label] >= max_correct_per_class:
                    continue
                current_save_dir = correct_dir
                correct_counts[label] += 1
            else:
                if wrong_counts[label] >= max_wrong_per_class:
                    continue
                current_save_dir = wrong_dir
                wrong_counts[label] += 1

            # 生成热力图（针对预测类别）
            targets = [ClassifierOutputTarget(pred)]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
            grayscale_cam = np.nan_to_num(grayscale_cam, nan=0.0, posinf=1.0, neginf=0.0)
            grayscale_cam = np.clip(grayscale_cam, 0.0, 1.0)

            rgb_img = denormalize_image(img_tensor)
            overlay, heatmap, original_bgr = overlay_cam_on_image(rgb_img, grayscale_cam, alpha=0.45)

            class_name = class_names[label]
            safe_name = str(class_name).replace("/", "_").replace("\\", "_")
            status_str = "CORRECT" if is_correct else "WRONG"
            base_name = f"{status_str}_{safe_name}_true{label}_pred{pred}_conf{confidence:.3f}_{cam_method}"

            cv2.imwrite(os.path.join(current_save_dir, base_name + "_overlay.png"), overlay)
            cv2.imwrite(os.path.join(current_save_dir, base_name + "_heatmap.png"), heatmap)
            cv2.imwrite(os.path.join(current_save_dir, base_name + "_original.png"), original_bgr)

            log_records.append({
                'index': idx, 'class_name': class_name,
                'true_label': label, 'pred_label': pred,
                'pred_name': class_names[pred],
                'is_correct': is_correct, 'confidence': confidence,
                'filename': base_name
            })

            if (idx + 1) % 50 == 0:
                print(f"已处理 {idx+1}/{len(dataset)} 个样本...")
    finally:
        cam.activations_and_grads.release()

    # 保存元数据
    if log_records:
        pd.DataFrame(log_records).to_csv(os.path.join(save_dir, "gradcam_metadata.csv"), index=False)

    total_correct = sum(correct_counts.values())
    total_wrong = sum(wrong_counts.values())
    print(f"\n{cam_method} 生成完成！正确: {total_correct}张, 错误: {total_wrong}张")
    for i, name in enumerate(class_names):
        print(f"  {name}: 正确 {correct_counts[i]}张, 错误 {wrong_counts[i]}张")


# ==================== 主程序 ====================
def parse_args():
    parser = argparse.ArgumentParser(description='OCT九分类外部测试')
    parser.add_argument('--xlsx', type=str, required=True,
                        help='外部测试Excel路径（需包含process_img_path, label_dict, datatag, dataset列）')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='模型checkpoint路径')
    parser.add_argument('--output', type=str, default='./external_test_results',
                        help='结果输出目录')
    parser.add_argument('--batch_size', type=int, default=64, help='批大小')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader进程数')
    parser.add_argument('--no_vis', action='store_true', help='关闭逐样本可视化')
    parser.add_argument('--no_cam', action='store_true', help='关闭CAM热力图生成')
    parser.add_argument('--cam_method', type=str, default='layercam',
                        choices=['gradcam', 'gradcam++', 'layercam'],
                        help='CAM方法')
    parser.add_argument('--cam_per_class', type=int, default=10,
                        help='每类最多生成的热力图数量（正确+错误各N张）')
    return parser.parse_args()


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.output, exist_ok=True)

    # 1. 读取并清洗数据
    df_all = pd.read_excel(args.xlsx)
    df_test = df_all[df_all['datatag'] == 'test'].copy()
    if 'black' in df_test.columns:
        before = len(df_test)
        df_test = df_test[df_test['black'] == 0]
        print(f"black列过滤：移除了 {before - len(df_test)} 条数据")
    before = len(df_test)
    df_test = df_test[df_test['label_dict'].isin(CLASS_NAMES)]
    print(f"无效标签过滤：移除了 {before - len(df_test)} 条数据")
    if df_test.empty:
        raise ValueError("清洗后无有效测试数据")

    # 检查文件存在性
    valid_mask = df_test['process_img_path'].apply(os.path.exists)
    if not valid_mask.all():
        invalid_count = (~valid_mask).sum()
        print(f"警告: 发现 {invalid_count} 个文件不存在，已剔除")
        df_test = df_test[valid_mask].reset_index(drop=True)

    # 2. 加载模型（完整9分类）
    print(f"加载模型: {args.ckpt}")
    model = OCTWrapper.load_from_checkpoint(
        args.ckpt, num_classes=NUM_CLASSES, class_names=CLASS_NAMES, weights_only=False
    ).to(device)
    model.eval()

    # 3. 按数据集分组测试
    dataset_summary = []
    if 'dataset' in df_test.columns:
        dataset_names = df_test['dataset'].unique()
    else:
        dataset_names = ['external']
        df_test['dataset'] = 'external'

    for ds_name in dataset_names:
        print(f"\n===== 处理数据集: {ds_name} =====")
        df_sub = df_test[df_test['dataset'] == ds_name].reset_index(drop=True)
        if len(df_sub) == 0:
            continue

        print(f"标签分布:\n{df_sub['label_dict'].value_counts()}")

        dataset = ExternalTestDataset(df_sub, NAME_TO_IDX, img_size=(256, 448))
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)

        # 推理
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for images, labels, paths, case_ids in dataloader:
                images = images.to(device)
                logits = model(images)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
                all_probs.extend(probs)

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # 计算指标（9x9矩阵）
        cm, metrics = compute_metrics(all_labels, all_preds, all_probs, CLASS_NAMES)
        present_true = sorted(list(set(all_labels)))

        # 保存结果
        ds_output_dir = os.path.join(args.output, ds_name)
        os.makedirs(ds_output_dir, exist_ok=True)
        plot_confusion_matrix(cm, CLASS_NAMES, os.path.join(ds_output_dir, 'ConfusionMatrix.png'),
                              present_indices=present_true, rectangular=True)
        plot_per_class_bar(metrics, CLASS_NAMES, os.path.join(ds_output_dir, 'PerClassBar.png'),
                           present_indices=present_true)
        save_performance_table(metrics, CLASS_NAMES, os.path.join(ds_output_dir, 'Performance.csv'))

        macro = metrics['Macro Avg']
        dataset_summary.append({
            'Dataset': ds_name, 'Samples': len(dataset),
            'Accuracy': macro['Accuracy'], 'Precision': macro['Precision'],
            'Recall': macro['Recall'], 'F1': macro['F1'], 'AUC': macro['AUC']
        })

    # 4. 总体结果（所有数据集合并）
    print("\n===== 生成总体结果 =====")
    overall_dir = os.path.join(args.output, 'Overall')
    os.makedirs(overall_dir, exist_ok=True)

    dataset_all = ExternalTestDataset(df_test, NAME_TO_IDX, img_size=(256, 448))
    dataloader_all = DataLoader(dataset_all, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True)

    vis_root = os.path.join(overall_dir, 'vis_images')
    if not args.no_vis:
        for name in CLASS_NAMES:
            os.makedirs(os.path.join(vis_root, name), exist_ok=True)

    all_records = []
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels, paths, case_ids in dataloader_all:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

            for i in range(len(images)):
                true_label = labels[i].item()
                pred_label = preds[i].item()
                prob = probs[i, pred_label].item()
                case_id = case_ids[i]

                if not args.no_vis:
                    save_visualization(images[i], true_label, pred_label, probs[i],
                                       CLASS_NAMES, case_id, vis_root)

                all_records.append({
                    'case_id': case_id, 'img_path': paths[i],
                    'true_label': CLASS_NAMES[true_label],
                    'pred_label': CLASS_NAMES[pred_label],
                    'confidence': prob,
                    'status': "CORRECT" if pred_label == true_label else "WRONG"
                })

    # 保存详细预测表
    pd.DataFrame(all_records).to_csv(os.path.join(overall_dir, 'detailed_predictions.csv'), index=False)

    # 总体指标
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    cm_all, metrics_all = compute_metrics(all_labels, all_preds, all_probs, CLASS_NAMES)
    present_all = sorted(list(set(all_labels)))

    plot_confusion_matrix(cm_all, CLASS_NAMES, os.path.join(overall_dir, 'ConfusionMatrix.png'),
                          present_indices=present_all, rectangular=True)
    plot_per_class_bar(metrics_all, CLASS_NAMES, os.path.join(overall_dir, 'PerClassBar.png'),
                       present_indices=present_all)
    save_performance_table(metrics_all, CLASS_NAMES, os.path.join(overall_dir, 'Performance.csv'))

    # 5. CAM热力图
    if not args.no_cam:
        print("\n===== 生成CAM热力图 =====")
        gradcam_dir = os.path.join(overall_dir, 'gradcam')
        backbone = model.model.model
        target_layer = backbone.layers[-1].blocks[-1].norm1
        feature_size = (8, 14)  # Swin-T最后一层特征图尺寸

        generate_gradcam(
            model=model, dataset=dataset_all, class_names=CLASS_NAMES,
            save_dir=gradcam_dir, device=device,
            target_layer=target_layer, feature_size=feature_size,
            max_correct_per_class=args.cam_per_class,
            max_wrong_per_class=args.cam_per_class,
            cam_method=args.cam_method,
        )

    # 6. 保存数据集汇总表
    if dataset_summary:
        pd.DataFrame(dataset_summary).to_csv(
            os.path.join(args.output, 'Dataset_Summary.csv'), index=False, float_format='%.4f')

    print(f"\n全部完成！结果保存在: {args.output}")


if __name__ == '__main__':
    main()
