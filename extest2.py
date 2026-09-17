import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from wrapper import OCTWrapper
import torch.nn as nn
from monai.transforms import Compose, ScaleIntensityd, ToTensord
from SimpleITK import ReadImage, GetArrayFromImage
import cv2
from collections import defaultdict
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

dataset_summary_list = []
# ==================== 配置区域 ====================
EXTERNAL_XLSX = '/data/chenxinfen/OCT1/new9/视网膜疾病分类_external_testV5.2.xlsx'
CKPT_PATH = '/data/chenxinfen/ModelTraining_monai/Training_code_V7/Model/last-v2.ckpt'
OUTPUT_BASE_DIR = './external_testing_plots-2'
BATCH_SIZE = 64
NUM_WORKERS = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ENABLE_VISUAL_CHECK = True          # 是否生成逐样本可视化
MAX_VISUAL_PER_CLASS = 20           # 每个类别最多保存多少张可视化图像（避免过多）
# 是否生成 Grad-CAM 热力图（需要 torchcam）
ENABLE_GRAD_CAM =  True
# 每个类别最多生成几张热力图
MAX_GRADCAM_PER_CLASS = 20
# =================================================
# 模型类型：6 表示六分类+OTHER
MODEL_TYPE = 9
# =================================================

# 根据模型类型定义类别名称和转换规则
if MODEL_TYPE == 6:
    CLASS_NAMES = ['NORMAL', 'CNV', 'DME', 'DRUSEN', 'MH', 'DR', 'OTHERS']
    NUM_CLASSES = len(CLASS_NAMES)
    OTHERS_LIST = ['AMD', 'CSC', 'ERM', 'RVO', 'VID', 'RRD', 'RAO']
elif MODEL_TYPE == 9:
    CLASS_NAMES = ['NORMAL', 'AMD', 'DR', 'ERM', 'MH', 'RRD', 'RVO', 'RAO', 'CSC']
    NUM_CLASSES = len(CLASS_NAMES)
elif MODEL_TYPE == 13:
    CLASS_NAMES = ['NORMAL', 'CNV', 'DME', 'DRUSEN', 'MH', 'DR', 'AMD',
                   'CSC', 'ERM', 'RVO', 'VID', 'RRD', 'RAO']
    NUM_CLASSES = len(CLASS_NAMES)
    OTHERS_LIST = []
elif MODEL_TYPE == 11:
    CLASS_NAMES = ['NORMAL', 'CNV', 'DME', 'DRUSEN', 'MH', 'DR', 
                   'CSC', 'ERM', 'RVO', 'RRD', 'RAO']
    NUM_CLASSES = len(CLASS_NAMES)
elif MODEL_TYPE == 12:
    CLASS_NAMES = ['NORMAL', 'DRUSEN', 'GA', 'CNV', 'DME', 'DR', 
                'ERM', 'MH', 'RRD', 'RVO', 'RAO', 'CSC']
    NUM_CLASSES = len(CLASS_NAMES)
else:
    raise ValueError("MODEL_TYPE must be 6, 9, 11, or 12")

# 原始13类名称（必须与模型训练时一致）
FULL_CLASS_NAMES = CLASS_NAMES

# ---------- 从 ds.py 复制过来的预处理函数 ----------
def detect_orientation(image: np.ndarray) -> np.ndarray:
        h, w = image.shape
        if w / h > 2.0:
            return image

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
        
        if col_profile_var > row_profile_var * 1.2:
            image = np.rot90(image, k=-1)

        return image

def adaptive_resize(image, target_h=256, target_w=448, fill_value=0):
        h, w = image.shape
        target_aspect = target_w / target_h   # 448/256 = 1.75
        img_aspect = w / h
        
        if img_aspect > target_aspect:
            new_w = int(target_aspect * h)
            left = (w - new_w) // 2
            right = w - new_w - left
            cropped = image[:, left:w-right]
            resized = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        else:
            new_w = int(h * target_aspect)
            pad_w = new_w - w
            left = pad_w // 2
            right = pad_w - left
            padded = cv2.copyMakeBorder(image, 0, 0, left, right,
                                        cv2.BORDER_CONSTANT, value=fill_value)
            resized = cv2.resize(padded, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        
        return resized

def crop_retinal_body(image, min_area_ratio=0.05, dilation_iter=3, margin=10):
        """
        基于轮廓检测，自动裁剪出视网膜主体区域（去除周围黑色背景）。
        
        Args:
            image: 输入图像 (H, W) 灰度图或彩色图
            min_area_ratio: 最小面积阈值（相对整图面积），低于此则返回原图
            dilation_iter: 形态学膨胀迭代次数，用于连接主体断裂区域
            margin: 外扩像素，避免切得太紧
        
        Returns:
            cropped: 裁剪后的图像
        """
        # 转为灰度图（如果是彩色）
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        h, w = gray.shape
        total_area = h * w
        
        # 大津二值化
        gray_8u = cv2.convertScaleAbs(gray)
        _, binary = cv2.threshold(gray_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 形态学闭运算（填孔）+ 膨胀（连接边缘）
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.dilate(binary, kernel, iterations=dilation_iter)
        
        # 寻找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image
        
        # 取最大面积轮廓
        max_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(max_contour)
        if area < min_area_ratio * total_area:
            return image
        
        x, y, w_box, h_box = cv2.boundingRect(max_contour)
        # 外扩 margin 像素
        x = max(0, x - margin)
        y = max(0, y - margin)
        w_box = min(image.shape[1] - x, w_box + 2 * margin)
        h_box = min(image.shape[0] - y, h_box + 2 * margin)
        
        cropped = image[y:y+h_box, x:x+w_box]
        return cropped

# ---------- 自定义 Dataset ----------
class ExternalTestDataset(Dataset):
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
        
        img = ReadImage(img_path)
        img_array = GetArrayFromImage(img)
        if len(img_array.shape) == 3:
            img_array = img_array[0]
        
        # 1. 方向校正（已存在）
        img_array = detect_orientation(img_array)
        
        # 2. 主体裁剪（新增，与训练时 crop_body=True 一致）
        # 注意：需要确保 crop_retinal_body 函数已定义（您已经在 1.py 中定义了它）
        # img_array = crop_retinal_body(img_array, min_area_ratio=0.05, dilation_iter=3, margin=10)
        
        # 3. 自适应缩放（已存在）
        img_array = adaptive_resize(img_array, self.img_size[0], self.img_size[1])
        
        img_array = np.expand_dims(img_array, axis=0)
        data_dict = {'image': img_array}
        data_dict = self.transform(data_dict)
        img_tensor = data_dict['image']
        
        return img_tensor, torch.tensor(label, dtype=torch.long), img_path, case_id

# ---------- 辅助函数 ----------
def convert_labels_to_mode(df, model_type, others_list):
    """根据模型类型将 label_dict 列转换为相应的标签"""
    df = df.copy()
    if model_type == 6:
        # 将 OTHERS_LIST 中的疾病统一替换为 'OTHERS'
        df.loc[df['label_dict'].isin(others_list), 'label_dict'] = 'OTHERS'
    # model_type == 13 不需要转换
    return df

def get_subset_classes_from_dataframe(df_test):
    present_names = df_test['label_dict'].unique()
    # 按原始 FULL_CLASS_NAMES 顺序排序
    present_names_sorted = sorted(present_names, key=lambda x: FULL_CLASS_NAMES.index(x))
    present_indices = [FULL_CLASS_NAMES.index(name) for name in present_names_sorted]
    return present_indices, present_names_sorted

def adapt_model_for_subset(model, subset_indices, subset_names):
    original_head = model.model.model.head.fc
    in_features = original_head.in_features
    num_subset = len(subset_indices)
    new_head = nn.Linear(in_features, num_subset)
    new_head.weight.data = original_head.weight.data[subset_indices, :]
    new_head.bias.data = original_head.bias.data[subset_indices]
    model.model.model.head.fc = new_head.to(DEVICE)
    model.num_classes = num_subset
    model.class_names = subset_names
    model.name_to_idx = {name: i for i, name in enumerate(subset_names)}
    model.test_step_outputs = []
    return model

def build_cam(method, model, target_layers, reshape_transform):
    """
    根据方法名称构建 CAM 对象。
    """
    method = method.lower()
    cam_classes = {
        "gradcam": GradCAM,
        "gradcam++": GradCAMPlusPlus,
        "layercam": LayerCAM,
    }
    if method not in cam_classes:
        raise ValueError(
            f"不支持的 CAM 方法：{method}，可选：{list(cam_classes.keys())}"
        )
    return cam_classes[method](
        model=model,
        target_layers=target_layers,
        reshape_transform=reshape_transform,
    )

def compute_metrics(all_labels, all_preds, all_probs, class_names):
    """计算每类的 Accuracy, Precision, Recall, F1, AUC，处理缺失类别"""
    n_classes = len(class_names)
    # 确保 all_labels 和 all_preds 是整数类型
    all_labels = all_labels.astype(int)
    all_preds = all_preds.astype(int)
    
    # 混淆矩阵，指定 labels 为所有类别
    cm = confusion_matrix(all_labels, all_preds, labels=range(n_classes))
    
    # 分类报告，指定 labels 和 target_names，zero_division=0
    report = classification_report(
        all_labels, all_preds, 
        labels=range(n_classes), 
        target_names=class_names,
        output_dict=True, 
        zero_division=0
    )
    
    # 计算每类 AUC（需要 one-hot 编码）
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
            except:
                auc_scores[name] = np.nan
    
    metrics = {}
    for name in class_names:
        metrics[name] = {
            'Accuracy': report[name]['precision'],   # 对于多分类，precision 可作为每类准确率的近似
            'Precision': report[name]['precision'],
            'Recall': report[name]['recall'],
            'F1': report[name]['f1-score'],
            'AUC': auc_scores[name]
        }
    
    # 总体宏平均
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
        'Precision': np.nan,
        'Recall': np.nan,
        'F1': np.nan,
        'AUC': np.nan
    }
    return cm, metrics

def plot_confusion_matrix(cm, class_names, save_path, present_indices=None, rectangular=False):
    """
    绘制混淆矩阵。
    - rectangular=False: 绘制正方形矩阵 (默认)
    - rectangular=True: 只显示 present_indices 对应的行（真实标签），保留所有列（预测标签），实现 n*9 长方形。
    - 当 rectangular=True 时，自动将 X 轴（Predicted）重新排序，把 present_indices 对应的预测列排在最前面。
    """
    if present_indices is not None and rectangular:
        # 将 present_indices 转为 list 以确保排序和索引操作
        present_indices = list(present_indices)
        
        # 1. 行切片：只提取 present_indices 对应的真实行
        cm = cm[present_indices, :]
        yticklabels = [class_names[i] for i in present_indices]
        
        # 2. 列重排：【核心改进】将测试集真实存在的类别（present_indices）排到 X 轴最前面
        total_classes = len(class_names)
        all_indices = list(range(total_classes))
        # 构造新顺序： present_indices (如 0,1,2) + 剩下的所有类别 (如 3,4,5,6,7,8)
        new_col_order = present_indices + [i for i in all_indices if i not in present_indices]
        
        # 根据新顺序重排列
        cm = cm[:, new_col_order]
        xticklabels = [class_names[i] for i in new_col_order]
        
    elif present_indices is not None and not rectangular:
        # 原来的逻辑：提取正方形子集
        cm = cm[np.ix_(present_indices, present_indices)]
        yticklabels = [class_names[i] for i in present_indices]
        xticklabels = yticklabels
    else:
        # 全部绘制的情况
        yticklabels = class_names
        xticklabels = class_names

    rows, cols = cm.shape
    
    # 计算百分比归一化矩阵
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
    
    # 构造标注：计数\n(百分比%)
    annot = np.empty_like(cm, dtype=object)
    for i in range(rows):
        for j in range(cols):
            count = cm[i, j]
            pct = cm_norm[i, j] * 100
            annot[i, j] = f"{count}\n({pct:.1f}%)"
            
    # 核心视觉优化：动态计算宽高比，保持格子为正方形
    base_cell_size = 1.2
    fig_width = max(8, cols * base_cell_size)
    fig_height = max(6, rows * base_cell_size)
    
    # 开启 square=True 强制让画出的格子是正方形
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
    if present_indices is not None:
        # 子集模式：只显示实际存在的类别
        class_names = [class_names[i] for i in present_indices]
    # 构建指标列表，缺失则填0
    prec_vals = []
    rec_vals = []
    f1_vals = []
    for name in class_names:
        # 从 metrics 中获取，如果 metrics 中没有该类别（理论上不应该）则填0
        if name not in metrics:
            prec_vals.append(0.0)
            rec_vals.append(0.0)
            f1_vals.append(0.0)
        else:
            prec_vals.append(metrics[name].get('Precision', 0.0) if not np.isnan(metrics[name].get('Precision', 0.0)) else 0.0)
            rec_vals.append(metrics[name].get('Recall', 0.0) if not np.isnan(metrics[name].get('Recall', 0.0)) else 0.0)
            f1_vals.append(metrics[name].get('F1', 0.0) if not np.isnan(metrics[name].get('F1', 0.0)) else 0.0)
    # 动态图片宽度：类别数 * 0.6英寸，最小12英寸
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

def add_dataset_summary(dataset_name, n_samples, macro_metrics):
    dataset_summary_list.append({
        'Dataset': dataset_name,
        'Samples': n_samples,
        'Accuracy (Macro)': macro_metrics['Accuracy'],
        'Precision (Macro)': macro_metrics['Precision'],
        'Recall (Macro)': macro_metrics['Recall'],
        'F1 (Macro)': macro_metrics['F1'],
        'AUC (Macro)': macro_metrics['AUC']
    })

def save_dataset_summary(output_path):
    pd.DataFrame(dataset_summary_list).to_csv(output_path, index=False, float_format='%.4f')

def save_performance_table(metrics, class_names, output_path, present_indices=None):
    metric_keys = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    metric_display = ['Acc', 'Prec', 'Recall', 'F1', 'AUC']
    
    if present_indices is not None:
        class_names = [class_names[i] for i in present_indices]
    data = {name: [metrics[name][k] for k in metric_keys] for name in class_names}
    df = pd.DataFrame(data, index=metric_display)
    # 不添加宏平均和总体准确率，保持简洁
    df.to_csv(output_path, float_format='%.4f')

def denormalize_image(img_tensor, mean=None, std=None):
    """img_tensor: [C, H, W], return RGB float [0,1] [H,W,C]"""
    img = img_tensor.detach().cpu().float()
    if mean is not None and std is not None:
        mean = torch.tensor(mean).view(-1, 1, 1)
        std = torch.tensor(std).view(-1, 1, 1)
        img = img * std + mean
    img = img.clamp(0, 1)
    if img.ndim == 3:
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        img = img.permute(1, 2, 0).numpy()
    elif img.ndim == 2:
        img = img.numpy()
        img = np.stack([img, img, img], axis=-1)
    else:
        raise ValueError(f"Unexpected image shape: {img.shape}")
    return img

def create_swin_reshape_transform(feature_h: int, feature_w: int):
    """
    为 Swin Transformer 创建 reshape_transform，支持矩形特征图。

    参数:
        feature_h, feature_w: 特征图的空间尺寸（如 16, 28）
    """
    def reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 3:
            b, n, c = tensor.shape
            expected_tokens = feature_h * feature_w

            # 兼容可能存在 cls token 的情况
            if n == expected_tokens + 1:
                tensor = tensor[:, 1:, :]
                n -= 1

            if n != expected_tokens:
                raise ValueError(
                    f"Token 数量不匹配：shape={tuple(tensor.shape)}, "
                    f"期望 token 数={expected_tokens} ({feature_h}×{feature_w})"
                )

            return (
                tensor.reshape(b, feature_h, feature_w, c)
                .permute(0, 3, 1, 2)
                .contiguous()
            )

        if tensor.ndim == 4:
            # 若为 [B, H, W, C] 转成 [B, C, H, W]
            if tensor.shape[1] == feature_h and tensor.shape[2] == feature_w:
                return tensor.permute(0, 3, 1, 2).contiguous()
            # 若已是 [B, C, H, W] 直接返回
            if tensor.shape[2] == feature_h and tensor.shape[3] == feature_w:
                return tensor.contiguous()

            raise ValueError(
                f"无法识别四维特征格式：tensor.shape={tuple(tensor.shape)}, "
                f"期望空间尺寸={feature_h}×{feature_w}"
            )

        raise ValueError(
            f"不支持的输出格式：tensor.ndim={tensor.ndim}, "
            f"tensor.shape={tuple(tensor.shape)}"
        )

    return reshape_transform

def overlay_cam_on_image(rgb_img, grayscale_cam, alpha=0.45):
    rgb_uint8 = np.uint8(255 * rgb_img)
    heatmap = np.uint8(255 * grayscale_cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    img_bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
    if heatmap.shape[:2] != img_bgr.shape[:2]:
        heatmap = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    overlay = cv2.addWeighted(img_bgr, 1 - alpha, heatmap, alpha, 0)
    return overlay, heatmap, img_bgr

def generate_gradcam(
    model,
    dataset,
    class_names,
    save_dir,
    device,
    target_layer,
    feature_size,
    max_correct_per_class=10,      # 每个类别最多生成几张正确热力图
    max_wrong_per_class=10,        # 每个类别最多生成几张错误热力图（若存在错误）
    mean=None,
    std=None,
    use_true_label=False,
    cam_method="layercam",
):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    # 创建正确和错误的子文件夹
    correct_dir = os.path.join(save_dir, "Correct")
    wrong_dir = os.path.join(save_dir, "Wrong")
    os.makedirs(correct_dir, exist_ok=True)
    os.makedirs(wrong_dir, exist_ok=True)

    # 记录日志
    log_records = []

    # 初始化双计数器：正确和错误分别计数
    n_classes = len(class_names)
    correct_counts = {i: 0 for i in range(n_classes)}
    wrong_counts = {i: 0 for i in range(n_classes)}

    feature_h, feature_w = feature_size
    reshape_transform = create_swin_reshape_transform(feature_h, feature_w)
    cam = build_cam(
        method=cam_method,
        model=model,
        target_layers=[target_layer],
        reshape_transform=reshape_transform,
    )

    try:
        for idx, sample in enumerate(dataset):
            if len(sample) < 2:
                continue
            img_tensor, label = sample[0], sample[1]
            if isinstance(label, torch.Tensor):
                label = int(label.detach().cpu().item())
            else:
                label = int(label)

            if not 0 <= label < n_classes:
                continue

            # 检查该类别是否已达到正确和错误的上限
            if correct_counts[label] >= max_correct_per_class and wrong_counts[label] >= max_wrong_per_class:
                continue  # 该类别已完成，跳过

            # 检查所有类别是否都已满足
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
                if isinstance(output, dict):
                    output = output["logits"]
                probs = torch.softmax(output, dim=1)
                pred = int(output.argmax(dim=1).item())
                confidence = float(probs[0, pred].item())

            is_correct = (pred == label)

            # 决定是否保存此样本
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

            # 生成热力图
            target_category = label if use_true_label else pred
            targets = [ClassifierOutputTarget(target_category)]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
            grayscale_cam = np.nan_to_num(grayscale_cam, nan=0.0, posinf=1.0, neginf=0.0)
            grayscale_cam = np.clip(grayscale_cam, 0.0, 1.0)

            rgb_img = denormalize_image(img_tensor, mean=mean, std=std)
            overlay, heatmap, original_bgr = overlay_cam_on_image(rgb_img, grayscale_cam, alpha=0.45)

            class_name = class_names[label]
            safe_name = str(class_name).replace("/", "_").replace("\\", "_")
            status_str = "CORRECT" if is_correct else "WRONG"
            base_name = (
                f"{status_str}_{safe_name}_true{label}_pred{pred}"
                f"_conf{confidence:.3f}_{cam_method}"
            )
            cv2.imwrite(os.path.join(current_save_dir, base_name + "_overlay.png"), overlay)
            cv2.imwrite(os.path.join(current_save_dir, base_name + "_heatmap.png"), heatmap)
            cv2.imwrite(os.path.join(current_save_dir, base_name + "_original.png"), original_bgr)

            log_records.append({
                'index': idx,
                'class_name': class_name,
                'true_label': label,
                'pred_label': pred,
                'pred_name': class_names[pred],
                'is_correct': is_correct,
                'confidence': confidence,
                'target_category': target_category,
                'filename': base_name
            })

            if (idx + 1) % 50 == 0:
                print(f"已处理 {idx+1}/{len(dataset)} 个样本...")

    finally:
        cam.activations_and_grads.release()

    # 保存日志CSV
    if log_records:
        import pandas as pd
        df_log = pd.DataFrame(log_records)
        csv_path = os.path.join(save_dir, "gradcam_metadata.csv")
        df_log.to_csv(csv_path, index=False)
        print(f"详细元数据已保存至：{csv_path}")

    # 统计并打印
    total_correct = sum(correct_counts.values())
    total_wrong = sum(wrong_counts.values())
    print(f"\n{cam_method} 生成完成！")
    print(f"  正确样本总数: {total_correct} 张 (保存于 {correct_dir})")
    print(f"  错误样本总数: {total_wrong} 张 (保存于 {wrong_dir})")
    for i, name in enumerate(class_names):
        print(f"  {name}: 正确 {correct_counts[i]} 张, 错误 {wrong_counts[i]} 张")
    print(f"  元数据日志: {csv_path}")

def save_visualization(img_tensor, label_idx, pred_idx, probs, class_names, case_id, save_dir):
    """保存单张图像的预测可视化"""
    os.makedirs(save_dir, exist_ok=True)
    img_np = img_tensor.squeeze().cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)
    img_color = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    
    # 获取 top-3
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
    # 行间距缩小为18，字号0.35
    for i, (idx, prob) in enumerate(zip(top_indices, top_probs)):
        text = f"{i+1}. {class_names[idx]}: {prob:.1%}"
        cv2.putText(img_color, text, (8, y_base - (2 - i) * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 0), 1)

    filename = f"{status}_{case_id}_{pred_name}.jpg"
    cv2.imwrite(os.path.join(save_dir, true_name, filename), img_color)

# ---------- 主程序 ----------
if __name__ == '__main__':
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    df_all = pd.read_excel(EXTERNAL_XLSX)
    df_test_all = df_all[df_all['datatag'] == 'test'].copy()
    # 过滤 black 列
    if 'black' in df_test_all.columns:
        before = len(df_test_all)
        df_test_all = df_test_all[df_test_all['black'] == 0]
        print(f"black 列过滤：移除了 {before - len(df_test_all)} 条 black=1 的数据")
    
    before = len(df_test_all)
    df_test_all = df_test_all[df_test_all['label_dict'].isin(FULL_CLASS_NAMES)]
    print(f"无效标签过滤：移除了 {before - len(df_test_all)} 条标签不在 {FULL_CLASS_NAMES} 中的数据")

    if df_test_all.empty:
        raise ValueError("清洗后无有效测试数据")

    # 1. 获取子集类别（按原始顺序，仅用于信息打印）
    subset_indices, subset_names = get_subset_classes_from_dataframe(df_test_all)
    print(f"外部测试集实际包含类别（按原始顺序）: {subset_names} (共 {len(subset_names)} 类)")

    # 2. 加载模型
    model = OCTWrapper.load_from_checkpoint(
        CKPT_PATH,
        num_classes=len(FULL_CLASS_NAMES),
        class_names=FULL_CLASS_NAMES,
        weights_only=False
    ).to(DEVICE)

    # 【关键修改】：删除子集适配逻辑，强制保留完整的 9 分类模型输出
    print("保持完整 9 分类模型结构，以观察预测到非本测试集类别（如类别4、5）的情况")
    
    # 保持完整映射
    name_to_idx = {name: i for i, name in enumerate(FULL_CLASS_NAMES)}
    # 更新模型内部名称映射（如有需要）
    model.class_names = FULL_CLASS_NAMES
    model.name_to_idx = name_to_idx

    dataset_names = df_test_all['dataset'].unique()
    for ds_name in dataset_names:
        print(f"\n===== 处理数据集: {ds_name} =====")
        df_sub = df_test_all[df_test_all['dataset'] == ds_name].reset_index(drop=True)
        if len(df_sub) == 0:
            continue
        # 原始标签分布
        print(f"原始 Excel 中 {ds_name} 的标签分布:")
        print(df_sub['label_dict'].value_counts())

        # 检查文件是否存在，剔除无效路径
        valid_mask = df_sub['process_img_path'].apply(os.path.exists)
        if not valid_mask.all():
            invalid_count = (~valid_mask).sum()
            print(f"警告: 发现 {invalid_count} 个文件不存在，已剔除")
            df_sub = df_sub[valid_mask].reset_index(drop=True)
            if df_sub.empty:
                print(f"数据集 {ds_name} 无有效样本，跳过")
                continue
            print(f"剔除后剩余样本数: {len(df_sub)}")

        dataset = ExternalTestDataset(df_sub, name_to_idx, img_size=(256, 448))
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=True)
        print(f"样本数: {len(dataset)}")

        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        with torch.no_grad():
            for images, labels, paths, case_ids in dataloader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                logits = model(images)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # 打印实际收集到的标签分布
        print(f"实际参与推理的样本数: {len(all_labels)}")
        if len(all_labels) > 0:
            unique, counts = np.unique(all_labels, return_counts=True)
            print("labels 分布（索引 -> 类别名）:")
            for idx, count in zip(unique, counts):
                print(f"  {idx} ({FULL_CLASS_NAMES[idx]}): {count}")

        # 【关键修改】：计算指标和画图时，直接使用 FULL_CLASS_NAMES 确保矩阵被补零为 9x9
        cm, metrics = compute_metrics(all_labels, all_preds, all_probs, FULL_CLASS_NAMES)

        ds_output_dir = os.path.join(OUTPUT_BASE_DIR, ds_name)
        os.makedirs(ds_output_dir, exist_ok=True)

        # 传入 present_indices=None 强制展示完整的 9x9 矩阵
        # plot_confusion_matrix(cm, FULL_CLASS_NAMES, os.path.join(ds_output_dir, 'ConfusionMatrix.png'), present_indices=None)
        present_true_classes = sorted(list(set(all_labels)))  # 获取真实存在的标签行索引
        plot_confusion_matrix(cm, FULL_CLASS_NAMES, os.path.join(ds_output_dir, 'ConfusionMatrix.png'), 
                            present_indices=present_true_classes, rectangular=True)
        # plot_per_class_bar(metrics, FULL_CLASS_NAMES, os.path.join(ds_output_dir, 'PerClassBar.png'), present_indices=None)
        plot_per_class_bar(metrics, FULL_CLASS_NAMES, os.path.join(ds_output_dir, 'PerClassBar.png'), present_indices=present_true_classes)
        save_performance_table(metrics, FULL_CLASS_NAMES, os.path.join(ds_output_dir, 'Performance.csv'), present_indices=None)

        # 收集汇总信息
        macro = metrics['Macro Avg']
        dataset_summary_list.append({
            'Dataset': ds_name,
            'Samples': len(dataset),
            'Accuracy (Macro)': macro['Accuracy'],
            'Precision (Macro)': macro['Precision'],
            'Recall (Macro)': macro['Recall'],
            'F1 (Macro)': macro['F1'],
            'AUC (Macro)': macro['AUC']
        })
        print(f"总样本数（DataLoader）: {len(dataset)}")
        print(f"实际收集的 labels 数量: {len(all_labels)}")
        print(f"labels 分布: {np.bincount(all_labels)}")
        print(f"原始 Excel 中该数据集的标签分布:")
        print(df_sub['label_dict'].value_counts())

    print("\n===== 生成总体结果（所有数据集合并） =====")
    overall_dir = os.path.join(OUTPUT_BASE_DIR, 'Overall')
    os.makedirs(overall_dir, exist_ok=True)

    # 重新创建 DataLoader（使用全部清洗后的数据）
    df_all_test = df_test_all.reset_index(drop=True)
    dataset_all = ExternalTestDataset(df_all_test, name_to_idx, img_size=(256, 448))
    dataloader_all = DataLoader(dataset_all, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=True)

    # 准备可视化目录
    vis_root = os.path.join(overall_dir, 'vis_images')
    if ENABLE_VISUAL_CHECK:
        for name in FULL_CLASS_NAMES:
            os.makedirs(os.path.join(vis_root, name), exist_ok=True)

    all_records = []
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels, paths, case_ids in dataloader_all:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            for i in range(len(images)):
                true_label = labels[i].item()
                pred_label = preds[i].item()
                prob = probs[i, pred_label].item()
                case_id = case_ids[i]

                if ENABLE_VISUAL_CHECK:
                    # 注意这里传入 FULL_CLASS_NAMES 保证保存时不会因预测到 4、5 而索引越界
                    save_visualization(images[i], true_label, pred_label, probs[i], FULL_CLASS_NAMES, case_id, vis_root)

                all_records.append({
                    'case_id': case_id,
                    'img_path': paths[i],
                    'true_label': FULL_CLASS_NAMES[true_label],
                    'pred_label': FULL_CLASS_NAMES[pred_label],
                    'confidence': prob,
                    'status': "CORRECT" if pred_label == true_label else "WRONG"
                })

    # 保存详细表格
    df_detail = pd.DataFrame(all_records)
    df_detail.to_csv(os.path.join(overall_dir, 'detailed_predictions.csv'), index=False)

    # 计算总体指标
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # 【关键修改】：总体混淆矩阵也强制画出完整的 9x9
    cm_all, metrics_all = compute_metrics(all_labels, all_preds, all_probs, FULL_CLASS_NAMES)
    # plot_confusion_matrix(cm_all, FULL_CLASS_NAMES, os.path.join(overall_dir, 'ConfusionMatrix.png'), present_indices=None)
    present_all_true = sorted(list(set(all_labels)))  # 获取总体真实存在的标签行索引
    plot_confusion_matrix(cm_all, FULL_CLASS_NAMES, os.path.join(overall_dir, 'ConfusionMatrix.png'), 
                          present_indices=present_all_true, rectangular=True)
    # plot_per_class_bar(metrics_all, FULL_CLASS_NAMES, os.path.join(overall_dir, 'PerClassBar.png'), present_indices=None)
    plot_per_class_bar(metrics_all, FULL_CLASS_NAMES, os.path.join(overall_dir, 'PerClassBar.png'), present_indices=present_all_true)
    save_performance_table(metrics_all, FULL_CLASS_NAMES, os.path.join(overall_dir, 'Performance.csv'), present_indices=None)

    if ENABLE_GRAD_CAM:
        gradcam_dir = os.path.join(overall_dir, 'gradcam')
        backbone = model.model.model
        target_layer = backbone.layers[-2].blocks[-1].norm1
        feature_size = (16,28)   # 对应 layers[-2] (16,28)的空间尺寸 -1，feature_size=(8,14) 或 layers[-3]（feature_size=(32,56)）

        generate_gradcam(
            model=model,
            dataset=dataset_all,
            class_names=FULL_CLASS_NAMES,  # 映射名称需使用完整名称
            save_dir=gradcam_dir,
            device=DEVICE,
            target_layer=target_layer,
            feature_size=feature_size,
            max_correct_per_class=10,   # 每个类别正确样本最多10张
            max_wrong_per_class=10,     # 每个类别错误样本最多10张（若存在）
            mean=None,
            std=None,
            use_true_label=False,       # 先解释预测类别，观察效果
            cam_method="gradcam",      # 可选 "gradcam", "gradcam++", "layercam"
        )

    # 保存数据集汇总表
    if dataset_summary_list:
        df_summary = pd.DataFrame(dataset_summary_list)
        df_summary.to_csv(os.path.join(OUTPUT_BASE_DIR, 'Dataset_Summary.csv'), index=False, float_format='%.4f')
        print(f"数据集汇总表已保存至 {OUTPUT_BASE_DIR}/Dataset_Summary.csv")

    print(f"\n全部完成！结果保存在 {OUTPUT_BASE_DIR}")