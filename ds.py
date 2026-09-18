import os
import hashlib
import random
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
from SimpleITK import GetArrayFromImage, ReadImage
from monai.transforms import (
    Compose, ToTensord, ScaleIntensityd, Lambdad,
    RandFlipd, RandRotated, RandAffined,
    RandGaussianNoised, RandGaussianSmoothd, RandCoarseDropoutd,
    RandAdjustContrastd, RandShiftIntensityd,
)
from monai.transforms import RandomizableTransform


# ==================== 九分类定义 ====================
CLASS_NAMES = ['NORMAL', 'AMD', 'DR', 'ERM', 'MH', 'RRD', 'RVO', 'RAO', 'CSC']
NUM_CLASSES = len(CLASS_NAMES)
LABEL_MAP = {name: i for i, name in enumerate(CLASS_NAMES)}


# ==================== 自定义标尺噪声增强 ====================
class AddRulerNoised(RandomizableTransform):
    """
    Args:
        template_path: 标尺模板图像路径
        size: 标尺尺寸（像素）
        prob: 叠加概率
        alpha_min/max: 叠加透明度范围
        mode: 'add'=加法叠加, 'blend'=混合叠加
    """
    def __init__(self, template_path="./assets/ruler_mask.png", size=55, prob=0.8,
                 alpha_min=0.3, alpha_max=0.7, mode='blend'):
        super().__init__(prob=prob)
        self.size = size
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.mode = mode
        # 加载标尺模板
        ruler_pil = Image.open(template_path).convert("L")
        ruler_pil = ruler_pil.resize((size, size), Image.Resampling.LANCZOS)
        self.ruler = np.array(ruler_pil, dtype=np.float32) / 255.0

    def _add_ruler(self, img_2d: np.ndarray) -> np.ndarray:
        h, w = img_2d.shape
        rh, rw = self.ruler.shape
        img_float = img_2d.astype(np.float32)
        alpha = random.uniform(self.alpha_min, self.alpha_max)
        # 随机选择四个角落之一
        pos = random.choice(["bottom_left", "bottom_right", "top_left", "top_right"])
        if pos == "bottom_left":
            ys, xs = h - rh, 0
        elif pos == "bottom_right":
            ys, xs = h - rh, w - rw
        elif pos == "top_left":
            ys, xs = 0, 0
        else:
            ys, xs = 0, w - rw

        roi = img_float[ys:ys+rh, xs:xs+rw]
        if self.mode == 'add':
            overlay = self.ruler * alpha * 245
            img_float[ys:ys+rh, xs:xs+rw] = np.clip(roi + overlay, 0, 255)
        else:  # blend
            img_float[ys:ys+rh, xs:xs+rw] = np.clip(
                roi * (1 - alpha) + self.ruler * alpha * 245, 0, 255
            )
        return img_float

    def __call__(self, data):
        if random.random() > self.prob:
            return data
        img = data['image']
        if img.ndim == 3:
            img_slice = img[0] if img.shape[0] == 1 else img
        else:
            img_slice = img
        img_aug = self._add_ruler(img_slice)
        # 恢复形状
        if data['image'].ndim == 3:
            if data['image'].shape[0] == 1:
                data['image'] = img_aug[np.newaxis, ...]
            else:
                data['image'] = img_aug
        else:
            data['image'] = img_aug
        return data


# ==================== 主数据集类 ====================
class OCT_dataset(Dataset):
    """
    Args:
        xlsx_path: Excel文件路径（需包含process_img_path, label_dict, datatag列）
        config: 配置字典
            - mode: 'train'/'valid'/'test'
            - num: 分类数（固定9）
            - size: 图像尺寸 [H, W]，默认[256, 448]
            - is_debug: 是否开启调试可视化
            - use_cache: 是否启用图像缓存
            - cache_dir: 缓存目录
            - test_samples: 测试集每类采样数（-1=全部，>0=均衡采样）
    """

    def __init__(self, xlsx_path: str, config) -> None:
        super(OCT_dataset, self).__init__()

        # 防止OpenCV多线程与PyTorch DataLoader冲突
        try:
            cv2.setNumThreads(0)
            cv2.ocl.setUseOpenCL(False)
        except Exception:
            pass

        self.config = config
        self.mode = config['mode']
        self.num = config.get('num', NUM_CLASSES)
        self.size = config['size']
        self.use_cache = config.get('use_cache', True)
        self.cache_dir = config.get('cache_dir', './img_cache')
        self.is_debug = config.get('is_debug', False)
        os.makedirs(self.cache_dir, exist_ok=True)

        # 九分类标签映射
        self.label_map = LABEL_MAP
        self.class_names_list = CLASS_NAMES

        # 读取Excel
        self.data = pd.read_excel(xlsx_path)

        # 按模式筛选数据
        if self.mode == 'test':
            source_data = self.data[self.data['datatag'] == 'test']
            if len(source_data) == 0:
                print("警告: 未找到datatag='test'的数据，尝试使用'valid'...")
                source_data = self.data[self.data['datatag'] == 'valid']

            target_n = config.get('test_samples', -1)
            if target_n > 0:
                # 均衡采样：每类抽取target_n张
                print(f"测试集均衡采样: 每类 {target_n} 张...")
                balanced_dfs = []
                for label_name in self.label_map.keys():
                    df_class = source_data[source_data['label_dict'] == label_name]
                    if len(df_class) >= target_n:
                        balanced_dfs.append(df_class.sample(n=target_n, random_state=42))
                    else:
                        print(f"  [提示] {label_name} 样本不足 {target_n} (实际: {len(df_class)})，已全选")
                        balanced_dfs.append(df_class)
                self.data = pd.concat(balanced_dfs).reset_index(drop=True)
            else:
                # 全部使用
                self.data = source_data.reset_index(drop=True)
                if 'black' in self.data.columns:
                    before = len(self.data)
                    self.data = self.data[self.data['black'] == 0]
                    print(f"black列过滤（test）：移除了 {before - len(self.data)} 条")
            print(f"测试集样本数: {len(self.data)}")
        else:
            # train/valid模式
            self.data = self.data[self.data['datatag'] == self.mode].reset_index(drop=True)
            if 'black' in self.data.columns:
                before = len(self.data)
                self.data = self.data[self.data['black'] == 0]
                print(f"black列过滤（{self.mode}）：移除了 {before - len(self.data)} 条")

        self.image_list = self.data['process_img_path']
        self.label_list = self.data['label_dict']

        # 过滤不存在的文件
        valid_indices = [i for i, path in enumerate(self.image_list) if os.path.exists(path)]
        self.image_list = self.image_list.iloc[valid_indices]
        self.label_list = self.label_list.iloc[valid_indices]
        print(f"{self.mode}模式有效样本数: {len(self.image_list)}")

        # 训练/验证模式：按类别分组，用于加权采样
        if self.mode in ['train', 'valid']:
            self.set_train_data()

        # 训练模式：难区分类别加权采样
        if self.mode == 'train':
            hard_class_weights = {'ERM': 2, 'RVO': 2}
            self.weighted_class_names = []
            for name in self.class_names_list:
                weight = hard_class_weights.get(name, 1)
                self.weighted_class_names.extend([name] * weight)
            print(f"加权类别采样列表长度: {len(self.weighted_class_names)}")

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, index: int):
        # 1. 获取图像路径和标签
        if self.mode in ['train', 'valid']:
            image_path, label_str = self.get_train_data_path_and_label(index)
        else:
            image_path = self.image_list.iloc[index]
            label_str = self.label_list.iloc[index]

        # 2. 加载并预处理图像（支持缓存）
        if self.use_cache:
            path_hash = hashlib.md5(image_path.encode()).hexdigest()
            cache_subdir = os.path.join(self.cache_dir, label_str)
            os.makedirs(cache_subdir, exist_ok=True)
            cache_file = os.path.join(cache_subdir, f"{path_hash}.npy")
            if os.path.exists(cache_file):
                try:
                    image_array = np.load(cache_file)
                except Exception:
                    image_array = self._load_and_preprocess(image_path)
                    if image_array is None:
                        return self.get_fallback_sample()
                    np.save(cache_file, image_array)
            else:
                image_array = self._load_and_preprocess(image_path)
                if image_array is None:
                    return self.get_fallback_sample()
                np.save(cache_file, image_array)
        else:
            image_array = self._load_and_preprocess(image_path)
            if image_array is None:
                return self.get_fallback_sample()

        # 3. 增加通道维度
        if len(image_array.shape) == 2:
            image_array = np.expand_dims(image_array, axis=0)

        # 4. 调试可视化（增广前）
        if self.is_debug:
            self._save_debug_image(image_array, label_str, index, 'pre')

        # 5. 数据增广与变换
        data_dict = {'image': image_array}
        trans = self.transform()
        data_dict = trans(data_dict)
        image = data_dict['image']

        # 6. 调试可视化（增广后）
        if self.is_debug:
            self._save_debug_image(image, label_str, index, 'post')

        label = self.label_map.get(label_str, 0)
        return image, label

    # ==================== 图像预处理 ====================
    def _load_and_preprocess(self, image_path):
        """读取原始图像并执行预处理，返回 (H, W) float32 数组，失败返回 None"""
        try:
            image = ReadImage(image_path)
            image_array = GetArrayFromImage(image)
        except Exception as e:
            print(f"读取图像失败: {e}, 路径: {image_path}")
            return None
        # 转为2D
        if len(image_array.shape) == 3:
            image_array = image_array[0]
        # 预处理流水线
        image_array = self.detect_orientation(image_array)
        image_array = self.crop_top_bottom(image_array, threshold=5, margin=10)
        image_array = self.adaptive_resize(
            image_array, target_h=self.size[0], target_w=self.size[1], fill_value=0
        )
        return image_array.astype(np.float32)

    def detect_orientation(self, image: np.ndarray) -> np.ndarray:
        """
        检测图像方向并统一为水平分层（视网膜层理水平）。
        通过比较中心区域行/列像素和的方差判断方向。
        """
        h, w = image.shape
        if w / h > 2.0:
            return image
        # 中心裁剪60%区域
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
        # 列方差显著大于行方差（>1.2倍），说明层理垂直，旋转90度
        if col_profile_var > row_profile_var * 1.2:
            image = np.rot90(image, k=-1)
        return image

    def crop_top_bottom(self, image, threshold=5, margin=10):
        """裁剪图像上下方向的黑色背景，保留左右完整"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        h, w = gray.shape
        row_max = np.max(gray, axis=1)
        valid_rows = np.where(row_max > threshold)[0]
        if len(valid_rows) == 0:
            return image
        top = max(0, valid_rows[0] - margin)
        bottom = min(h, valid_rows[-1] + margin)
        return image[top:bottom, :]

    def adaptive_resize(self, image, target_h=256, target_w=448, fill_value=0):
        """
        根据输入图像宽高比与目标宽高比的关系，自适应裁剪或填充，再缩放到目标尺寸。
        - 图像更扁：裁剪左右
        - 图像更窄：填充左右黑边
        """
        h, w = image.shape
        target_aspect = target_w / target_h
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

    # ==================== 数据获取 ====================
    def get_train_data_path_and_label(self, index: int):
        """训练/验证模式：加权随机选取图像路径和标签"""
        if hasattr(self, 'weighted_class_names'):
            total_weight = len(self.weighted_class_names)
            class_name = self.weighted_class_names[index % total_weight]
        else:
            class_idx = index % len(self.class_names_list)
            class_name = self.class_names_list[class_idx]

        img_paths_series = getattr(self, class_name)
        if len(img_paths_series) == 0:
            # 回退到非空类别
            for fallback_name in self.class_names_list:
                fallback_series = getattr(self, fallback_name)
                if len(fallback_series) > 0:
                    img_paths_series = fallback_series
                    class_name = fallback_name
                    break

        rand_id = np.random.randint(0, len(img_paths_series))
        image_path = img_paths_series.iloc[rand_id]
        return image_path, class_name

    def set_train_data(self):
        """对训练/验证数据按类别分组，动态创建 self.CLASS_NAME 属性"""
        self.label_list = self.label_list.astype(str)
        for cls_name in self.class_names_list:
            img_paths = self.image_list[self.label_list == cls_name]
            setattr(self, cls_name, img_paths)
            print(f"  {cls_name}: {len(img_paths)}")

    # ==================== 数据增广 ====================
    def transform(self):
        """数据增广与预处理变换（按模式区分）"""
        if self.mode == 'train':
            trans = Compose([
                Lambdad(keys=['image'], func=lambda x: x.astype(np.float32)),
                # 标尺噪声增强
                AddRulerNoised(
                    template_path="./assets/ruler_mask.png",
                    size=55, prob=0.8, alpha_min=0.6, alpha_max=0.9, mode='add'
                ),
                # 几何变换增广
                RandFlipd(keys=['image'], prob=0.2, spatial_axis=[1]),
                RandFlipd(keys=['image'], prob=0.2, spatial_axis=[0]),
                RandRotated(keys=['image'], range_x=0.26, prob=0.3, mode='bilinear', padding_mode='zeros'),
                RandAffined(keys=['image'], prob=0.3, translate_range=30, scale_range=(0, 0.2), rotate_range=0.1),
                # 像素强度增广
                RandGaussianNoised(keys=['image'], prob=0.3, mean=0.0, std=0.03),
                RandGaussianSmoothd(keys=['image'], prob=0.1, sigma_x=(0.3, 0.7), sigma_y=(0.3, 0.7)),
                RandCoarseDropoutd(keys=['image'], prob=0.2, holes=1, spatial_size=12, fill_value=0),
                RandAdjustContrastd(keys=['image'], prob=0.3, gamma=(0.8, 1.2)),
                RandShiftIntensityd(keys=['image'], prob=0.1, offsets=(0.0, 0.1)),
                # 标准化
                ScaleIntensityd(keys=['image'], minv=0.0, maxv=1.0),
                ToTensord(keys=['image']),
            ])
        else:
            # valid/test模式：仅标准化，无增广
            trans = Compose([
                Lambdad(keys=['image'], func=lambda x: x.astype(np.float32)),
                ScaleIntensityd(keys=['image'], minv=0.0, maxv=1.0),
                ToTensord(keys=['image']),
            ])
        return trans

    # ==================== 辅助函数 ====================
    def get_fallback_sample(self):
        """返回备用样本防止程序中断"""
        empty_image = torch.zeros((1, self.size[0], self.size[1]), dtype=torch.float32)
        return empty_image, 0

    def _save_debug_image(self, image_array, label_str, index, stage):
        """保存调试图像（增广前/后）"""
        debug_dir = f'./debug_{stage}_aug'
        os.makedirs(debug_dir, exist_ok=True)
        if isinstance(image_array, torch.Tensor):
            img_np = image_array[0].detach().cpu().numpy()
        else:
            img_np = image_array[0] if image_array.ndim == 3 else image_array
        img_np = (img_np * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(debug_dir, f'{stage}_{index}_{label_str}.png'), img_np)
