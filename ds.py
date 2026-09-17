import numpy as np
from scipy import ndimage
import torch
from torch.utils.data import Dataset, DataLoader
from monai.transforms import Compose, ToTensord, Resized, HistogramNormalized, RandGaussianNoised, RandAdjustContrastd, ScaleIntensityd, Lambdad
from monai.transforms import RandFlipd, RandRotated, RandAffined, RandGaussianSmoothd, RandCoarseDropoutd, RandShiftIntensityd
import pandas as pd
from SimpleITK import GetArrayFromImage, ReadImage
import os
import cv2
import matplotlib.pyplot as plt
import random
from PIL import Image
from monai.transforms import RandomizableTransform


os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 仅启用第 0 块 GPU

# ====================== 自定义标尺噪声增强 ======================
class AddRulerNoised(RandomizableTransform):
    def __init__(self, template_path="./ruler_mask.png", size=55, prob=0.8,
                 alpha_min=0.3, alpha_max=0.7, mode='blend'):
        super().__init__(prob=prob)
        self.size = size
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.mode = mode
        # 加载模板
        ruler_pil = Image.open(template_path).convert("L")
        ruler_pil = ruler_pil.resize((size, size), Image.Resampling.LANCZOS)
        self.ruler = np.array(ruler_pil, dtype=np.float32) / 255.0

    def _add_ruler(self, img_2d: np.ndarray) -> np.ndarray:
        h, w = img_2d.shape
        rh, rw = self.ruler.shape
        img_float = img_2d.astype(np.float32)
        alpha = random.uniform(self.alpha_min, self.alpha_max)
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
        # ====== 关键修复：强制叠加，忽略概率 ======
        # 原 if not self.randomize(None): return data 可能因版本问题导致 prob=1 仍返回 False
        # 为了调试，我们先强制执行；正式使用时可以恢复随机条件，但建议改用以下方式
        # 若要保留概率，可改为：
        # if random.random() > self.prob: return data
        # 因为 self.prob 是传入的概率，直接使用 random.random() 控制
        if random.random() > self.prob:
            return data

        img = data['image']
        if img.ndim == 3:
            # 假设形状为 (1, H, W) 或 (C, H, W)，取第一通道
            img_slice = img[0] if img.shape[0] == 1 else img
        else:
            img_slice = img
        img_aug = self._add_ruler(img_slice)
        # 恢复形状
        if data['image'].ndim == 3:
            if data['image'].shape[0] == 1:
                data['image'] = img_aug[np.newaxis, ...]
            else:
                # 如果是多通道，需要重建（但您的图像是单通道，所以不会到这里）
                data['image'] = img_aug
        else:
            data['image'] = img_aug
        return data

class OCT_dataset(Dataset):
    
    def __init__(self, xlsx_path:str, config) -> None:
        super(OCT_dataset, self).__init__()
        
        # 防止OpenCV多线程与PyTorch DataLoader冲突
        try:
            cv2.setNumThreads(0)
            cv2.ocl.setUseOpenCL(False)
        except: pass
        
        self.use_cache = config.get('use_cache', True)  # 默认启用缓存

        self.config = config
        self.mode = self.config['mode']
        self.num = self.config['num']
        self.size = self.config['size']
        self.cache_dir = config.get('cache_dir', './img_cache')
        os.makedirs(self.cache_dir, exist_ok=True) 
        self.is_debug = self.config.get('is_debug', False) # 是否开启调试模式（可视化）
        
        self.class_names_list = [] 
        # 读取数据
        self.data = pd.read_excel(xlsx_path)

        # === 根据分类数量预处理标签 ===
        if self.num == 6:
            # 定义哪些归为 OTHERS
            others_list = ['AMD', 'CSC', 'ERM', 'RVO', 'VID', 'RRD', 'RAO']
            # 将这些标签统一修改为 'OTHERS'
            self.data.loc[self.data['label_dict'].isin(others_list), 'label_dict'] = 'OTHERS'
            
            # 定义 6分类+OTHERS 的映射
            self.label_map = {
                'NORMAL': 0, 'CNV': 1, 'DME': 2, 'DRUSEN': 3, 'MH': 4, 'DR': 5, 'OTHERS': 6
            }
            self.class_names_list = ['NORMAL', 'CNV', 'DME', 'DRUSEN', 'MH', 'DR', 'OTHERS']

        elif self.num == 9:
            # 定义9分类的逻辑
            # others_list = ['VID', 'RRD', 'RAO', 'RVO'] # 剩下的归为 OTHERS
            # self.data.loc[self.data['label_dict'].isin(others_list), 'label_dict'] = 'OTHERS'
            
            self.label_map = {
                'NORMAL': 0, 'AMD': 1, 'DR': 2, 'ERM': 3, 'MH': 4, 'RRD': 5, 'RVO': 6, 'RAO': 7, 'CSC': 8
            }
            self.class_names_list = ['NORMAL', 'AMD', 'DR', 'ERM', 'MH', 'RRD', 'RVO', 'RAO', 'CSC']
        
        elif self.num == 13:
            # 全分类模式：不进行 OTHERS 合并，直接映射所有标签
            self.label_map = {
                'NORMAL': 0, 'CNV': 1, 'DME': 2, 'DRUSEN': 3, 'MH': 4, 'DR': 5,
                'AMD': 6, 'CSC': 7, 'ERM': 8, 'RVO': 9, 'VID': 10, 'RRD': 11, 'RAO': 12
            }
            self.class_names_list = [
                'NORMAL', 'CNV', 'DME', 'DRUSEN', 'MH', 'DR', 
                'AMD', 'CSC', 'ERM', 'RVO', 'VID', 'RRD', 'RAO'
            ]
        
        elif self.num == 3:
            # AMD 三分类：DRUSEN, CNV, AMD
            self.label_map = {
                'GA': 0,
                'DRUSEN': 1,
                'CNV': 2
            }
            self.class_names_list = ['GA', 'DRUSEN', 'CNV']

        elif self.num == 11:
            # 排除AMD 和 VID 分类，其他11类不变
            self.label_map = {
                'NORMAL': 0, 'CNV': 1, 'DME': 2, 'DRUSEN': 3, 'MH': 4, 'DR': 5,
                'CSC': 6, 'ERM': 7, 'RVO': 8, 'RRD': 9, 'RAO': 10
            }
            self.class_names_list = ['NORMAL', 'CNV', 'DME', 'DRUSEN', 'MH', 'DR', 
                'CSC', 'ERM', 'RVO', 'RRD', 'RAO']
        
        elif self.num == 12:
            # 拆解 AMD 为三种类型，排除 VID
            self.label_map = {
                'NORMAL': 0, 'DRUSEN': 1, 'GA': 2, 'CNV': 3, 'DME': 4, 'DR': 5,
                'ERM': 6, 'MH': 7, 'RRD': 8, 'RVO': 9, 'RAO': 10, 'CSC': 11
            }
            self.class_names_list = ['NORMAL', 'DRUSEN', 'GA', 'CNV', 'DME', 'DR', 
                'ERM', 'MH', 'RRD', 'RVO', 'RAO', 'CSC']
        
        #  === Test 模式逻辑优化 ===
        if self.mode == 'test':
            # 1. 确定数据源：只取 datatag 为 'test' 的行
            source_data = self.data[self.data['datatag'] == 'test']
            
            if len(source_data) == 0:
                print("警告: 未在Excel中找到 datatag='test' 的数据！尝试使用 'valid' 数据作为备选...")
                source_data = self.data[self.data['datatag'] == 'valid']

            # 2. 获取采样配置：-1 代表全部，>0 代表具体数量
            target_n = self.config.get('test_samples', -1) 
            
            if target_n > 0:
                # --- 方案 A: 均衡采样---
                print(f"正在构建测试集 (均衡采样: 每类 {target_n} 张)...")
                balanced_dfs = []
                for label_name in self.label_map.keys():
                    df_class = source_data[source_data['label_dict'] == label_name]
                    if len(df_class) >= target_n:
                        balanced_dfs.append(df_class.sample(n=target_n, random_state=42))
                    else:
                        print(f"  [提示] {label_name} 样本不足 {target_n} (实际: {len(df_class)})，已全选。")
                        balanced_dfs.append(df_class)
                self.data = pd.concat(balanced_dfs).reset_index(drop=True)
                
            else:
                # --- 方案 B: 全部使用 (All Data) ---
                print("正在构建测试集 (使用全部 Test 数据)...")
                self.data = source_data.reset_index(drop=True)
                if 'black' in self.data.columns:
                    before = len(self.data)
                    self.data = self.data[self.data['black'] == 0]
                    print(f"black 列过滤（test）：移除了 {before - len(self.data)} 条 black=1 的数据")
            print(f"测试集准备完成，总样本数: {len(self.data)}")
            
        else:
            self.data = self.data[self.data['datatag'] == self.mode]
            self.data = self.data.reset_index(drop=True)
            # 新增 black 过滤
            if 'black' in self.data.columns:
                before = len(self.data)
                self.data = self.data[self.data['black'] == 0]
                print(f"black 列过滤：移除了 {before - len(self.data)} 条 black=1 的数据")

        self.image_list = self.data['process_img_path']
        self.label_list = self.data['label_dict']
        
        # 过滤掉不存在的文件
        valid_indices = []
        for i, path in enumerate(self.image_list):
            if os.path.exists(path):
                valid_indices.append(i)
        
        self.image_list = self.image_list.iloc[valid_indices]
        self.label_list = self.label_list.iloc[valid_indices]
        
        print(f"{self.mode}模式下的有效样本数: {len(self.image_list)}")
        
        # 初始化分类数据池
        if self.mode=='train':
            self.set_train_data()
        elif self.mode=='valid':
            self.set_train_data()

        if self.mode == 'train':
            # 定义难区分的类别及其采样权重（权重为整数，例如 2 表示采样概率是普通类别的2倍）
            hard_class_weights = {
                'ERM': 2,
                'RVO': 2,
                # 'AMD': 1.5,
                # 'DRUSEN': 2,
                # 可继续添加其他类别
            }
            self.weighted_class_names = []
            for name in self.class_names_list:
                weight = hard_class_weights.get(name, 1)
                self.weighted_class_names.extend([name] * weight)
            print(f"加权类别采样列表长度: {len(self.weighted_class_names)} (原始类别数: {len(self.class_names_list)})")
        

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, index:int):
        import hashlib

        # 1. 获取图像路径和标签
        if self.mode in ['train', 'valid']:
            image_path, label_str = self.get_train_data_path_and_label(index)
        else:
            image_path = self.image_list.iloc[index]
            label_str = self.label_list.iloc[index]

        # 2. 根据 use_cache 决定加载方式
        if self.use_cache:
            path_hash = hashlib.md5(image_path.encode()).hexdigest()
            cache_subdir = os.path.join(self.cache_dir, label_str)
            os.makedirs(cache_subdir, exist_ok=True)
            cache_file = os.path.join(cache_subdir, f"{path_hash}.npy")
            if os.path.exists(cache_file):
                try:
                    image_array = np.load(cache_file)
                except Exception as e:
                    print(f"加载缓存失败 {cache_file}: {e}，重新生成")
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
            # 不缓存，直接预处理
            image_array = self._load_and_preprocess(image_path)
            if image_array is None:
                return self.get_fallback_sample()

        # 3. 后续处理（增加通道、调试、增广等）保持不变
        if len(image_array.shape) == 2:
            image_array = np.expand_dims(image_array, axis=0)

        if self.is_debug:
            pre_aug_dir = './debug_pre_aug'
            os.makedirs(pre_aug_dir, exist_ok=True)
            pre_img = image_array[0].astype(np.uint8)
            cv2.imwrite(os.path.join(pre_aug_dir, f'pre_{index}_{label_str}.png'), pre_img)

        data_dict = {'image': image_array}
        trans = self.transform()
        data_dict = trans(data_dict)
        image = data_dict['image']

        if self.is_debug:
            post_aug_dir = './debug_post_aug'
            os.makedirs(post_aug_dir, exist_ok=True)
            img_np = image[0].detach().cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(post_aug_dir, f'post_{index}_{label_str}.png'), img_np)
            self.plot_check(image, label_str, index)
            print(f"原始标签字符串: '{label_str}', 映射后数字: {self.label_map.get(label_str, 0)}")

        label = self.label_map.get(label_str, 0)
        return image, label

    # def __getitem__(self, index:int):

        # 1. 根据模式获取原始图像数据
        if self.mode=='train':
            image_array, label_str = self.get_train_data(index) #doujiancha 
        elif self.mode=='valid':
            image_array, label_str = self.get_train_data(index)
        elif self.mode=='test':
            image_array, label_str = self.get_test_data(index)

        # 2. 预处理：确保是2D图像 (H, W)
        if len(image_array.shape) == 3:
            image_array = image_array[0] # 取第一个切片

        # 3. 核心功能：自动检测并调整方向（确保水平分层）
        image_array = self.detect_orientation(image_array)

        # # ---- 新增：自动检测视网膜主体并裁剪 ----
        # if self.config.get('crop_body', False):
        #     image_array = self.crop_retinal_body(image_array, min_area_ratio=0.05, dilation_iter=3, margin=10)
        
        image_array = self.crop_top_bottom(image_array, threshold=5, margin=10)

        # 4. 核心功能：先填充后等比缩放 (Pad -> Resize)
        image_array = self.adaptive_resize(image_array, target_h=self.size[0], target_w=self.size[1], fill_value=0)

        # --- 新增：强制转换为 float32，并检查 NaN/Inf ---
        image_array = image_array.astype(np.float32)
        if np.isnan(image_array).any() or np.isinf(image_array).any():
            print(f"Warning: NaN or Inf in image at index {index}, using fallback")
            return self.get_fallback_sample()
        
        # 5. 增加通道维度 (H, W) -> (1, H, W)
        if len(image_array.shape) == 2:
            image_array = np.expand_dims(image_array, axis=0)
        
        image_array = image_array.astype(np.float32) # xizneng

        label = self.label_map.get(label_str, 0)
        
        # 如果debug模式，保存增广前的图像
        if self.is_debug:
            pre_aug_dir = './debug_pre_aug'
            os.makedirs(pre_aug_dir, exist_ok=True)
            pre_img = image_array[0].astype(np.uint8) 
            cv2.imwrite(os.path.join(pre_aug_dir, f'pre_{index}_{label_str}.png'), pre_img)

        # 6. 数据增广与变换
        data_dict = {'image': image_array}
        trans = self.transform()
        data_dict = trans(data_dict)
        image = data_dict['image']

        # 如果debug模式，保存增广后的图像（原有plot_check也可保留）
        if self.is_debug:
            post_aug_dir = './debug_post_aug'
            os.makedirs(post_aug_dir, exist_ok=True)
            img_np = image[0].detach().cpu().numpy()  # 值域 [0,1]
            img_np = (img_np * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(post_aug_dir, f'post_{index}_{label_str}.png'), img_np)

        # 原有的 plot_check 会额外生成带标题的matplotlib图，可保留或注释掉
        if self.is_debug:
            self.plot_check(image, label_str, index)
            print(f"原始标签字符串: '{label_str}', 映射后数字: {label}")
        # image = torch.tensor(image_array, dtype=torch.float32) # 暂不进行数据增广           

        # === 可视化 Plot ===
        # 仅在Debug模式或为了检查时运行，避免拖慢训练
        # if self.is_debug:
        #     self.plot_check(image, label_str, index)
        #     print(f"原始标签字符串: '{label_str}', 映射后数字: {label}")

        return image, label
            
    def crop_retinal_body(self, image, min_area_ratio=0.05, dilation_iter=3, margin=10):
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

    def crop_top_bottom(self, image, threshold=5, margin=10):
        """
        裁剪图像上下方向的黑色背景区域，保留左右完整。
        基于每行像素最大值检测有效内容区域。
        
        Args:
            image: 灰度图 (H, W)
            threshold: 有效像素阈值，高于此值视为非背景
            margin: 在有效边界外保留的像素数，防止切到边缘
        
        Returns:
            裁剪后图像 (H', W)，其中 H' ≤ H
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        h, w = gray.shape
        # 每行最大值
        row_max = np.max(gray, axis=1)
        # 找到有效行
        valid_rows = np.where(row_max > threshold)[0]
        if len(valid_rows) == 0:
            # 没有有效内容，返回原图
            return image
        top = max(0, valid_rows[0] - margin)
        bottom = min(h, valid_rows[-1] + margin)
        # 裁剪上下，宽度不变
        cropped = image[top:bottom, :]
        return cropped

    def plot_check(self, tensor_img, label_str, index):
        """保存增广后的图像进行检查"""
        save_dir = './debug_result'
        os.makedirs(save_dir, exist_ok=True)
        
        # Tensor转numpy: (1, H, W) -> (H, W)
        img_np = tensor_img[0].detach().cpu().numpy()
        
        plt.figure(figsize=(6, 3))
        plt.imshow(img_np, cmap='gray')
        plt.title(f"Label: {label_str} | ID: {index}")
        plt.axis('off')
        
        save_path = os.path.join(save_dir, f"aug_{index}_{label_str}.png")
        plt.savefig(save_path)
        plt.close() # 关闭以释放内存
        print(f"已保存Debug图像: {save_path}")

    def adaptive_resize(self, image, target_h=256, target_w=448, fill_value=0):
        """
        根据输入图像宽高比与目标宽高比的关系，自适应裁剪或填充，再缩放到目标尺寸。
        
        Args:
            image: 输入图像 (H, W)
            target_h: 目标高度 (256)
            target_w: 目标宽度 (448)
            fill_value: 填充像素值（0=黑色）
        
        Returns:
            resized: 输出图像 (target_h, target_w)
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

    def pad_and_resize(self, image, target_h, target_w):
        """
        先填充，再等比例缩放。
        """
        h, w = image.shape
        target_h, target_w = self.size
        
        # 计算目标宽高比
        target_aspect = target_h / target_w
        img_aspect = h / w
        
        pad_h, pad_w = 0, 0
        
        # 如果图像比目标“更高/更窄”，需要填充宽度
        if img_aspect > target_aspect:
            new_w = int(h / target_aspect)
            pad_w = new_w - w
            top, bottom = 0, 0
            left, right = pad_w // 2, pad_w - (pad_w // 2)
        
        # 如果图像比目标“更矮/更宽”，需要填充高度
        else:
            new_h = int(w * target_aspect)
            pad_h = new_h - h
            top, bottom = pad_h // 2, pad_h - (pad_h // 2)
            left, right = 0, 0
            
        # 1. 填充 (Padding)
        # 这里的 value=0 表示用黑色填充, value=255表示白色填充
        image_padded = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        
        # 2. 缩放 (Resize) 到目标尺寸
        image_resized = cv2.resize(image_padded, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        
        return image_resized
    
    def get_fallback_sample(self):
        """返回备用样本防止程序中断"""
        empty_image = torch.zeros((1, self.size[0], self.size[1]), dtype=torch.float32)
        return empty_image, 0

    def get_train_data_path_and_label(self, index:int):
        """返回训练/验证模式下随机选取的图像路径和标签字符串"""
        if hasattr(self, 'weighted_class_names'):
            total_weight = len(self.weighted_class_names)
            class_name = self.weighted_class_names[index % total_weight]
        else:
            num_classes = len(self.class_names_list)
            class_idx = index % num_classes
            class_name = self.class_names_list[class_idx]

        img_paths_series = getattr(self, class_name)
        if len(img_paths_series) == 0:
            for fallback_name in self.class_names_list:
                fallback_series = getattr(self, fallback_name)
                if len(fallback_series) > 0:
                    img_paths_series = fallback_series
                    class_name = fallback_name
                    break

        rand_id = np.random.randint(0, len(img_paths_series))
        image_path = img_paths_series.iloc[rand_id]
        label = class_name
        return image_path, label

    def get_train_data(self, index:int):
        # 动态计算当前 index 应该取哪个类别
        # 例如 7分类时，num_classes=7，index % 7 得到 0~6
        if hasattr(self, 'weighted_class_names'):
            total_weight = len(self.weighted_class_names)
            class_name = self.weighted_class_names[index % total_weight]
        else:
            # 回退到原始均衡采样（适用于 valid 模式或其他情况）
            num_classes = len(self.class_names_list)
            class_idx = index % num_classes
            class_name = self.class_names_list[class_idx]

        # 获取该类别的图片列表
        img_paths_series = getattr(self, class_name)

        # 如果该类别没有数据（如 valid 集中某类为空），则随机取一个非空类别作为回退
        if len(img_paths_series) == 0:
            for fallback_name in self.class_names_list:
                fallback_series = getattr(self, fallback_name)
                if len(fallback_series) > 0:
                    img_paths_series = fallback_series
                    class_name = fallback_name
                    break

        # 随机抽取一张图片
        rand_id = np.random.randint(0, len(img_paths_series))
        image_path = img_paths_series.iloc[rand_id]
        label = class_name

        if not os.path.exists(image_path):
            print(f"文件不存在: {image_path}")
            return self.get_fallback_sample()

        try:
            image = ReadImage(image_path)
            image_array = GetArrayFromImage(image)
            return image_array, label
        except Exception as e:
            print(f"读取图像时出错: {e}, 路径: {image_path}")
            return self.get_fallback_sample()
        
    def get_valid_data(self, index:int):
        image_path = self.image_list.iloc[index]
        label = self.label_list.iloc[index]
        
        try:
            image = ReadImage(image_path)
            image_array = GetArrayFromImage(image)
            return image_array, label
        except Exception as e:
            print(f"读取图像时出错: {e}, 路径: {image_path}")
            return self.get_fallback_sample()
    
    def get_test_data(self, index:int):
        """test模式专用：直接读取指定index的样本，无随机采样、无数据增广"""
        image_path = self.image_list.iloc[index]
        label_str = self.label_list.iloc[index]
        
        try:
            image = ReadImage(image_path)
            image_array = GetArrayFromImage(image)
            return image_array, label_str
        except Exception as e:
            print(f"读取测试图像时出错: {e}, 路径: {image_path}")
            return self.get_fallback_sample()
        
    def set_train_data(self):
        """对训练数据进行按类别分组"""
        self.label_list = self.label_list.astype(str)
        # 动态创建属性，例如 self.NORMAL, self.CNV ...
        for cls_name in self.class_names_list:
            # 筛选出属于该类别的所有图片路径
            img_paths = self.image_list[self.label_list == cls_name]
            # 动态绑定到实例变量，例如 self.NORMAL = ...
            setattr(self, cls_name, img_paths)
            print(f"  {cls_name}: {len(img_paths)}")

    def transform(self):
        """数据增广与预处理"""
        if self.mode == 'train':
            trans = Compose([
                Lambdad(keys=['image'], func=lambda x: x.astype(np.float32)),
                # ----- 标尺噪声（仅训练时添加）-----
                AddRulerNoised(
                    template_path="./ruler_mask.png",   # 确保此文件存在
                    size=55,
                    prob=0.8,
                    alpha_min=0.6,
                    alpha_max=0.9,
                    mode='add'          # 训练用 blend，更自然
                ),
                 # 1. 几何变换增广
                RandFlipd(keys=['image'], prob=0.2, spatial_axis=[1]),  # 水平翻转 1 0.2
                RandFlipd(keys=['image'], prob=0.2, spatial_axis=[0]),  # 新增垂直翻转
                RandRotated(keys=['image'], range_x=0.26, prob=0.3, mode='bilinear', padding_mode='zeros'), # 轻微旋转 1 0.3
                RandAffined(keys=['image'], prob=0.3, translate_range=30, scale_range=(0, 0.2), rotate_range=0.1), # 平移和缩放  调整不同的比例放大比例高，无缩小
                
                # 2. 像素强度增广
                RandGaussianNoised(keys=['image'], prob=0.3, mean=0.0, std=0.03), # 高斯噪声 1 0.3
                RandGaussianSmoothd(keys=['image'], prob=0.1, sigma_x=(0.3, 0.7), sigma_y=(0.3, 0.7)), # 新增高斯模糊 1 0.1  降低阈值
                RandCoarseDropoutd(keys=['image'], prob=0.2, holes=1, spatial_size=12, fill_value=0), # 新增随机擦除 1 0.2  降低size
                RandAdjustContrastd(keys=['image'], prob=0.3, gamma=(0.8, 1.2)), # 对比度调整 1 0.3 控制参数调整变弱
                RandShiftIntensityd(keys=['image'], prob=0.1, offsets=(0.0, 0.1)), # 新增强度偏移 1 0.2  暗度调0，亮度调0.1

                # 3. 标准化
                ScaleIntensityd(keys=['image'], minv=0.0, maxv=1.0),
                ToTensord(keys=['image']), 
            ])
        elif self.mode == 'valid':
            trans = Compose([
                Lambdad(keys=['image'], func=lambda x: x.astype(np.float32)),
                ScaleIntensityd(keys=['image'], minv=0.0, maxv=1.0),
                ToTensord(keys=['image']),
            ])
        elif self.mode == 'test':
            trans = Compose([
                Lambdad(keys=['image'], func=lambda x: x.astype(np.float32)),
                ScaleIntensityd(keys=['image'], minv=0.0, maxv=1.0),
                ToTensord(keys=['image']),
            ])
        return trans
    
    def detect_orientation(self, image: np.ndarray) -> np.ndarray:
        """检测图像方向并统一方向"""
        h, w = image.shape

        if w / h > 2.0:
            return image

        # 中心裁剪 (Center Crop) 只取图像中心 60% 的区域进行判断
        h_start, h_end = int(h * 0.2), int(h * 0.8)
        w_start, w_end = int(w * 0.2), int(w * 0.8)
        # 防止图像太小导致切片为空
        if h_end - h_start < 10 or w_end - w_start < 10:
            center_img = image # 图像太小就不切了
        else:
            center_img = image[h_start:h_end, w_start:w_end]

        # 转换为浮点型防止溢出
        img_float = center_img.astype(np.float32)
        min_val, max_val = img_float.min(), img_float.max()
        if max_val > min_val:
            img_float = (img_float - min_val) / (max_val - min_val)

        row_profile_var = np.var(np.sum(img_float, axis=1))
        col_profile_var = np.var(np.sum(img_float, axis=0))
        
        if col_profile_var > row_profile_var * 1.2:
            image = np.rot90(image, k=-1)

        return image
        
    def read_img(self, path, label):
        if not os.path.exists(path): return np.zeros(self.size), label
        try:
            img = ReadImage(path)
            arr = GetArrayFromImage(img)
            return arr, label
        except:
            return np.zeros(self.size), label
    
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
        # 方向检测
        image_array = self.detect_orientation(image_array)
        # 上下裁剪
        image_array = self.crop_top_bottom(image_array, threshold=5, margin=10)
        # 自适应resize
        image_array = self.adaptive_resize(
            image_array,
            target_h=self.size[0],
            target_w=self.size[1],
            fill_value=0
        )
        return image_array.astype(np.float32)
# if __name__ == '__main__':
#     # === 修改后的测试入口，避免OOM ===
#     print("开始Dataset测试...")
    
#     config = {'mode':'train','num':12,'size':[256,448],'is_debug':True}
#     train_xlsx = '/home/chenxinfei/OCT/视网膜疾病分类_train.xlsx'
    
#     if os.path.exists(train_xlsx):
#         ds = OCT_dataset(train_xlsx, config=config)
#         # num_workers=0 避免多进程带来的额外开销
#         dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)
        
#         print("正在生成Debug图像到 ./debug_result 文件夹...")
#         for i, batch in enumerate(dl):
#             imgs, labels = batch
#             print(f"Batch {i}Shape: {imgs.shape} | Labels: {labels}")
            
#             # 只跑3个batch用于验证，避免时间过长
#             if i >= 2: 
#                 print("测试完成。请检查 ./debug_result 文件夹中的图像。")
#                 break
#     else:
#         print("Excel路径不存在，请修改路径。")



if __name__ == '__main__':
    from model import OCT_Model
    import time

    # 1. 数据预处理（包括数据增广）
    train_xlsx = '/data/chenxinfen/OCT1/new9/视网膜疾病分类_train_splitV5.xlsx'
    log_dir = '/data/chenxinfen/ModelTraining_monai/Training_code_V5.1/logs/logs_9cls'

    ds_train = OCT_dataset(train_xlsx, config={'mode':'train','num':9,'size':[256,448],'is_debug':True,'crop_body': False })
    ds_valid = OCT_dataset(train_xlsx, config={'mode':'valid','num':9,'size':[256,448],'is_debug':True,'crop_body': False })
    dl_train = DataLoader(ds_train,
                          batch_size=128,
                          shuffle=True,
                          num_workers=4)   # 8 / 16
    dl_valid = DataLoader(ds_valid,
                          batch_size=128,
                          shuffle=True,
                          num_workers=4)

    model = OCT_Model(num_classes=9)
    model.to('cuda')
    start_time = time.time()
    for idx_batch, batch in enumerate(dl_train):
        print("batch {}-load_time {:.3f}s".format(idx_batch, time.time() - start_time), end='\r')
        print('label =',batch[1])
        model.forward(batch[0].to('cuda'))
# if __name__ == '__main__':
#     test_img = np.ones((256, 448), dtype=np.float32) * 128
#     aug = AddRulerNoised(prob=1.0, mode='add')
#     data = {'image': test_img}
#     out = aug(data)
#     cv2.imwrite("test_call.jpg", out['image'])
#     print("通过 __call__ 生成 test_call.jpg，请查看是否有标尺。")
# del model # 释放模型内存case id -1  copy