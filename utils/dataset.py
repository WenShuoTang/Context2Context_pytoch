import torch
from torch.utils.data import Dataset
import numpy as np

class SimilarPixelBankDataset(Dataset):
    def __init__(self, pixel_bank, length=1):
        """
        pixel_bank: 形状为 (H, W, N) 的复数相似像素库
        length: 每个 epoch 的迭代次数。由于是 Zero-shot 单图自训练，通常设为 1 即可
        """
        self.pixel_bank = pixel_bank
        self.H, self.W, self.N = pixel_bank.shape
        self.length = length
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        # 像素级随机不重复抽样生成 2 个不相同的索引
        idx1 = np.random.randint(0, self.N, size=(self.H, self.W))
        offset = np.random.randint(1, self.N, size=(self.H, self.W))
        idx2 = (idx1 + offset) % self.N  # 确保 idx2 != idx1
        
        grid_h, grid_w = np.meshgrid(np.arange(self.H), np.arange(self.W), indexing='ij')
        
        # 抽取复数像素样本
        S1 = self.pixel_bank[grid_h, grid_w, idx1]
        S2 = self.pixel_bank[grid_h, grid_w, idx2]
        
        # 转换为强度域（Intensity Domain）: I = |S|^2
        I1 = np.abs(S1) ** 2
        I2 = np.abs(S2) ** 2
        
        # 转换至对数域（Logarithmic Domain）：Z = log(I + eps) 避免溢出
        eps = 1e-8
        y1 = np.log(I1 + eps)
        y2 = np.log(I2 + eps)
        
        # 转为 PyTorch Tensor 并增加通道维度 -> (1, H, W)
        y1_tensor = torch.from_numpy(y1).float().unsqueeze(0)
        y2_tensor = torch.from_numpy(y2).float().unsqueeze(0)
        
        return y1_tensor, y2_tensor