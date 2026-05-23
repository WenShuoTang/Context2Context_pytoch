import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.network import Context2ContextNet
from utils.ccm_search import construct_similar_pixel_bank
from utils.dataset import SimilarPixelBankDataset
import numpy as np

def train_context2context(slc_image_path, save_model_path='checkpoint.pth', num_epochs=500, M=9, N=20):
    # 1. 读取或生成模拟的复数 SLC 图像
    if os.path.exists(slc_image_path):
        print(f"-> 正在加载真实的复数标量 SAR 数据: {slc_image_path}")
        S = np.load(slc_image_path)
    else:
        print("-> 未找到数据。正在自动生成 256x256 的仿真模拟复数 SLC 图像...")
        H, W = 256, 256
        clean_ref = np.zeros((H, W))
        clean_ref[64:192, 64:192] = 2.0
        clean_ref[100:150, 100:150] = 5.0
        # 产生服从复高斯分布的单瞻相干斑噪声
        noise = np.random.normal(0, 1/np.sqrt(2), (H, W)) + 1j * np.random.normal(0, 1/np.sqrt(2), (H, W))
        S = np.sqrt(clean_ref + 0.1) * noise
        np.save(slc_image_path, S)
    
    # 2. 调用核心算法构建相似像素库
    print("-> 正在分析上下文相关性并构建相似像素库 (全向量化计算)...")
    pixel_bank = construct_similar_pixel_bank(S, M=M, N=N)
    print(f"-> 相似像素库构建成功。数据体体积形状: {pixel_bank.shape}")
    
    # 3. 实例化 DataLoader（全图作为一个 Batch 训练）
    dataset = SimilarPixelBankDataset(pixel_bank, length=1)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # 4. 配置网络、优化器及论文指定的学习率衰减策略
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Context2ContextNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 论文设定：在第 200 和 300 个 epoch 分别衰减 50% 学习率
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 300], gamma=0.5)
    
    # L=1 单瞻雷达图像对数域相干斑均值补偿常数：约 -0.557
    COMPENSATION = -0.557
    criterion = nn.MSELoss()
    
    model.train()
    print("-> 开始 Context2Context 零样本模型自训练迭代...")
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        for y1, y2 in dataloader:
            y1, y2 = y1.to(device), y2.to(device)
            
            # 正向推断
            output = model(y1)
            
            # 基于噪声期望的平移补偿：Target = y2 - (-0.557)
            target = y2 - COMPENSATION
            
            # 计算 L2（MSE）损失函数值
            loss = criterion(output, target)
            
            # 梯度更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        scheduler.step()
        
        # 每隔 20 轮打印输出
        if epoch % 20 == 0 or epoch == 1:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch:03d}/{num_epochs}] | 损失 Loss: {epoch_loss:.5f} | 学习率 LR: {current_lr:.6f}")
            
    # 存储最优权重
    torch.save(model.state_dict(), save_model_path)
    print(f"-> 训练完毕！网络权重文件已成功保存至 {save_model_path}")

if __name__ == '__main__':
    train_context2context(slc_image_path='data/sample_slc_image.npy', num_epochs=500)