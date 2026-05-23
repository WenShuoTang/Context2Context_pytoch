import os
import torch
import numpy as np
from models.network import Context2ContextNet

def inference_context2context(slc_image_path, model_path='checkpoint.pth', output_npy_path='results/filtered_image.npy'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 载入原始斑点图像并求得对数强度域数据
    S = np.load(slc_image_path)
    if S.ndim == 3:
        S = S[:, :, 0]
        
    I_noisy = np.abs(S) ** 2
    y_noisy = np.log(I_noisy + 1e-8)
    
    y_tensor = torch.from_numpy(y_noisy).float().unsqueeze(0).unsqueeze(0).to(device)
    
    # 2. 加载训练好的模型参数
    model = Context2ContextNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 3. 评估网络
    with torch.no_grad():
        y_filtered_tensor = model(y_tensor)
        
    y_filtered = y_filtered_tensor.squeeze().cpu().numpy()
    # 通过指数变换还原到线性强度域
    I_filtered = np.exp(y_filtered)
    
    # 4. 直接保存处理后的强度图 npy 文件，不进行任何绘制与打印
    output_dir = os.path.dirname(output_npy_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    np.save(output_npy_path, I_filtered)

if __name__ == '__main__':
    inference_context2context(slc_image_path='data/sample_slc_image.npy')