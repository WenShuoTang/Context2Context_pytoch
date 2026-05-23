import os
import warnings
import numpy as np
import matplotlib.pyplot as plt

# 禁用所有可能的警告信息
warnings.filterwarnings('ignore')

def process_and_show_sar(npy_path, colormap='gray'):
    # 1. 隐蔽加载
    S = np.load(npy_path)
    if S.ndim == 3:
        S = S[:, :, 0]
        
    # 2. 核心数学变换
    log_intensity = np.log10(np.abs(S) ** 2 + 1e-6)
    
    # 3. 图像生成设置
    plt.figure(figsize=(10, 10), dpi=300)
    
    # 使用百分比拉伸并强制 'nearest' 杜绝模糊
    vmin, vmax = np.percentile(log_intensity, [1, 99])
    plt.imshow(log_intensity, cmap=colormap, vmin=vmin, vmax=vmax, interpolation='nearest')
    
    # 4. 纯净输出：移除坐标轴、边框、标题、颜色条
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    
    # 5. 自动创建结果目录并静默保存/展示
    os.makedirs('visualize_results', exist_ok=True)
    plt.savefig('visualize_results/my_sar_view.png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()

if __name__ == '__main__':
    # 请确保路径正确，运行后直接出图
    process_and_show_sar('data/sample_slc_image.npy')