import os
import scipy.io as sio
import numpy as np

def convert_mat_to_slc_npy(mat_path, output_npy_path='sample_slc_image.npy', crop_size=None):
    """
    将 MATLAB 的 .mat 格式复数矩阵转换为 Context2Context 项目所需的 .npy 复数矩阵。
    
    参数:
    mat_path (str): 输入的 .mat 文件路径
    output_npy_path (str): 输出的 .npy 文件路径，默认命名为 'sample_slc_image.npy'
    crop_size (tuple): 裁剪大小 (H, W)。如果原图特别大(如几千像素)，建议裁剪成 (512, 512) 或 (256, 256) 
                      用于快速训练。如果是 None 则保留全图。
    """
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"未找到指定的 .mat 文件: {mat_path}")
        
    print(f"-> 正在加载 MATLAB 文件: {mat_path}")
    # 1. 载入 .mat 文件
    mat_data = sio.loadmat(mat_path)
    
    # 2. 自动寻找存储图像矩阵的变量名（排除其中的系统默认键）
    data_key = None
    for key in mat_data.keys():
        if not key.startswith('__'):  # 排除 __header__, __version__, __globals__
            # 检查该变量是否为二维或三维矩阵
            if isinstance(mat_data[key], np.ndarray) and mat_data[key].ndim in [2, 3]:
                data_key = key
                break
                
    if data_key is None:
        print("❌ 未能在 .mat 文件中自动识别到矩阵变量！文件中包含的键有:")
        print(list(mat_data.keys()))
        print("请手动指定变量名，例如修改代码中的 data_key = '你的变量名'")
        return
    
    print(f"-> 成功识别到矩阵变量名: '{data_key}'")
    S = mat_data[data_key]
    
    # 3. 检查数据是否为复数 (Complex) 类型
    if not np.iscomplexobj(S):
        print("⚠️ 警告: 读取的数据不是复数矩阵(Complex)。")
        print("提示: Context2Context 算法需要包含实部与虚部的单瞻复数 SLC 数据。")
        print("如果您的 .mat 中仅存了强度(Intensity)或幅度(Amplitude)，计算上下文协方差矩阵(CCM)时可能退化。")
        # 强制转换为复数类型以防报错
        S = S.astype(np.complex128)
    else:
        print("-> 验证成功：数据为合法的复数矩阵。")
        
    # 如果数据是 3 维的（如有些极化 SAR 包含 [H, W, 3] 或者是多波段）
    # 论文针对的是单通道单瞻标量 SLC，这里默认取第一层通道 [H, W]
    if S.ndim == 3:
        print(f"-> 检测到数据为3维矩阵 {S.shape}，默认提取第一通道 [:, :, 0]...")
        S = S[:, :, 0]
        
    # 4. 可选：裁剪图像（因为高分辨率雷达图可能极大，零样本自训练单大图太耗时）
    if crop_size is not None:
        h, w = S.shape
        th, tw = crop_size
        if h > th and w > tw:
            start_h = (h - th) // 2
            start_w = (w - tw) // 2
            S = S[start_h:start_h+th, start_w:start_w+tw]
            print(f"-> 已将图像中心裁剪至指定尺寸: {S.shape}")
        else:
            print(f"-> 原图尺寸 {S.shape} 小于裁剪尺寸 {crop_size}，保持原样。")
            
    # 5. 保存为 .npy 格式
    np.save(output_npy_path, S)
    print(f"====== 转换完成 ======")
    print(f"▶️ 原始矩阵形状: {mat_data[data_key].shape}")
    print(f"▶️ 最终保存路径: {output_npy_path}")
    print(f"▶️ 最终矩阵形状: {S.shape}")
    print(f"▶️ 数据存储类型: {S.dtype}")
    print(f"现在您可以直接运行项目中的 `train.py` 进行去噪了。")

if __name__ == '__main__':
    # ================= 使用示例 =================
    # 替换成你本地的 .mat 文件路径
    YOUR_MAT_FILE = "test1/GF3_KAS_SL_028685_E139.7_N35.5_20220120_L1A_HH_L10000000001_07168_01070.mat" 
    
    # 建议初次测试时裁剪为 256x256 或 512x512，这样模型的零样本自训练可以在几分钟内完成。
    # 如果想要去噪整张大图，请设置 crop_size=None
    convert_mat_to_slc_npy(
        mat_path=YOUR_MAT_FILE, 
        output_npy_path='sample_slc_image.npy', # 保持该文件名，前文的 train.py 会自动读取
        crop_size=(256, 256) 
    )