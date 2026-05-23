import numpy as np

def compute_ccm(S):
    """
    计算复数图像所有像素的上下文协方差矩阵 (CCM)
    输入 S: 单瞻复数 SAR 图像 (SLC)，形状为 (H, W)
    输出: 形状为 (H, W, 3, 3) 的复数 CCM 矩阵
    """
    H, W = S.shape
    S_pad = np.pad(S, pad_width=1, mode='reflect')
    CCM = np.zeros((H, W, 3, 3), dtype=np.complex128)
    
    # 辅助函数：根据行列偏移获取滑动视窗
    def get_shifted(dr, dc):
        return S_pad[1+dr : 1+dr+H, 1+dc : 1+dc+W]
    
    # 论文中定义的 4 个方向散射矢量 k1 - k4
    k1 = [get_shifted(0, -1),  get_shifted(0, 0), get_shifted(0, 1)]  # 水平
    k2 = [get_shifted(-1, -1), get_shifted(0, 0), get_shifted(1, 1)]  # 主对角线
    k3 = [get_shifted(-1, 0),  get_shifted(0, 0), get_shifted(1, 0)]  # 垂直
    k4 = [get_shifted(-1, 1),  get_shifted(0, 0), get_shifted(1, -1)] # 副对角线
    
    # 累加外积外加平均：C_CCM = 1/4 * \sum (k_i * k_i^\dagger)
    for k in [k1, k2, k3, k4]:
        for i in range(3):
            for j in range(3):
                CCM[:, :, i, j] += k[i] * np.conj(k[j])
    CCM /= 4.0
    return CCM

def construct_similar_pixel_bank(S, M=9, N=20):
    """
    通过基于 CCM 的相似度检验，构建形状为 (H, W, N) 的相似像素库
    M: 搜索窗尺寸 (默认 9)
    N: 每个空间位置保留的相似像素数 (默认 20)
    """
    H, W = S.shape
    CCM = compute_ccm(S)
    
    # 计算每个像素自身 CCM 的行列式对数
    det_CCM = np.real(np.linalg.det(CCM))
    ln_det_CCM = np.log(np.clip(det_CCM, a_min=1e-12, a_max=None))
    
    half_M = M // 2
    # 边界填充
    CCM_pad = np.pad(CCM, ((half_M, half_M), (half_M, half_M), (0, 0), (0, 0)), mode='reflect')
    S_pad = np.pad(S, pad_width=half_M, mode='reflect')
    
    ln_Q_all = []
    S_neighbors = []
    
    # 在 M x M 窗口内遍历所有邻域偏移
    for dr in range(-half_M, half_M + 1):
        for dc in range(-half_M, half_M + 1):
            CCM_shifted = CCM_pad[half_M + dr : half_M + dr + H, half_M + dc : half_M + dc + W]
            S_shifted = S_pad[half_M + dr : half_M + dr + H, half_M + dc : half_M + dc + W]
            
            # 复 Wishart 二元统计检验核心公式
            det_sum = np.real(np.linalg.det(CCM + CCM_shifted))
            ln_det_sum = np.log(np.clip(det_sum, a_min=1e-12, a_max=None))
            
            det_Y = np.real(np.linalg.det(CCM_shifted))
            ln_det_Y = np.log(np.clip(det_Y, a_min=1e-12, a_max=None))
            
            # 维度 q=3 时, 常数项 2*q*ln(2) = 6*ln(2)
            ln_Q = 6.0 * np.log(2.0) + ln_det_CCM + ln_det_Y - 2.0 * ln_det_sum
            
            ln_Q_all.append(ln_Q)
            S_neighbors.append(S_shifted)
            
    ln_Q_all = np.stack(ln_Q_all, axis=0)      # (M^2, H, W)
    S_neighbors = np.stack(S_neighbors, axis=0)  # (M^2, H, W)
    
    # 对邻域相似度按降序排序（ln_Q 越接近 0 越相似）
    sorted_indices = np.argsort(ln_Q_all, axis=0)
    top_N_indices = sorted_indices[-N:, :, :]  # 取最后（最大）的 N 个
    
    # 空间格网高级索引，抽取复数值
    grid_h, grid_w = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    pixel_bank = np.zeros((H, W, N), dtype=np.complex128)
    for n in range(N):
        idx_n = top_N_indices[n, :, :]
        pixel_bank[:, :, n] = S_neighbors[idx_n, grid_h, grid_w]
        
    return pixel_bank