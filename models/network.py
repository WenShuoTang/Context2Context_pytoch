import torch
import torch.nn as nn

class Context2ContextNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, hidden_channels=64):
        super(Context2ContextNet, self).__init__()
        
        # 1) 主干网络 (Backbone)：5 层 64 通道 3x3 卷积，后接 BN 和 PReLU
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.PReLU(hidden_channels),
            
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.PReLU(hidden_channels),
            
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.PReLU(hidden_channels),
            
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.PReLU(hidden_channels),
            
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.PReLU(hidden_channels)
        )
        
        # 2) 最终输出层：1x1 卷积，用于映射回单通道（恢复对数域滤波图像）
        self.final_conv = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        out = self.backbone(x)
        out = self.final_conv(out)
        return out