# model.py
import torch
from torch import nn

class Direct_Conv3D_GRU(nn.Module):
    def __init__(self, in_channels, 
                 forecast_hours,
                 coord_dim, 
                 hist_input_size,
                 hidden_size, 
                 coord_feat_dim,
                 kt):
        super().__init__()
        self.forecast_hours = forecast_hours
        self.kt = kt

        # 历史观测编码器
        self.gru = nn.GRU(
            input_size=hist_input_size,
            hidden_size=hidden_size,
            # num_layers=2, 
            # dropout=0.4,  
            batch_first=True,
            bidirectional=True
        )

        # NWP 图像 CNN（通道数可配置）
        self.cnn3d = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=(kt,3,3), padding=(kt//2,1,1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=(kt,3,3), padding=(kt//2,1,1)),
            nn.ReLU(inplace=True),
        )
        self.cnn_out_dim = 64
        # self.cnn_fc_drop = nn.Dropout(p=0.4)
        # 坐标映射
        self.fc_coord = nn.Linear(coord_dim, coord_feat_dim)
        
        # self.coord_drop = nn.Dropout(p=0.3)
        # self.gru_drop = nn.Dropout(p=0.3)

        # 融合后预测 F 步
        fuse_dim = self.cnn_out_dim + 2*hidden_size + coord_feat_dim
        self.head = nn.Linear(fuse_dim, 1)

    def forward(self, coords, hist_obs, nwp_seq):
        """
        coords   : [B, coord_dim]
        hist_obs : [B, H_hist] 
        nwp_seq  : [B, F, C, H, W]
        """
        # ---- 兼容输入形状 ----
        # hist_obs -> [B, H_hist, 1]
        # assert coords.size(-1) == self.fc_coord.in_features, \
        # f"coords last dim = {coords.size(-1)}, but fc_coord.in_features = {self.fc_coord.in_features}"

        # print("DEBUG: coord_dim(in_features) =", self.fc_coord.in_features)
        # print("DEBUG: coords.shape =", tuple(coords.shape))

        B, F, C, H, W = nwp_seq.shape
        out_enc, _ = self.gru(hist_obs)
        enc_feat = out_enc[:, -1, :]  # 取最后时间步 [B, 2*hidden]

        # ---- NWP 每步跑 CNN，然后拉平拼接 ----
        x = nwp_seq.permute(0, 2, 1, 3, 4).contiguous()  # [B, C, F, H, W]
        x = self.cnn3d(x)                                # [B, 64, F, H, W]
        # 只对空间做全局池化（时间 F 保留）
        feat = x.mean(dim=(-1, -2))                      # [B, 64, F]
        feat = feat.permute(0, 2, 1).contiguous()        # [B, F, 64]

        # ---- 坐标特征 ----
        coord_feat = self.fc_coord(coords)       # [B, coord_feat_dim]

        # ---- 融合 & 预测 ----
        enc_broadcast = enc_feat.unsqueeze(1).expand(-1, F, -1)    # [B, F, 2*hidden]
        coord_broadcast = coord_feat.unsqueeze(1).expand(-1, F, -1)# [B, F, coord_feat_dim]
        fused = torch.cat([feat, enc_broadcast, coord_broadcast], dim=-1)  # [B, F, fuse_dim]

        # ---- 逐时预测 ----
        out = self.head(fused).squeeze(-1)                         # [B, F]
        return out
