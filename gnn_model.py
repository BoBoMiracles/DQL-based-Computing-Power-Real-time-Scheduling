import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data

class GNNPolicy(nn.Module):
    def __init__(self, node_feat_dim=5, hidden_dim=64, action_space_size=10):
        super().__init__()
        # 节点特征编码 - 输入特征维度根据模拟器调整为5
        self.node_enc = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # 图卷积网络 - 移除边特征相关参数
        self.conv1 = pyg_nn.GATConv(hidden_dim, hidden_dim)
        self.conv2 = pyg_nn.GATConv(hidden_dim, hidden_dim)
        
        # 全局注意力池化
        self.pool = pyg_nn.GlobalAttention(gate_nn=nn.Linear(hidden_dim, 1))
        
        # Q值预测头 - 输出动作空间大小的Q值
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_space_size)  # 输出动作空间大小的Q值
        )

    def forward(self, data):
        # 1. 节点编码
        x = self.node_enc(data.x)
        
        # 2. 图卷积 - 移除边特征输入
        x = F.relu(self.conv1(x, data.edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv2(x, data.edge_index))
        
        # 3. 全局池化
        global_feat = self.pool(x, data.batch)
        
        # 4. 预测每个动作的Q值
        return self.q_head(global_feat)