import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data

class GNNPolicy(nn.Module):
    def __init__(self, node_feat_dim=6, edge_feat_dim=2, hidden_dim=64):
        super().__init__()
        
        # 节点特征编码
        self.node_enc = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # 边特征编码
        self.edge_enc = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 图注意力网络
        self.conv1 = pyg_nn.GATConv(hidden_dim, hidden_dim, edge_dim=hidden_dim)
        self.conv2 = pyg_nn.GATConv(hidden_dim, hidden_dim, edge_dim=hidden_dim)
        
        # 全局注意力池化
        self.pool = pyg_nn.GlobalAttention(gate_nn=nn.Linear(hidden_dim, 1))
        
        # Q值预测头
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        # 1. 节点编码
        x = self.node_enc(data.x)  # [num_nodes, hidden_dim]
        
        # 2. 边编码
        edge_attr = self.edge_enc(data.edge_attr)  # [num_edges, hidden_dim]
        
        # 3. 第一层图卷积
        x = F.relu(self.conv1(x, data.edge_index, edge_attr=edge_attr))  # [num_nodes, hidden_dim]
        
        # 4. Dropout
        x = F.dropout(x, p=0.2, training=self.training)
        
        # 5. 第二层图卷积
        x = F.relu(self.conv2(x, data.edge_index, edge_attr=edge_attr))  # [num_nodes, hidden_dim]
        
        # 6. 全局池化
        global_feat = self.pool(x, data.batch)  # [batch_size, hidden_dim]
        
        # 7. 复制全局特征到每个节点
        global_feat_per_node = global_feat[data.batch]  # [num_nodes, hidden_dim]
        
        # 8. 拼接局部和全局特征
        x = torch.cat([x, global_feat_per_node], dim=1)  # [num_nodes, hidden_dim*2]
        
        # 9. 预测Q值
        return self.q_head(x).squeeze()  # [num_nodes]