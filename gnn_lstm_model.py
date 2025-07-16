import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data

class GNNLSTMPolicy(nn.Module):
    """结合GNN和LSTM的DQN策略网络"""
    def __init__(self, node_feat_dim=5, hidden_dim=64, action_space_size=10, lstm_hidden_dim=128, history_len=10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.history_len = history_len
        
        # 节点特征编码
        self.node_enc = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # 图卷积网络
        self.conv1 = pyg_nn.GATConv(hidden_dim, hidden_dim)
        self.conv2 = pyg_nn.GATConv(hidden_dim, hidden_dim)
        
        # 全局注意力池化
        self.pool = pyg_nn.GlobalAttention(gate_nn=nn.Linear(hidden_dim, 1))
        
        # LSTM模块
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Q值预测头
        self.q_head = nn.Sequential(
            nn.Linear(lstm_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_space_size)
        )

    def forward(self, data, hidden_state=None):
        # 1. 节点编码
        x = self.node_enc(data.x)
        
        # 2. 图卷积
        x = F.relu(self.conv1(x, data.edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv2(x, data.edge_index))
        
        # 3. 全局池化
        global_feat = self.pool(x, data.batch)  # [batch_size, hidden_dim]
        
        # 4. LSTM处理历史特征
        # 将全局特征重塑为序列形式 [batch_size, seq_len, hidden_dim]
        # 这里我们假设每个batch只有一个样本，所以seq_len=1
        global_feat = global_feat.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # 初始化LSTM隐藏状态
        if hidden_state is None:
            h0 = torch.zeros(1, global_feat.size(0), self.lstm_hidden_dim).to(global_feat.device)
            c0 = torch.zeros(1, global_feat.size(0), self.lstm_hidden_dim).to(global_feat.device)
            hidden_state = (h0, c0)
        
        # LSTM前向传播
        lstm_out, hidden_state = self.lstm(global_feat, hidden_state)
        lstm_feat = lstm_out[:, -1, :]  # 取最后一个时间步的输出 [batch_size, lstm_hidden_dim]
        
        # 5. 预测动作Q值
        q_values = self.q_head(lstm_feat)
        
        return q_values, hidden_state