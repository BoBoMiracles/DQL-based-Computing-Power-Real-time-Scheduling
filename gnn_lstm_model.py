import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data, Batch

class GNNLSTMPolicy(nn.Module):
    """结合GNN和LSTM的DQN策略网络"""
    def __init__(self, sequence_length, node_feat_dim=8, hidden_dim=64, action_space_size=10, lstm_hidden_dim=128, req_feat_dim=5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.sequence_length = sequence_length
        self.num_layers = 1 # LSTM层数固定为1
        self.request_feat_dim = req_feat_dim

        # GNN/Pooling 部分 (保持您的设计)
        self.node_enc = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        self.conv1 = pyg_nn.GATConv(hidden_dim, hidden_dim)
        self.conv2 = pyg_nn.GATConv(hidden_dim, hidden_dim)
        self.pool = pyg_nn.GlobalAttention(gate_nn=nn.Linear(hidden_dim, 1))
        
        # 核心修复: LSTM模块
        # LSTM的输入维度 = GNN输出维度 + 请求特征维度
        self.lstm = nn.LSTM(
            input_size=hidden_dim + req_feat_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=self.num_layers,
            batch_first=True # [B, T, F] 格式
        )
        
        # Q值预测头
        self.q_head = nn.Sequential(
            nn.Linear(lstm_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_space_size)
        )

    def forward(self, data, batch_size_override=None, hidden_state=None):
        # 1. GNN/Pooling
        x = self.node_enc(data.x)
        x = F.relu(self.conv1(x, data.edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv2(x, data.edge_index))
        global_feat_flat = self.pool(x, data.batch) # [Total_Timesteps, hidden_dim]
        
        # 2. 拼接请求特征
        # data.req_feat 维度应为 [Total_Timesteps, req_feat_dim]
        combined_feat_flat = torch.cat([global_feat_flat, data.req_feat], dim=1)

        # 3. 确定批次大小并重塑序列
        # Total_Timesteps = B * T  
        # 从 Batch 对象中获取实际的图数量，这才是 Total_Timesteps
        total_timesteps = global_feat_flat.size(0)

        # 确定批次大小 B
        if batch_size_override is not None:
            batch_size = batch_size_override
        else:
            # 推理时，通常是单个序列 (B=1, T=sequence_length)
            # 或者训练时 B = Total_Timesteps / sequence_length
            if total_timesteps % self.sequence_length != 0:
                 raise ValueError(f"Total timesteps ({total_timesteps}) is not divisible by sequence_length ({self.sequence_length}). Batching error.")
            batch_size = total_timesteps // self.sequence_length

        # 重塑为 [B, T, F] 格式
        # 使用 combined_feat_flat 和新的特征维度
        lstm_input = combined_feat_flat.view(batch_size, self.sequence_length, self.hidden_dim + self.request_feat_dim)
        
        # 3. LSTM前向传播
        if hidden_state is None:
             # 如果没有提供隐藏状态，则自动初始化零状态
             h0 = torch.zeros(self.num_layers, batch_size, self.lstm_hidden_dim).to(lstm_input.device)
             c0 = torch.zeros(self.num_layers, batch_size, self.lstm_hidden_dim).to(lstm_input.device)
             hidden_state = (h0, c0)

        lstm_out, new_hidden_state = self.lstm(lstm_input, hidden_state) 
        # lstm_out: [batch_size, sequence_length, lstm_hidden_dim]
        
        # 4. Q值预测
        # 针对每个时间步的输出进行Q值预测，然后展平
        # 展平为 [B * T, lstm_hidden_dim]
        lstm_out_flat = lstm_out.reshape(-1, self.lstm_hidden_dim) 
        
        # Q值输出: [B * T, action_space_size]
        q_values_flat = self.q_head(lstm_out_flat)
        
        return q_values_flat, new_hidden_state
