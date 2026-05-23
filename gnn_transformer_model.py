import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data, Batch
import math

class PositionalEncoding(nn.Module):
    """
    实现 Transformer 的位置编码
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [T, B, F] (序列长度, 批次大小, 特征维度)
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class GNNTransformerPolicy(nn.Module):
    """结合GNN和Transformer的DQN策略网络"""
    def __init__(self, sequence_length, node_feat_dim=8, gnn_hidden_dim=64, action_space_size=10, trans_hidden_dim=128, req_feat_dim=5, nhead=4, num_encoder_layers=2):
        super().__init__()
        self.gnn_hidden_dim = gnn_hidden_dim
        self.trans_hidden_dim = trans_hidden_dim
        self.sequence_length = sequence_length
        self.request_feat_dim = req_feat_dim

        # GNN/Pooling 部分 (保持不变)
        self.node_enc = nn.Sequential(
            nn.Linear(node_feat_dim, gnn_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(gnn_hidden_dim)
        )
        self.conv1 = pyg_nn.GATConv(gnn_hidden_dim, gnn_hidden_dim)
        self.conv2 = pyg_nn.GATConv(gnn_hidden_dim, gnn_hidden_dim)
        self.pool = pyg_nn.GlobalAttention(gate_nn=nn.Linear(gnn_hidden_dim, 1))
        
        # -------------------
        # LSTM -> Transformer 替换
        # -------------------
        
        # 1. 线性层：将 GNN+Req 特征 映射到 Transformer 的隐藏维度
        self.input_proj = nn.Linear(gnn_hidden_dim + req_feat_dim, trans_hidden_dim)
        
        # 2. 位置编码
        self.pos_encoder = PositionalEncoding(trans_hidden_dim, dropout=0.1, max_len=sequence_length + 5)
        
        # 3. Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=trans_hidden_dim, 
            nhead=nhead, 
            dim_feedforward=trans_hidden_dim * 2,
            dropout=0.1,
            batch_first=False # 我们将使用 [T, B, F] 格式
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layers, 
            num_layers=num_encoder_layers
        )
        
        # Q值预测头
        # (输入维度从 lstm_hidden_dim 变为 trans_hidden_dim)
        self.q_head = nn.Sequential(
            nn.Linear(trans_hidden_dim, gnn_hidden_dim),
            nn.ReLU(),
            nn.Linear(gnn_hidden_dim, action_space_size)
        )

    def forward(self, data, batch_size_override=None):
        """
        注意：Transformer 不返回 hidden_state
        """
        # 1. GNN/Pooling
        x = self.node_enc(data.x)
        x = F.relu(self.conv1(x, data.edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv2(x, data.edge_index))
        global_feat_flat = self.pool(x, data.batch) # [Total_Timesteps, gnn_hidden_dim]
        
        # 2. 拼接请求特征
        # data.req_feat 维度应为 [Total_Timesteps, req_feat_dim]
        combined_feat_flat = torch.cat([global_feat_flat, data.req_feat], dim=1)

        # 3. 确定批次大小 B
        total_timesteps = global_feat_flat.size(0)
        
        if batch_size_override is not None:
            batch_size = batch_size_override
        else:
            if total_timesteps % self.sequence_length != 0:
                 raise ValueError(f"Total timesteps ({total_timesteps}) is not divisible by sequence_length ({self.sequence_length}). Batching error.")
            batch_size = total_timesteps // self.sequence_length

        # 4. 投影到 Transformer 维度
        # [Total_Timesteps, trans_hidden_dim]
        trans_input_flat = self.input_proj(combined_feat_flat)
        
        # 5. Batch 中的图按 [B, T] 展平，Transformer 需要 [T, B, F]
        trans_input = trans_input_flat.view(
            batch_size, self.sequence_length, self.trans_hidden_dim
        ).transpose(0, 1).contiguous()
        
        # 6. 位置编码
        trans_input = self.pos_encoder(trans_input)
        
        # 7. Transformer 前向传播
        # Transformer (batch_first=False) 期望 [T, B, F]
        # 注意：我们不需要（也不能）传递 hidden_state
        trans_out = self.transformer_encoder(trans_input) 
        # trans_out: [T, B, trans_hidden_dim]
        
        # 8. Q值预测，转回 [B, T, F] 后展平，保持与 agent 的 last_indices 一致
        trans_out_flat = trans_out.transpose(0, 1).contiguous().view(-1, self.trans_hidden_dim) 
        
        # Q值输出: [T * B, action_space_size]
        q_values_flat = self.q_head(trans_out_flat)
        
        # Transformer 不返回 hidden_state，我们返回 None 以匹配（修改后的）Agent 的期望
        return q_values_flat, None
