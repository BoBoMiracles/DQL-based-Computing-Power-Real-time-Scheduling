import numpy as np
import random
import torch
from collections import deque
from torch import optim
from gnn_model import GNNPolicy
from torch_geometric.data import Data, Batch
from utils import StateTransformer
import torch.nn.functional as F

class GNNAgent:
    """基于GNN的DQN智能体"""
    def __init__(self, env, device='cuda'):
        self.env = env
        self.device = device
        self.policy_net = GNNPolicy().to(device)
        self.target_net = GNNPolicy().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        
        # 经验回放
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        
        # 状态转换器
        self.transformer = StateTransformer(env)
    
    def get_action(self, state, epsilon=0.1):
        """ε-greedy策略"""
        if random.random() < epsilon:
            return random.choice(self.env.get_valid_actions())
        
        with torch.no_grad():
            graph_data = self.transformer.state_to_graph(state).to(self.device)
            # 确保batch维度正确
            graph_data.batch = torch.zeros(graph_data.num_nodes, 
                                        dtype=torch.long, 
                                        device=self.device)
            
            q_values = self.policy_net(graph_data)
            
            # 过滤非法动作
            valid_mask = self._get_valid_mask(state)
            q_values[~valid_mask] = -float('inf')
            return torch.argmax(q_values).item()
    
    def _get_valid_mask(self, state):
        """生成合法动作掩码"""
        mask = torch.zeros(len(self.env.nodes['base_stations']) + 1)  # +1 for cloud
        for i, bs in enumerate(self.env.nodes['base_stations'].values()):
            if bs['type'] == 'main' and bs['compute'] > 0:
                mask[i] = 1
        mask[-1] = 1  # 云端始终有效
        return mask.bool().to(self.device)
    
    def update_model(self):
        """更新策略网络"""
        if len(self.memory) < self.batch_size:
            return
        
        # 采样批次
        batch = random.sample(self.memory, self.batch_size)
        
        # 转换数据
        states = [self.transformer.state_to_graph(s) for s, _, _, _ in batch]
        batch_data = Batch.from_data_list(states).to(self.device)
        
        # 计算当前Q值
        current_q = self.policy_net(batch_data)
        
        # 计算目标Q值
        next_q = self.target_net(batch_data).max(1)[0]
        target_q = torch.tensor([r + (0.99 * next_q[i]) if not done else r 
                               for i, (_, _, r, done) in enumerate(batch)])
        
        # 计算损失
        loss = F.mse_loss(current_q, target_q)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
    
    def update_target_net(self):
        """更新目标网络"""
        self.target_net.load_state_dict(self.policy_net.state_dict())