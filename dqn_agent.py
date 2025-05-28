import numpy as np
import random
import torch
import torch.nn.functional as F
from collections import deque
from torch import optim
from gnn_model import GNNPolicy
from torch_geometric.data import Batch
from utils import StateTransformer

class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class GNNAgent:
    """基于GNN的DQN智能体 - 完整修复版本"""
    def __init__(self, env, device='cuda'):
        self.env = env
        self.device = device
        self.policy_net = GNNPolicy().to(device)
        self.target_net = GNNPolicy().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.memory = ReplayBuffer(capacity=10000)
        self.batch_size = 32
        self.transformer = StateTransformer(env)
        self.steps_done = 0
    
    def get_action(self, state, epsilon=0.1):
        """ε-greedy策略 - 使用节点Q值最大值"""
        if random.random() < epsilon:
            return random.choice(self.env.get_valid_actions())
        
        with torch.no_grad():
            graph_data = self.transformer.state_to_graph(state).to(self.device)
            q_values = self.policy_net(graph_data)
            
            # 找到具有最大Q值的节点
            max_idx = torch.argmax(q_values).item()
            
            # 根据节点索引确定动作类型
            num_rooms = len(self.env.nodes['rooms'])
            
            if max_idx == 0:  # 云端节点
                return {'target_type': 'cloud', 'target_id': 'cloud'}
            elif max_idx < 1 + num_rooms:  # 机房节点
                # 机房不直接处理请求，选择默认动作
                return random.choice(self.env.get_valid_actions())
            else:  # 基站节点
                bs_idx = max_idx - 1 - num_rooms
                bs_keys = list(self.env.nodes['base_stations'].keys())
                if bs_idx < len(bs_keys):
                    bs_id = bs_keys[bs_idx]
                    return {'target_type': 'base_stations', 'target_id': bs_id}
        
        # 回退到随机动作
        return random.choice(self.env.get_valid_actions())
    
    def remember(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.push(state, action, reward, next_state, done)
    
    def update_model(self):
        """更新策略网络 - 使用图级Q值"""
        batch = self.memory.sample(self.batch_size)
        if batch is None:
            return
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 转换状态为图数据
        state_graphs = [self.transformer.state_to_graph(s) for s in states]
        next_state_graphs = [self.transformer.state_to_graph(s) for s in next_states]
        
        # 创建批次图
        state_batch = Batch.from_data_list(state_graphs).to(self.device)
        next_state_batch = Batch.from_data_list(next_state_graphs).to(self.device)
        
        # 计算当前Q值
        current_q = self.policy_net(state_batch)
        
        # 计算目标Q值
        with torch.no_grad():
            next_q = self.target_net(next_state_batch)
            
            # 按图分组取最大值
            next_q_max = []
            ptr = next_state_batch.ptr
            for i in range(len(ptr) - 1):
                start_idx = ptr[i]
                end_idx = ptr[i+1]
                graph_q = next_q[start_idx:end_idx]
                next_q_max.append(graph_q.max())
            
            next_q_max = torch.stack(next_q_max)  # 形状为 [batch_size]
            
            # 转换为张量
            rewards_tensor = torch.tensor(rewards, device=self.device, dtype=torch.float)
            dones_tensor = torch.tensor(dones, device=self.device, dtype=torch.float)
            
            # 计算目标Q值
            target_q = rewards_tensor + (1 - dones_tensor) * 0.99 * next_q_max
        
        # 对当前Q值也按图分组取最大值
        current_q_max = []
        ptr = state_batch.ptr
        for i in range(len(ptr) - 1):
            start_idx = ptr[i]
            end_idx = ptr[i+1]
            graph_q = current_q[start_idx:end_idx]
            current_q_max.append(graph_q.max())
        
        current_q_max = torch.stack(current_q_max)  # 形状为 [batch_size]
        
        # 计算损失 - 现在两者形状都是 [batch_size]
        loss = F.mse_loss(current_q_max, target_q)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_net(self):
        """更新目标网络"""
        self.target_net.load_state_dict(self.policy_net.state_dict())