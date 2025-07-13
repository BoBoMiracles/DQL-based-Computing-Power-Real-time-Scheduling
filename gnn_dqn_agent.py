import numpy as np
import random
import torch
import torch.nn.functional as F
from collections import deque
from torch import optim
from gnn_model import GNNPolicy
from torch_geometric.data import Batch
from torch_geometric.data import Data


class StateTransformer:
    """环境状态到图数据的转换器 - 适配新模拟器状态格式"""
    def __init__(self, env):
        self.env = env
    
    def state_to_graph(self, state):
        """将环境状态转换为图数据"""
        # 直接使用状态字典中的特征
        return Data(
            x=state['x'],
            edge_index=state['edge_index'],
            batch=torch.zeros(state['x'].size(0), dtype=torch.long)  # 所有节点在一个图中
        )

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
    """基于GNN的DQN智能体 - 适配新模拟器"""
    def __init__(self, env, device='cuda'):
        self.env = env
        self.device = device
        
        # 获取动作空间大小（机房数量+1）
        self.action_space_size = len(env.nodes['rooms']) + 1
        self.policy_net = GNNPolicy(action_space_size=self.action_space_size).to(device)
        self.target_net = GNNPolicy(action_space_size=self.action_space_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.memory = ReplayBuffer(capacity=10000)
        self.batch_size = 32
        self.transformer = StateTransformer(env)
        self.steps_done = 0
        
        # 构建动作列表（云端+所有机房）
        self.action_list = ['cloud'] + list(env.nodes['rooms'].keys())
    
    def get_action(self, state, epsilon=0.1):
        """ε-greedy策略 - 使用动作Q值"""
        if random.random() < epsilon:
            # 随机选择合法动作
            valid_mask = state['valid_actions']
            valid_indices = valid_mask.nonzero().squeeze().tolist()
            if not valid_indices:
                return 'cloud'  # 默认选择云端
            action_idx = random.choice(valid_indices)
            return self.action_list[action_idx]
        
        with torch.no_grad():
            graph_data = self.transformer.state_to_graph(state).to(self.device)
            # 增加批次维度
            graph_data.batch = torch.zeros(graph_data.num_nodes, dtype=torch.long, device=self.device)
            q_values = self.policy_net(graph_data).squeeze(0)  # [action_space_size]
            
            # 应用合法动作掩码
            valid_mask = state['valid_actions'].to(self.device)
            masked_q = q_values.clone()
            masked_q[~valid_mask] = -float('inf')  # 屏蔽非法动作
            
            # 选择Q值最大的动作
            action_idx = torch.argmax(masked_q).item()
            return self.action_list[action_idx]
    
    def remember(self, state, action, reward, next_state, done):
        """存储经验"""
        # 将动作转换为索引
        action_idx = self.action_list.index(action)
        self.memory.push(state, action_idx, reward, next_state, done)
    
    def update_model(self):
        """更新策略网络"""
        batch = self.memory.sample(self.batch_size)
        if batch is None:
            return None
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 转换状态为图数据
        state_graphs = [self.transformer.state_to_graph(s) for s in states]
        next_state_graphs = [self.transformer.state_to_graph(s) for s in next_states]
        
        # 创建批次图
        state_batch = Batch.from_data_list(state_graphs).to(self.device)
        next_state_batch = Batch.from_data_list(next_state_graphs).to(self.device)
        
        # 计算当前Q值
        current_q = self.policy_net(state_batch)  # [batch_size, action_space_size]
        
        # 计算目标Q值
        with torch.no_grad():
            next_q = self.target_net(next_state_batch)  # [batch_size, action_space_size]
            next_q_max = next_q.max(1)[0]  # 取每个状态的最大Q值
            
            # 转换为张量
            rewards_tensor = torch.tensor(rewards, device=self.device, dtype=torch.float)
            dones_tensor = torch.tensor(dones, device=self.device, dtype=torch.float)
            
            # 计算目标Q值
            target_q = rewards_tensor + (1 - dones_tensor) * 0.99 * next_q_max
        
        # 选择执行的动作对应的Q值
        actions_tensor = torch.tensor(actions, device=self.device, dtype=torch.long)
        current_q_selected = current_q.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        
        # 计算损失
        loss = F.mse_loss(current_q_selected, target_q)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_net(self):
        """更新目标网络"""
        self.target_net.load_state_dict(self.policy_net.state_dict())