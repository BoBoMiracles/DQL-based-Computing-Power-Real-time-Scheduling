import torch
import random
import numpy as np
from collections import deque
from torch import optim
from gnn_lstm_model import GNNLSTMPolicy
from torch_geometric.data import Batch
from torch_geometric.data import Data
import torch.nn.functional as F

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

class LSTMDQNAgent:
    """基于GNN+LSTM的DQN智能体"""
    def __init__(self, env, device='cuda', history_len=5):
        self.env = env
        self.device = device
        
        # 获取动作空间大小（机房数量+1）
        self.action_space_size = len(env.nodes['rooms']) + 1
        
        # 构建动作列表（云端+所有机房）
        self.action_list = ['cloud'] + list(env.nodes['rooms'].keys())
        
        # 使用GNN+LSTM策略网络
        self.policy_net = GNNLSTMPolicy(
            action_space_size=self.action_space_size
        ).to(device)
        
        self.target_net = GNNLSTMPolicy(
            action_space_size=self.action_space_size
        ).to(device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # LSTM相关参数
        self.history_len = history_len
        self.hidden_state = None
        self.state_history = deque(maxlen=history_len)
        
        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.memory = ReplayBuffer(capacity=10000)
        self.batch_size = 32
        self.transformer = StateTransformer(env)
        self.steps_done = 0

    def reset_hidden_state(self):
        """重置LSTM隐藏状态"""
        self.hidden_state = None
        self.state_history.clear()

    def get_action(self, state, epsilon=0.1):
        """ε-greedy策略 - 使用历史信息和LSTM"""
        # 将当前状态添加到历史队列
        self.state_history.append(state)
        
        # 如果历史记录不足，使用随机策略
        if len(self.state_history) < self.history_len:
            # 随机选择合法动作
            valid_mask = state['valid_actions']
            valid_indices = valid_mask.nonzero().squeeze().tolist()
            if not valid_indices:
                return 'cloud'  # 默认选择云端
            action_idx = random.choice(valid_indices)
            return self.action_list[action_idx]
        
        # 构建历史状态序列
        history_states = list(self.state_history)[-self.history_len:]
        
        with torch.no_grad():
            # 转换历史状态为图数据
            graph_data = [self.transformer.state_to_graph(s) for s in history_states]
            batch_data = Batch.from_data_list(graph_data).to(self.device)
            
            # 预测Q值（使用LSTM）
            q_values, self.hidden_state = self.policy_net(batch_data, self.hidden_state)
            
            # 获取当前状态的合法动作掩码
            valid_mask = state['valid_actions'].to(self.device)
            
            # 应用合法动作掩码 - 只考虑最后一个状态
            # 因为我们只关心当前状态的决策
            masked_q = q_values[-1].clone()  # 只取最后一个状态的Q值
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
        """更新策略网络 - 使用历史序列训练LSTM"""
        batch = self.memory.sample(self.batch_size)
        if batch is None:
            return None
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 为每个样本构建历史序列
        state_sequences = []
        next_state_sequences = []
        
        for i in range(len(batch)):
            # 获取当前样本的历史序列
            state_seq = self._get_history_sequence(states[i])
            next_state_seq = self._get_history_sequence(next_states[i])
            
            state_sequences.append(state_seq)
            next_state_sequences.append(next_state_seq)
        
        # 转换状态序列为图数据
        state_graphs = []
        for seq in state_sequences:
            graphs = [self.transformer.state_to_graph(s) for s in seq]
            state_graphs.append(Batch.from_data_list(graphs))
        
        next_state_graphs = []
        for seq in next_state_sequences:
            graphs = [self.transformer.state_to_graph(s) for s in seq]
            next_state_graphs.append(Batch.from_data_list(graphs))
        
        # 创建批次图
        state_batch = Batch.from_data_list(state_graphs).to(self.device)
        next_state_batch = Batch.from_data_list(next_state_graphs).to(self.device)
        
        # 计算当前Q值
        current_q, _ = self.policy_net(state_batch)  # [batch_size * history_len, action_space_size]
        
        # 计算目标Q值
        with torch.no_grad():
            next_q, _ = self.target_net(next_state_batch)  # [batch_size * history_len, action_space_size]
            
            # 只取每个序列最后一个状态的目标Q值
            # 因为奖励是针对序列的最后一个状态（当前状态）的
            next_q_last = next_q.view(len(batch), self.history_len, -1)[:, -1, :]  # [batch_size, action_space_size]
            next_q_max = next_q_last.max(1)[0]  # [batch_size]
            
            # 转换为张量
            rewards_tensor = torch.tensor(rewards, device=self.device, dtype=torch.float)
            dones_tensor = torch.tensor(dones, device=self.device, dtype=torch.float)
            
            # 计算目标Q值
            target_q = rewards_tensor + (1 - dones_tensor) * 0.99 * next_q_max
        
        # 只取每个序列最后一个状态的当前Q值
        current_q_last = current_q.view(len(batch), self.history_len, -1)[:, -1, :]  # [batch_size, action_space_size]
        
        # 选择执行的动作对应的Q值
        actions_tensor = torch.tensor(actions, device=self.device, dtype=torch.long)
        current_q_selected = current_q_last.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        
        # 计算损失
        loss = F.mse_loss(current_q_selected, target_q)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def _get_history_sequence(self, state):
        """为给定状态构建历史序列（如果历史不足则用当前状态填充）"""
        if len(self.state_history) < self.history_len:
            # 历史不足，用当前状态填充
            return [state] * self.history_len
        else:
            # 返回最近的history_len个状态
            return list(self.state_history)[-self.history_len:]
    
    def update_target_net(self):
        """更新目标网络"""
        self.target_net.load_state_dict(self.policy_net.state_dict())