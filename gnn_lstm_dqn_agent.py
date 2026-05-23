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


class SequenceReplayBuffer:
    """支持序列训练的经验回放缓冲区"""
    def __init__(self, capacity=1000, sequence_length=5):
        self.buffer = deque(maxlen=capacity)
        self.sequence_length = sequence_length
        self.current_episode = []  # 存储当前episode的序列
    
    def start_episode(self):
        """开始新的episode"""
        self.current_episode = []
    
    def add_step(self, state, action, reward, next_state, done):
        """添加一个时间步"""
        self.current_episode.append((state, action, reward, next_state, done))
        
        # 如果episode结束，将完整序列存入缓冲区
        if done and len(self.current_episode) >= self.sequence_length:
            # 将完整episode存入缓冲区
            self.buffer.append(list(self.current_episode))
    
    def sample_sequences(self, batch_size):
        """采样完整序列（而不是单个时间步）"""
        if len(self.buffer) < batch_size:
            return None
        
        # 随机选择batch_size个完整episode
        episodes = random.sample(self.buffer, batch_size)
        
        # 从每个episode中截取固定长度的序列
        sequences = []
        for episode in episodes:
            if len(episode) >= self.sequence_length:
                # 随机选择序列起点
                start_idx = random.randint(0, len(episode) - self.sequence_length)
                sequence = episode[start_idx:start_idx + self.sequence_length]
                sequences.append(sequence)
        
        return sequences if sequences else None
    
    def __len__(self):
        return len(self.buffer)

class GNNLSTMDQNAgent:
    """统一的LSTM DQN智能体 - 修复训练推理不一致问题"""
    def __init__(self, env, device='cuda', sequence_length=5, update_type='soft', tau=0.01, target_update_freq=500):
        self.env = env
        self.device = device
        self.sequence_length = sequence_length
        
        # 获取动作空间大小
        self.action_space_size = len(env.nodes['rooms']) + 1
        self.action_list = ['cloud'] + list(env.nodes['rooms'].keys())
        
        # 使用序列化的经验回放
        self.memory = SequenceReplayBuffer(capacity=1000, sequence_length=sequence_length)
        
        # 策略网络和目标网络
        self.policy_net = GNNLSTMPolicy(
            sequence_length=sequence_length,
            action_space_size=self.action_space_size
        ).to(device)
        
        self.target_net = GNNLSTMPolicy(
            sequence_length=sequence_length,
            action_space_size=self.action_space_size
        ).to(device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-5)
        self.batch_size = 8  # 减小批大小，因为现在训练的是序列
        
        # 推理时的状态缓存
        self.state_sequence = deque(maxlen=sequence_length)
        self.transformer = StateTransformer(env)
        

        # 新增：学习率调度器
        # self.scheduler = optim.lr_scheduler.StepLR(
        #     self.optimizer, 
        #     step_size=50,    # 每50个epoch调整一次
        #     gamma=0.5        # 学习率减半
        # )
        
        # 或者使用余弦退火调度器（推荐）
        # self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
        #     self.optimizer, 
        #     T_max=200,       # 余弦周期（总训练轮次）
        #     eta_min=1e-7     # 最小学习率
        # )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-7
        )

        self.gamma = 0.99 # 折扣因子

        self.update_type = update_type
        self.tau = tau
        self.target_update_freq = target_update_freq
        self.update_counter = 0

    def reset_episode(self):
        """重置episode状态"""
        self.state_sequence.clear()
        self.memory.start_episode()

    def get_action(self, state, epsilon=0.1):
        """ε-greedy策略 - 修复逻辑：始终使用完整序列进行模型推理"""
        
        # 将当前状态加入序列
        self.state_sequence.append(state)
        
        # 确保序列长度达到要求才能使用模型
        if len(self.state_sequence) < self.sequence_length or random.random() < epsilon:
            # 随机选择合法动作 (探索或序列长度不足)
            valid_mask = state['valid_actions']
            valid_indices = valid_mask.nonzero().squeeze().tolist()
            if not valid_indices: return 'cloud'
            
            if len(self.state_sequence) < self.sequence_length:
                 # 序列不足时随机，但只在序列满后才进入模型的 ε-greedy 逻辑
                 action_idx = random.choice(valid_indices)
            else:
                 # ε-greedy 探索
                 action_idx = random.choice(valid_indices)
            
            return self.action_list[action_idx]

        # 使用完整序列进行推理 (利用)
        with torch.no_grad():
            sequence_graphs = [self.transformer.state_to_graph(s) for s in self.state_sequence]
            batch_data = Batch.from_data_list(sequence_graphs).to(self.device)
            
            # 修复：推理时 batch_size_override=1
            # q_values_flat: [T, action_space_size]
            q_values_flat, _ = self.policy_net(batch_data, batch_size_override=1, hidden_state=None) 
            
            # 取最后一个时间步的Q值，因为这是当前状态 S_T 的 Q 值
            last_q_values = q_values_flat[-1]
            
            # 应用合法动作掩码
            valid_mask = state['valid_actions'].to(self.device)
            masked_q = last_q_values.clone()
            masked_q[~valid_mask] = -float('inf')
            
            action_idx = torch.argmax(masked_q).item()
            return self.action_list[action_idx]

    def remember_step(self, state, action, reward, next_state, done):
        """记录一个时间步的经验"""
        self.memory.add_step(state, action, reward, next_state, done)

    def update_model(self):
        """修复后的训练方法 - 仅对序列最后一个时间步计算损失"""
        sequences = self.memory.sample_sequences(self.batch_size)
        if sequences is None: return None
        
        # 1. 扁平化数据以创建 PyG Batch
        all_states, all_actions, all_rewards, all_next_states, all_dones = [], [], [], [], []
        # 仅抽取序列最后一个时间步的动作、奖励、next_state 和 done
        last_actions, last_rewards, last_dones = [], [], []
        
        for sequence in sequences:
            states, actions, rewards, next_states, dones = zip(*sequence)
            all_states.extend(states)
            all_actions.extend(actions)
            all_next_states.extend(next_states)
            
            last_actions.append(self.action_list.index(actions[-1])) # 动作索引化
            last_rewards.append(rewards[-1])
            last_dones.append(dones[-1])
        
        # 2. 转换为图数据批次
        state_graphs = [self.transformer.state_to_graph(s) for s in all_states]
        next_state_graphs = [self.transformer.state_to_graph(s) for s in all_next_states]
        
        state_batch = Batch.from_data_list(state_graphs).to(self.device)
        next_state_batch = Batch.from_data_list(next_state_graphs).to(self.device)
        
        # 3. 计算 Q 值
        # current_q_flat: [B * T, action_space_size]
        current_q_flat, _ = self.policy_net(state_batch, batch_size_override=self.batch_size, hidden_state=None)
        
        with torch.no_grad():
            next_q_flat, _ = self.target_net(next_state_batch, batch_size_override=self.batch_size, hidden_state=None)

        # 4. 抽取最后一个时间步的 Q 值和动作
        # 从 [B * T, ...] 中每隔 T 步取一次，取 T-1 索引
        last_indices = [(i + 1) * self.sequence_length - 1 for i in range(self.batch_size)]

        # 策略网络 Q(S_T, a)
        last_current_q = current_q_flat[last_indices] # [B, action_space_size]
        
        # 目标网络 Q_target(S'_T, a')
        last_next_q = next_q_flat[last_indices] # [B, action_space_size]

        # 5. 计算目标 Q 值 (Target Q)
        next_q_max = last_next_q.max(1)[0] # [B]
        
        rewards_tensor = torch.tensor(last_rewards, device=self.device, dtype=torch.float)
        dones_tensor = torch.tensor(last_dones, device=self.device, dtype=torch.float)
        
        # Target = R_T + gamma * max_a' Q_target(S'_T, a') * (1 - Done_T)
        target_q = rewards_tensor + (1 - dones_tensor) * self.gamma * next_q_max
        
        # 6. 选择执行的动作对应的 Q 值 Q(S_T, A_T)
        actions_tensor = torch.tensor(last_actions, device=self.device, dtype=torch.long)
        current_q_selected = last_current_q.gather(1, actions_tensor.unsqueeze(1)).squeeze(1) # [B]
        
        # 7. 计算损失并优化
        # loss = F.mse_loss(current_q_selected, target_q)
        loss = F.huber_loss(current_q_selected, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # 8. 调度器更新
        self.scheduler.step(loss.item()) 

        # Hard update only: Increment step counter
        if self.update_type == 'hard':
            self.update_counter += 1
        
        # 步骤 9. 更新目标网络
        self.update_target_net()
        
        return loss.item()

    def update_target_net(self):
        """更新目标网络 (硬更新或软更新)"""
        if self.update_type == 'soft':
            # 软更新
            for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)
        
        elif self.update_type == 'hard':
            # 硬更新 (每 target_update_freq 步执行一次)
            if self.update_counter % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
