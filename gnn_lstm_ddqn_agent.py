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
        # 1. 确保所有输入特征都在 CPU 上
        # 即使 state['x'] 已经是 CPU 的，调用 .cpu() 也几乎没有开销
        # 如果 state['x'] 是 GPU 的（来自续训加载），.cpu() 会把它搬回内存，防止混合设备报错
        x = state['x'].cpu()
        edge_index = state['edge_index'].cpu()
        req_feat = state['req_feat'].cpu()
        
        # 2. 生成图对象 (全部在 CPU)
        return Data(
            x=x,
            edge_index=edge_index,
            req_feat=req_feat,
            # 这里的 device 显式指定为 cpu，或者直接不写 device 参数（默认就是 cpu）
            batch=torch.zeros(x.size(0), dtype=torch.long, device='cpu')
        )


class SequenceReplayBuffer:
    """支持序列训练的经验回放缓冲区"""
    def __init__(self, capacity=1000, sequence_length=5):
        self.buffer = deque(maxlen=capacity)
        self.sequence_length = sequence_length
        self.current_episode = []  # 存储当前episode的序列
        self.capacity = capacity 
    
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
    
    # 新增：序列化方法
    def get_state(self):
        return {
            'buffer': [list(episode) for episode in self.buffer],
            'current_episode': list(self.current_episode),
            'capacity': self.capacity,
            'sequence_length': self.sequence_length
        }
    
    # 新增：反序列化方法
    def set_state(self, state):
        self.buffer = deque([list(episode) for episode in state['buffer']], 
                           maxlen=state['capacity'])
        self.current_episode = list(state['current_episode'])
        self.capacity = state['capacity']
        self.sequence_length = state['sequence_length']

    def __len__(self):
        return len(self.buffer)

class GNNLSTMDDQNAgent:
    """统一的LSTM DDQN智能体 - 修复训练推理不一致问题"""
    def __init__(self, env, sequence_length, device='cuda', update_type='soft', tau=0.01, target_update_freq=500):
        self.env = env
        self.device = device
        self.sequence_length = sequence_length
        
        # 获取动作空间大小
        self.action_space_size = len(env.nodes['rooms']) + 1
        sorted_room_ids = sorted(env.nodes['rooms'].keys())
        self.action_list = ['cloud'] + list(sorted_room_ids)
        
        # 使用序列化的经验回放
        self.memory = SequenceReplayBuffer(capacity=1000, sequence_length=sequence_length)
        
        # 策略网络和目标网络
        self.policy_net = GNNLSTMPolicy(
            sequence_length=sequence_length,
            action_space_size=self.action_space_size,
            req_feat_dim=5
        ).to(device)
        
        self.target_net = GNNLSTMPolicy(
            sequence_length=sequence_length,
            action_space_size=self.action_space_size, 
            req_feat_dim=5
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
            self.optimizer, mode='min', factor=0.5, patience=50, min_lr=1e-5
        )

        self.gamma = 0.95 # 折扣因子

        self.steps_done = 0

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

            action_idx = random.choice(valid_indices)
            
            return self.action_list[action_idx]

        # 使用完整序列进行推理 (利用)
        with torch.no_grad():
            # 先在 CPU 列表化
            sequence_graphs = [self.transformer.state_to_graph(s) for s in self.state_sequence]
            # Batch 后再 to device
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
        """实现 Double DQN (DDQN) 逻辑"""
        sequences = self.memory.sample_sequences(self.batch_size)
        if sequences is None: return None
        
        # 1. 扁平化数据以创建 PyG Batch
        all_states, all_actions, all_rewards, all_next_states, all_dones = [], [], [], [], []
        # 仅抽取序列最后一个时间步的动作、奖励、next_state 和 done
        last_actions, last_rewards, last_dones, last_next_valid_masks = [], [], [], []
        
        for sequence in sequences:
            states, actions, rewards, next_states, dones = zip(*sequence)
            all_states.extend(states)
            all_actions.extend(actions)
            all_next_states.extend(next_states)
            
            # 将动作转换为索引
            last_actions.append(self.action_list.index(actions[-1])) # 动作索引化
            last_rewards.append(rewards[-1])
            last_dones.append(dones[-1])
            last_next_valid_masks.append(next_states[-1]['valid_actions'])
        
        # 2. 转换为图数据批次
        # 在 CPU 上转换成图对象 (不要在这里调用 .to(self.device))
        state_graphs = [self.transformer.state_to_graph(s) for s in all_states]
        next_state_graphs = [self.transformer.state_to_graph(s) for s in all_next_states]
        
        # 在 CPU 上进行 Batch 拼接 (速度很快)
        state_batch_cpu = Batch.from_data_list(state_graphs)
        next_state_batch_cpu = Batch.from_data_list(next_state_graphs)
        
        # 一次性将整个 Batch 传输到 GPU (极大减少 CUDA 通信开销)
        state_batch = state_batch_cpu.to(self.device)
        next_state_batch = next_state_batch_cpu.to(self.device)
        
        # 3. 计算 Q 值
        # current_q_flat: [B * T, action_space_size]
        current_q_flat, _ = self.policy_net(state_batch, batch_size_override=self.batch_size, hidden_state=None)

        # 4. 抽取最后一个时间步的 Q 值和动作
        # 从 [B * T, ...] 中每隔 T 步取一次，取 T-1 索引
        last_indices = [(i + 1) * self.sequence_length - 1 for i in range(self.batch_size)]

        # 策略网络 Q(S_T, a)
        last_current_q = current_q_flat[last_indices] # [B, action_space_size]
        
        # 目标网络 Q_target(S'_T, a') (原始值，用于评估)
        with torch.no_grad():
            next_q_target_flat, _ = self.target_net(next_state_batch, batch_size_override=self.batch_size, hidden_state=None)
            last_next_q_target = next_q_target_flat[last_indices] # [B, action_space_size]

        # ==================== DDQN 核心逻辑修改 ====================
        with torch.no_grad():
            # 1. 使用 POLICY 网络选择下一个状态 S'_T 的最佳动作 A_hat = argmax_a' Q_policy(S'_T, a')
            
            # 注意：这里需要再次调用 policy_net，或使用 policy_net 对 next_state_batch 的评估。
            # 为了效率，我们假设 policy_net 评估的 Q 值在计算 current_q_flat 时没有对 next_state_batch 评估。
            # 为了确保 no_grad 块内的纯净性，我们重新对 next_state_batch 使用 policy_net
            
            # 策略网络评估下一个状态 Q_policy(S'_T, a')
            next_q_policy_flat, _ = self.policy_net(next_state_batch, batch_size_override=self.batch_size, hidden_state=None)
            last_next_q_policy = next_q_policy_flat[last_indices] # [B, action_space_size]
            next_valid_masks = torch.stack(last_next_valid_masks).to(self.device)
            last_next_q_policy = last_next_q_policy.masked_fill(~next_valid_masks, -1e9)

            # 获取策略网络选择的最佳动作索引 A_hat
            best_action_indices = last_next_q_policy.argmax(1).unsqueeze(1) # [B, 1]
            
            # 2. 使用 TARGET 网络评估该最佳动作的 Q 值 Q_target(S'_T, A_hat)
            # last_next_q_target 是目标网络对 S'_T 的所有动作评估 [B, action_space_size]
            next_q_max = last_next_q_target.gather(1, best_action_indices).squeeze(1) # [B]
            
            # 5. 计算目标 Q 值 (Target Q)
            rewards_tensor = torch.tensor(last_rewards, device=self.device, dtype=torch.float)
            dones_tensor = torch.tensor(last_dones, device=self.device, dtype=torch.float)
            
            # Target = R_T + gamma * Q_target(S'_T, A_hat) * (1 - Done_T)
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
        # self.scheduler.step(loss.item()) 

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

    # 修改：save_state 方法
    def save_state(self, filepath):
        state = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            # 兼容性处理：如果 scheduler 存在则保存
            'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self, 'scheduler') else None,
            'memory_state': self.memory.get_state(),
            'steps_done': getattr(self, 'steps_done', 0),
            'update_counter': getattr(self, 'update_counter', 0),
            # --- 新增：保存随机数生成器状态，保证环境交互和采样的一致性 ---
            'rng_states': {
                'torch': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                'numpy': np.random.get_state(),
                'python': random.getstate()
            },
            # ---------------------------------------------------------
            'init_params': {
                'device': self.device,
                'update_type': self.update_type,
                'tau': self.tau,
                'target_update_freq': self.target_update_freq,
                # 注意：LSTM/Transformer 需要保存 sequence_length，普通 GNN 不需要
                'sequence_length': getattr(self, 'sequence_length', None) 
            }
        }
        
        # 针对 LSTM/Transformer Agent，还需要保存状态序列缓存
        if hasattr(self, 'state_sequence'):
             state['state_sequence'] = list(self.state_sequence)

        torch.save(state, filepath)
        # print(f"Agent state saved to {filepath}") # 可选：减少打印刷屏

    # 修改：load_state 方法
    def load_state(self, filepath, env=None):
        if env is not None:
            self.env = env
            
        # 加载文件
        state = torch.load(filepath, map_location=self.device, weights_only=False)
        
        # 1. 恢复网络和优化器
        self.policy_net.load_state_dict(state['policy_net_state_dict'])
        self.target_net.load_state_dict(state['target_net_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        
        if hasattr(self, 'scheduler') and state.get('scheduler_state_dict') is not None:
            self.scheduler.load_state_dict(state['scheduler_state_dict'])
        
        # 2. 恢复经验回放缓冲区 (最占内存的部分，但必须恢复以保证分布一致)
        self.memory.set_state(state['memory_state'])
        
        # 3. 恢复 LSTM/Transformer 的推理序列缓存
        if hasattr(self, 'state_sequence') and 'state_sequence' in state:
            # 重新构建 deque
            self.state_sequence = deque(state['state_sequence'], maxlen=self.sequence_length)
            print(f"Sequence buffer restored. Len: {len(self.state_sequence)}")

        # 4. 恢复计数器
        self.steps_done = state.get('steps_done', 0)
        self.update_counter = state.get('update_counter', 0)
        
        # 5. --- 最终修复：更稳健的随机数状态恢复 ---
        if 'rng_states' in state:
            rng = state['rng_states']
            
            # 处理 torch 随机状态
            if rng.get('torch') is not None:
                torch_rng_state = rng['torch']
                
                # 如果是张量但不是 CPU 上的 ByteTensor
                if isinstance(torch_rng_state, torch.Tensor):
                    # 转换为 ByteTensor
                    if torch_rng_state.dtype != torch.uint8:
                        torch_rng_state = torch_rng_state.to(torch.uint8)
                    
                    # 移动到 CPU
                    torch_rng_state = torch_rng_state.cpu().contiguous()
                    
                    # 额外检查：确保形状正确
                    expected_size = 5056  # 从您的调试信息中看到的大小
                    if torch_rng_state.numel() != expected_size:
                        print(f"⚠️ Warning: Torch RNG state size mismatch. Expected {expected_size}, got {torch_rng_state.numel()}")
                        
                    try:
                        torch.set_rng_state(torch_rng_state)
                        print(f"✅ torch RNG state restored. Size: {torch_rng_state.shape}")
                    except Exception as e:
                        print(f"❌ Failed to restore torch RNG state: {e}")
            
            # 处理 CUDA 随机状态
            if rng.get('cuda') is not None and torch.cuda.is_available():
                cuda_rng_state = rng['cuda']
                
                if isinstance(cuda_rng_state, torch.Tensor):
                    # 确保是 ByteTensor (uint8)
                    if cuda_rng_state.dtype != torch.uint8:
                        cuda_rng_state = cuda_rng_state.to(torch.uint8)
                    
                    # 【关键】：CUDA 随机状态必须放在 CPU 上才能被 set_rng_state 接受
                    cuda_rng_state = cuda_rng_state.cpu()
                    
                    try:
                        torch.cuda.set_rng_state(cuda_rng_state)
                        print(f"✅ CUDA RNG state restored.")
                    except Exception as e:
                        print(f"❌ Failed to restore CUDA RNG state: {e}")
            
            # 恢复 numpy 和 python 随机状态
            if rng.get('numpy') is not None:
                try:
                    np.random.set_state(rng['numpy'])
                    print("✅ Numpy RNG state restored.")
                except Exception as e:
                    print(f"❌ Failed to restore numpy RNG state: {e}")
            
            if rng.get('python') is not None:
                try:
                    random.setstate(rng['python'])
                    print("✅ Python RNG state restored.")
                except Exception as e:
                    print(f"❌ Failed to restore python RNG state: {e}")
        else:
            print("⚠️ Warning: RNG states not found in checkpoint. Slight fluctuation may occur.")
        # ---------------------------------------------------------

        print(f"Agent state loaded from {filepath}")
        print(f"Resumed from step: {self.steps_done}, Memory size: {len(self.memory)}")
        
        return state
            
