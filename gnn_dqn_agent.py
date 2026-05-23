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

class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity 
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        return random.sample(self.buffer, batch_size)
    
    # 新增：序列化方法
    def get_state(self):
        return {
            'buffer': list(self.buffer),
            'capacity': self.capacity
        }
    
    # 新增：反序列化方法
    def set_state(self, state):
        self.buffer = deque(state['buffer'], maxlen=state['capacity'])
        self.capacity = state['capacity']

    def __len__(self):
        return len(self.buffer)

class GNNAgent:
    """基于GNN的DQN智能体 - 适配新模拟器"""
    def __init__(self, env, device='cuda', update_type='hard', tau=0.01, target_update_freq=500):
        self.env = env
        self.device = device
        
        # 获取动作空间大小（机房数量+1）
        self.action_space_size = len(env.nodes['rooms']) + 1
        self.policy_net = GNNPolicy(action_space_size=self.action_space_size, req_feat_dim=5).to(device)
        self.target_net = GNNPolicy(action_space_size=self.action_space_size, req_feat_dim=5).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-5)
        self.memory = ReplayBuffer(capacity=10000)
        self.batch_size = 32
        self.transformer = StateTransformer(env)
        self.steps_done = 0
        
        # 构建动作列表（云端+所有机房）
        sorted_room_ids = sorted(env.nodes['rooms'].keys())
        self.action_list = ['cloud'] + list(sorted_room_ids)

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
        
        # 或者使用自适应调度器（最推荐）
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min',      # 监控loss最小化
            factor=0.5,      # 学习率减半
            patience=10,     # 10个epoch无改善则调整
            min_lr=1e-7      # 最小学习率
        )

        self.update_type = update_type # 'soft' or 'hard'
        self.tau = tau                 # Soft update parameter
        self.target_update_freq = target_update_freq # Hard update frequency (steps)
        self.update_counter = 0        # For hard update step counting
    
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
            # graph_data.batch = torch.zeros(graph_data.num_nodes, dtype=torch.long, device=self.device)
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
        
        # 不要在列表推导式里 .to(self.device)
        state_graphs = [self.transformer.state_to_graph(s) for s in states]
        next_state_graphs = [self.transformer.state_to_graph(s) for s in next_states]
        
        # 先 Batch，再整体移动到 GPU
        state_batch = Batch.from_data_list(state_graphs).to(self.device)
        next_state_batch = Batch.from_data_list(next_state_graphs).to(self.device)
        
        # 计算当前Q值
        current_q = self.policy_net(state_batch)  # [batch_size, action_space_size]
        
        # 计算目标Q值
        with torch.no_grad():
            next_q = self.target_net(next_state_batch)  # [batch_size, action_space_size]
            next_valid_masks = torch.stack(
                [s['valid_actions'] for s in next_states]
            ).to(self.device)
            next_q = next_q.masked_fill(~next_valid_masks, -1e9)
            next_q_max = next_q.max(1)[0]  # 取每个状态的最大合法Q值
            
            # 转换为张量
            rewards_tensor = torch.tensor(rewards, device=self.device, dtype=torch.float)
            dones_tensor = torch.tensor(dones, device=self.device, dtype=torch.float)
            
            # 计算目标Q值
            target_q = rewards_tensor + (1 - dones_tensor) * 0.99 * next_q_max
        
        # 选择执行的动作对应的Q值
        actions_tensor = torch.tensor(actions, device=self.device, dtype=torch.long)
        current_q_selected = current_q.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        
        # 计算损失
        # loss = F.mse_loss(current_q_selected, target_q)
        loss = F.huber_loss(current_q_selected, target_q)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Hard update only: Increment step counter
        if self.update_type == 'hard':
            self.update_counter += 1
            
        # 目标网络更新
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
                # print("Hard Target Net Updated.") # 可选打印

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
