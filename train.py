import torch
from environmental_simulation import ComputingNetworkSimulator
from dqn_agent import GNNAgent
from utils import StateTransformer
import numpy as np

def train():
    # 初始化环境
    env = ComputingNetworkSimulator('base_stations.csv', 'rooms.csv')
    agent = GNNAgent(env, device='cuda')
    
    # 训练参数
    episodes = 1000
    target_update = 10  # 目标网络更新间隔
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995
    
    epsilon = epsilon_start
    
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # 计算当前epsilon
            epsilon = max(epsilon_end, epsilon_decay * epsilon)
            
            # 选择动作
            action = agent.get_action(state, epsilon)
            
            # 执行动作
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            
            # 存储经验
            agent.remember(state, action, reward, next_state, done)
            
            # 更新模型
            agent.update_model()
            
            # 状态转移
            state = next_state
        
        # 更新目标网络
        if ep % target_update == 0:
            agent.update_target_net()
        
        print(f"Episode {ep+1}/{episodes}, Reward: {total_reward:.1f}, Epsilon: {epsilon:.3f}")
        
        # 保存模型
        if (ep + 1) % 100 == 0:
            torch.save(agent.policy_net.state_dict(), f"gnn_dqn_ep{ep+1}.pth")
    
    # 保存最终模型
    torch.save(agent.policy_net.state_dict(), "gnn_dqn_final.pth")

if __name__ == "__main__":
    train()