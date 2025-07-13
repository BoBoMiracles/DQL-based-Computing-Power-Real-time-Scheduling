import torch
from simulator import ComputingNetworkSimulator
from gnn_lstm_dqn_agent import LSTMDQNAgent
import numpy as np
import time

def train():
    # 初始化环境
    env = ComputingNetworkSimulator('gurobi_solution_service_sources.csv', 'gurobi_solution_compute_nodes.csv')
    agent = LSTMDQNAgent(env, device='cuda')
    
    # 训练参数
    episodes = 1000
    target_update = 10  # 目标网络更新间隔
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995
    
    epsilon = epsilon_start
    rewards_history = []
    loss_history = []
    start_time = time.time()
    
    for ep in range(episodes):
        state = env.reset()
        agent.reset_hidden_state()  # 重置LSTM状态
        
        total_reward = 0
        done = False
        
        while not done:
            # 计算当前epsilon
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            
            # 选择动作
            action = agent.get_action(state, epsilon)
            
            # 执行动作
            next_state, reward, done, metrics = env.step(action)
            total_reward += reward
            
            # 存储经验
            agent.remember(state, action, reward, next_state, done)
            
            # 更新模型
            loss = agent.update_model()
            if loss is not None:
                loss_history.append(loss)
            
            # 状态转移
            state = next_state
        
        # 更新目标网络
        if ep % target_update == 0:
            agent.update_target_net()
            
        rewards_history.append(total_reward)
        elapsed = time.time() - start_time
        
        print(f"Episode {ep+1}/{episodes}, "
              f"Reward: {total_reward:.1f}, "
              f"Epsilon: {epsilon:.3f}, "
              f"Time: {elapsed:.2f}s")
        
        # 保存模型
        if (ep + 1) % 100 == 0:
            torch.save(agent.policy_net.state_dict(), f"gnn_lstm_dqn_ep{ep+1}.pth")
    
    # 保存最终模型
    torch.save(agent.policy_net.state_dict(), "gnn_lstm_dqn_final.pth")
    
    # 保存训练历史
    np.save('gnn_lstm_rewards_history.npy', np.array(rewards_history))
    np.save('gnn_lstm_loss_history.npy', np.array(loss_history))

if __name__ == "__main__":
    train()