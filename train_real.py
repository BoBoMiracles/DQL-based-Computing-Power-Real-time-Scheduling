import torch
from simulator_real import ComputingNetworkSimulator
from gnn_dqn_agent_real import GNNAgent
from gnn_lstm_dqn_agent_real import LSTMDQNAgent
import numpy as np
import time
import argparse
import os

def train(model_type, device, lstm_len, arr_rate, sim_time):
    # 初始化环境
    env = ComputingNetworkSimulator('gurobi_solution_service_sources_real.csv', 
                                   'gurobi_solution_compute_nodes_real.csv', 
                                   rate=arr_rate, 
                                   simulation_time=sim_time)
    
    # 在训练开始前可视化拓扑结构
    os.makedirs('topology_visualizations', exist_ok=True)
    output_path = f'topology_visualizations/topology_rate{arr_rate}.png'
    env.visualize_topology(output_path)

    rate = env.request_rate
    
    # 根据模型类型选择智能体
    if model_type == 'gnn_lstm':
        agent = LSTMDQNAgent(env, device=device, history_len=lstm_len)  # 决定lstm的历史时间窗长度
        len = agent.history_len
        folder_name = f'real_models/gnn_lstm{len}_model_rate{rate}'
        print("Training GNN+LSTM model...")
    else:
        agent = GNNAgent(env, device=device)
        folder_name = f'real_models/gnn_model_rate{rate}'
        print("Training GNN model...")
    
    # 训练参数
    episodes = 100
    target_update = 10  # 目标网络更新间隔
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995
    
    epsilon = epsilon_start
    rewards_history = []
    loss_history = []
    start_time = time.time()
    os.makedirs(folder_name, exist_ok=True)
    
    for ep in range(episodes):
        state = env.reset()
        
        # 重置LSTM状态（如果是LSTM模型）
        if model_type == 'gnn_lstm':
            agent.reset_hidden_state()
        
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
              f"Loss: {loss:.3f}, "
              f"Epsilon: {epsilon:.3f}, "
              f"Time: {elapsed:.2f}s")
        
        # 保存模型
        if (ep + 1) % 20 == 0:
            model_name = f"{model_type}_dqn_ep{ep+1}.pth"
            save_path = os.path.join(folder_name, model_name)
            torch.save(agent.policy_net.state_dict(), save_path)
            print(f"Model saved to {save_path}")
    
    # 保存最终模型
    final_name = f"{model_type}_dqn_final.pth"
    final_path = os.path.join(folder_name, final_name)
    torch.save(agent.policy_net.state_dict(), final_path)
    
    # 保存训练历史
    reward_name = f'{model_type}_rewards_history.npy'
    reward_path = os.path.join(folder_name, reward_name)
    loss_name = f'{model_type}_loss_history.npy'
    loss_path = os.path.join(folder_name, loss_name)
    np.save(reward_path, np.array(rewards_history))
    np.save(loss_path, np.array(loss_history))
    print(f"Training completed. Final model saved to {final_path}")

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Train DQN agent for computing network scheduling')
    parser.add_argument('--model', type=str, default='gnn', choices=['gnn', 'gnn_lstm'],
                        help='Model type: gnn or gnn_lstm (default: gnn)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device to use for training (default: cuda)')
    
    args = parser.parse_args()
    
    # 开始训练
    rate = (3, )
    t = 0
    if args.model == 'gnn_lstm':   
        for r in rate:
            if r == 0.5:
                t = 7200
            elif r == 0.3:
                t = 12000
            else:
                t = 3600
            for len in (10,):
                print(f"Training GNN_LSTM model with time window = {len} and arrival rate = {r}, sim_time = {t}")
                train(model_type=args.model, device=args.device, lstm_len=len, arr_rate=r, sim_time=t)
    else: 
        for r in rate:
            if r == 0.5:
                t = 7200
            elif r == 0.3:
                t = 12000
            else:
                t = 3600
            print(f"Training GNN model with arrival rate = {r}, sim_time = {t}")
            train(model_type=args.model, device=args.device, lstm_len=0, arr_rate=r, sim_time=t)