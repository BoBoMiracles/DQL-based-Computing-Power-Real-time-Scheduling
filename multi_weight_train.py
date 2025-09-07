import torch
from simulator import ComputingNetworkSimulator
from gnn_lstm_dqn_agent import LSTMDQNAgent
from gnn_dqn_agent import GNNAgent  # 导入GNN智能体
import numpy as np
import time
import os
import pickle
import argparse
import shutil

def train_experiment(experiment_id, model_type, device, lstm_len, arr_rate, reward_weights):
    """训练单个实验配置"""
    # 根据到达率调整仿真时间
    if arr_rate == 5.0:
        sim_time = 2400
    elif arr_rate == 10.0:
        sim_time = 1200
    else:
        sim_time = 3600  # 默认1小时
    
    # 创建实验目录
    weights_str = "_".join([f"{w:.1f}" for w in reward_weights])
    if model_type == 'gnn_lstm':
        experiment_dir = os.path.join("weight_experiment_results", 
                                 f"exp{experiment_id}",
                                 f"model_{model_type}{lstm_len}",
                                 f"rate_{arr_rate}",
                                 f"weights_{weights_str}") 
    else: 
        experiment_dir = os.path.join("weight_experiment_results", 
                                 f"exp{experiment_id}",
                                 f"model_{model_type}",
                                 f"rate_{arr_rate}",
                                 f"weights_{weights_str}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # 保存实验参数
    experiment_params = {
        'experiment_id': experiment_id,
        'model_type': model_type,
        'device': device,
        'lstm_len': lstm_len if model_type == 'gnn_lstm' else None,
        'arrival_rate': arr_rate,
        'simulation_time': sim_time,
        'reward_weights': reward_weights
    }
    with open(os.path.join(experiment_dir, "params.pkl"), 'wb') as f:
        pickle.dump(experiment_params, f)
    
    if model_type == 'gnn_lstm':
        print(f"\n=== 开始实验 {experiment_id}: model={model_type}{lstm_len}, rate={arr_rate}, weights={reward_weights}, sim_time={sim_time}s ===") 
    else: print(f"\n=== 开始实验 {experiment_id}: model={model_type}, rate={arr_rate}, weights={reward_weights}, sim_time={sim_time}s ===") 
    
    # 初始化环境
    env = ComputingNetworkSimulator(
        'gurobi_solution_service_sources.csv',
        'gurobi_solution_compute_nodes.csv',
        rate=arr_rate, 
        simulation_time=sim_time,
        reward_weights=reward_weights
    )
    
    # 根据模型类型初始化智能体
    if model_type == 'gnn_lstm':
        agent = LSTMDQNAgent(
            env, 
            device=device, 
            history_len=lstm_len
        )
    else:  # gnn
        agent = GNNAgent(
            env, 
            device=device
        )
    
    # 训练参数
    episodes = 100
    target_update = 10  # 目标网络更新间隔
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995
    
    epsilon = epsilon_start
    rewards_history = []
    loss_history = []
    success_rates = []
    cloud_usage_rates = []
    avg_latencies = []
    
    start_time = time.time()
    
    # 训练循环
    for ep in range(episodes):
        state = env.reset()
        if model_type == 'gnn_lstm':
            agent.reset_hidden_state()
        
        total_reward = 0
        episode_success = 0
        episode_cloud = 0
        episode_count = 0
        episode_latency = 0
        done = False
        
        while not done:
            # 更新epsilon
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            
            # 选择动作
            action = agent.get_action(state, epsilon)
            
            # 执行动作
            next_state, reward, done, metrics = env.step(action)
            total_reward += reward
            
            # 记录每步指标
            if metrics['last_success']:
                episode_success += 1
            if metrics['used_cloud']:
                episode_cloud += 1
            episode_count += 1
            episode_latency += metrics['last_latency']
            
            # 存储经验
            agent.remember(state, action, reward, next_state, done)
            
            # 更新模型
            loss = agent.update_model()
            if loss is not None:
                loss_history.append(loss)
            
            # 状态转移
            state = next_state
        
        # 计算整体成功率、云端使用率和平均延迟
        success_rate = episode_success / episode_count if episode_count > 0 else 0
        cloud_rate = episode_cloud / episode_count if episode_count > 0 else 0
        avg_latency = episode_latency / episode_count if episode_count > 0 else 0
        
        # 更新目标网络
        if ep % target_update == 0:
            agent.update_target_net()
            
        # 记录指标
        rewards_history.append(total_reward)
        success_rates.append(success_rate)
        cloud_usage_rates.append(cloud_rate)
        avg_latencies.append(avg_latency)
        
        elapsed = time.time() - start_time
        print(f"Experiment {experiment_id} | Episode {ep+1}/{episodes} | "
              f"Reward: {total_reward:.1f} | "
              f"Success: {success_rate:.2%} | "
              f"Cloud: {cloud_rate:.2%} | "
              f"Latency: {avg_latency:.2f}ms | "
              f"Loss: {loss if loss is not None else 'N/A'} | "
              f"Epsilon: {epsilon:.3f} | "
              f"Time: {elapsed:.2f}s")
        
        # 每20个episode保存中间模型
        if (ep + 1) % 20 == 0:
            model_name = f"{model_type}_dqn_ep{ep+1}.pth"
            save_path = os.path.join(experiment_dir, model_name)
            torch.save(agent.policy_net.state_dict(), save_path)
    
    # 保存最终模型和指标
    final_model = f"{model_type}_dqn_final.pth"
    torch.save(agent.policy_net.state_dict(), os.path.join(experiment_dir, final_model))
    
    # 保存训练历史
    training_history = {
        'rewards': rewards_history,
        'losses': loss_history,
        'success_rates': success_rates,
        'cloud_usage': cloud_usage_rates,
        'avg_latencies': avg_latencies
    }
    with open(os.path.join(experiment_dir, "training_history.pkl"), 'wb') as f:
        pickle.dump(training_history, f)
    
    print(f"实验 {experiment_id} 完成! 结果保存在: {experiment_dir}")
    
    # 返回一些最终指标用于总结
    final_metrics = {
        'final_reward': rewards_history[-1],
        'avg_success_rate': np.mean(success_rates[-10:]),  # 最后10个episode的平均成功率
        'avg_cloud_rate': np.mean(cloud_usage_rates[-10:]),
        'avg_latency': np.mean(avg_latencies[-10:])
    }
    return final_metrics

def run_all_experiments(device='cuda', model_choice='all'):
    """运行所有实验配置，允许通过model_choice指定运行的模型类型"""
    # 创建总实验目录
    base_dir = "weight_experiment_results"
    # if os.path.exists(base_dir):
    #     shutil.rmtree(base_dir)
    # os.makedirs(base_dir)
    
    # 实验参数配置
    reward_weights = [
        (0.5, 1.0, 0.3, 0.2),  # 默认权重
        (1.0, 0.7, 0.2, 0.1),  # 高基础权重
        (0.3, 1.5, 0.1, 0.1),  # 高延迟惩罚
        (0.3, 0.7, 0.8, 0.2),  # 高利用率奖励
        (0.5, 0.8, 0.2, 0.5)   # 高一致性奖励
    ]
    
    arrival_rates = [2.0, 10.0]
    # arrival_rates = [10.0, ]
    
    # 根据用户选择确定要运行的模型类型
    if model_choice == 'gnn':
        model_types = ['gnn']
    elif model_choice == 'gnn_lstm':
        model_types = ['gnn_lstm']
    else:  # all或其他
        model_types = ['gnn', 'gnn_lstm']
    
    lstm_len = 10  # LSTM时间窗
    
    experiment_id = 1
    all_results = []
    
    # 遍历所有配置
    for model_type in model_types:
        for rate in arrival_rates:
            for weights in reward_weights:
                try:
                    final_metrics = train_experiment(
                        experiment_id=experiment_id,
                        model_type=model_type,
                        device=device,
                        lstm_len=lstm_len,
                        arr_rate=rate,
                        reward_weights=weights
                    )
                    
                    # 保存结果
                    result = {
                        'experiment_id': experiment_id,
                        'model_type': model_type,
                        'arrival_rate': rate,
                        'reward_weights': weights,
                        'final_metrics': final_metrics
                    }
                    all_results.append(result)
                    
                    # 写入实验日志
                    with open(os.path.join(base_dir, "experiment_summary.txt"), 'a') as f:
                        if model_type == 'gnn_lstm':
                            f.write(f"实验 {experiment_id} - 模型: {model_type}{lstm_len}, 到达率: {rate}, 权重: {weights}\n")
                        else: f.write(f"实验 {experiment_id} - 模型: {model_type}, 到达率: {rate}, 权重: {weights}\n")
                        f.write(f"最终奖励: {final_metrics['final_reward']:.1f}, ")
                        f.write(f"平均成功率: {final_metrics['avg_success_rate']:.2%}, ")
                        f.write(f"云端使用率: {final_metrics['avg_cloud_rate']:.2%}, ")
                        f.write(f"平均延迟: {final_metrics['avg_latency']:.2f}ms\n\n")
                    
                    experiment_id += 1
                except Exception as e:
                    print(f"实验 {experiment_id} 失败: {str(e)}")
                    # 记录失败信息
                    with open(os.path.join(base_dir, "experiment_summary.txt"), 'a') as f:
                        f.write(f"实验 {experiment_id} - 模型: {model_type}{lstm_len}, 到达率: {rate}, 权重: {weights} - 失败: {str(e)}\n\n")
                    experiment_id += 1
    
    # 保存所有结果
    with open(os.path.join(base_dir, "all_experiment_results.pkl"), 'wb') as f:
        pickle.dump(all_results, f)
    
    print("\n=== 所有实验完成! ===")
    print(f"总计实验数: {len(all_results)}")
    print(f"结果保存在: {base_dir}")
    print(f"训练的模型类型: {model_types}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='运行多权重和到达率实验')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='训练使用的设备 (默认: cuda)')
    # 添加模型选择参数
    parser.add_argument('--model', type=str, default='all', 
                        choices=['all', 'gnn', 'gnn_lstm'],
                        help='指定训练的模型类型: gnn, gnn_lstm 或 all (默认: all)')
    args = parser.parse_args()
    
    # 运行所有实验，传递模型选择参数
    run_all_experiments(device=args.device, model_choice=args.model)