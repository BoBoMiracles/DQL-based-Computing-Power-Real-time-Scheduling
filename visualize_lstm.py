import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pandas as pd
from tqdm import tqdm
import random
import time
from simulator import ComputingNetworkSimulator
from gnn_dqn_agent import GNNAgent, StateTransformer
from gnn_lstm_dqn_agent import LSTMDQNAgent

# === 全局参数 ===
rates = [0.1, 0.3, 0.5]  # 不同请求率
lstm_lengths = [3, 5, 10]  # LSTM时间窗长度
FOLDER_MAPPING = {}  # 将在主函数中填充
results_folder = 'lstm_comparison_results_longtime'  # 统一结果目录
num_experiments = 10  # 相同请求序列比较的实验次数
skip_first_requests = 0  # 舍弃前5个请求的处理结果

# 根据请求率确定仿真时间
def get_simulation_time(rate):
    if rate == 0.3:
        return 120000  
    elif rate == 0.5:
        return 72000  
    return 360000 

# 更新模型加载路径的函数
def get_model_path(model_type, rate, lstm_length=None):
    """根据模型类型返回正确的文件夹路径"""
    if model_type == 'gnn':
        return os.path.join(f'new_models/gnn_model_rate{rate}')
    elif model_type == 'gnn_lstm' and lstm_length:
        return os.path.join(f'new_models/gnn_lstm{lstm_length}_model_rate{rate}')
    return ''

def plot_all_training_histories(rate):
    """绘制所有模型的训练历史对比（奖励和损失）"""
    os.makedirs(f"{results_folder}/rate{rate}/training_history", exist_ok=True)
    
    # 绘制奖励历史
    plt.figure(figsize=(14, 8))
    
    # 基础GNN模型
    model_type = 'gnn'
    folder_path = get_model_path(model_type, rate)
    rewards = np.load(os.path.join(folder_path, f"{model_type}_rewards_history.npy"))
    window_size = 50
    moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    # 从窗口大小开始绘制，避免初始波动
    plt.plot(np.arange(window_size-1, len(rewards)), moving_avg, 'b-', 
             linewidth=2.5, label='GNN')
    
    # 不同时间窗的GNN+LSTM模型
    colors = ['g-', 'r-', 'c-']  # 不同时间窗使用不同颜色
    for idx, lstm_len in enumerate(lstm_lengths):
        model_type = 'gnn_lstm'
        folder_path = get_model_path(model_type, rate, lstm_len)
        rewards = np.load(os.path.join(folder_path, f"gnn_lstm_rewards_history.npy"))
        window_size = 50
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        # 从窗口大小开始绘制，避免初始波动
        plt.plot(np.arange(window_size-1, len(rewards)), moving_avg, colors[idx], 
                 linewidth=2.5, label=f'GNN+LSTM({lstm_len})')
    
    plt.title(f"Training Reward Comparison (Rate={rate}/s)")
    plt.xlabel("Episode")
    plt.ylabel(f"{window_size}-Episode Moving Avg Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{results_folder}/rate{rate}/training_history/all_models_reward_comparison.png", dpi=300)
    plt.close()
    
    # 绘制损失历史
    plt.figure(figsize=(14, 8))
    
    # 基础GNN模型
    model_type = 'gnn'
    folder_path = get_model_path(model_type, rate)
    losses = np.load(os.path.join(folder_path, f"{model_type}_loss_history.npy"))
    
    # 使用大窗口计算移动平均
    loss_window_size = 1000
    moving_avg = pd.Series(losses).rolling(loss_window_size, min_periods=1).mean()
    plt.plot(moving_avg, 'b-', linewidth=2.5, label='GNN')
    
    # 不同时间窗的GNN+LSTM模型
    for idx, lstm_len in enumerate(lstm_lengths):
        model_type = 'gnn_lstm'
        folder_path = get_model_path(model_type, rate, lstm_len)
        losses = np.load(os.path.join(folder_path, f"gnn_lstm_loss_history.npy"))
        moving_avg = pd.Series(losses).rolling(loss_window_size, min_periods=1).mean()
        plt.plot(moving_avg, colors[idx], linewidth=2.5, label=f'GNN+LSTM({lstm_len})')
    
    plt.title(f"Training Loss Comparison (Rate={rate}/s)")
    plt.xlabel("Training Step")
    plt.ylabel(f"Loss (Moving Avg, Window={loss_window_size})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{results_folder}/rate{rate}/training_history/all_models_loss_comparison.png", dpi=300)
    plt.close()

def compare_models_for_rate(rate):
    """比较指定请求率下所有模型的性能"""
    # 初始化环境
    simulation_time = get_simulation_time(rate)
    env = ComputingNetworkSimulator(
        'gurobi_solution_service_sources.csv', 
        'gurobi_solution_compute_nodes.csv', 
        rate=rate, 
        simulation_time=simulation_time
    )
    
    # 测试所有模型
    models_data = []
    model_names = []
    
    # 测试基础GNN模型
    model_type = 'gnn'
    folder_path = get_model_path(model_type, rate)
    agent = GNNAgent(env, device='cuda')
    model_path = os.path.join(folder_path, "gnn_dqn_final.pth")
    if os.path.exists(model_path):
        agent.policy_net.load_state_dict(torch.load(model_path, map_location='cuda'))
        gnn_metrics = test_model(agent, env, num_episodes=5)
        models_data.append(gnn_metrics)
        model_names.append('GNN')
    
    # 测试不同时间窗的GNN+LSTM模型
    for lstm_len in lstm_lengths:
        model_type = 'gnn_lstm'
        folder_path = get_model_path(model_type, rate, lstm_len)
        agent = LSTMDQNAgent(env, device='cuda')
        model_path = os.path.join(folder_path, "gnn_lstm_dqn_final.pth")
        if os.path.exists(model_path):
            agent.policy_net.load_state_dict(torch.load(model_path, map_location='cuda'))
            lstm_metrics = test_model(agent, env, num_episodes=5)
            models_data.append(lstm_metrics)
            model_names.append(f'GNN+LSTM({lstm_len})')
    
    # 测试随机策略
    random_metrics = test_random_policy(env, num_episodes=5)
    models_data.append(random_metrics)
    model_names.append('Random')
    
    # 创建性能对比数据
    metrics_df = pd.DataFrame({
        'Model': model_names,
        'Success Rate': [data['succeed_requests'] / data['total_requests'] for data in models_data],
        'Avg Reward per Request': [data['total_reward'] / data['total_requests'] for data in models_data],
        'Avg Latency (ms)': [data['total_latency'] / data['succeed_requests'] if data['succeed_requests'] > 0 else 0 for data in models_data],
        'Cloud Usage (%)': [data['cloud_requests'] / data['total_requests'] * 100 for data in models_data],
        'Resource Utilization (%)': [(data['total_processing'] / data['total_requests']) * 100 for data in models_data],
        'Avg Processing Time (ms)': [data['avg_processing_time'] * 1000 for data in models_data]  # 转换为毫秒
    })
    
    # 确保结果目录存在
    os.makedirs(f"{results_folder}/rate{rate}/model_comparison", exist_ok=True)
    
    # 绘制性能对比柱状图（3x2布局）- 包含所有模型
    plt.figure(figsize=(18, 12))
    
    plt.subplot(3, 2, 1)
    sns.barplot(x='Model', y='Success Rate', data=metrics_df)
    plt.title(f"Success Rate Comparison (Rate={rate}/s)")
    plt.ylabel("Success Rate")
    plt.xticks(rotation=20)
    
    plt.subplot(3, 2, 2)
    sns.barplot(x='Model', y='Avg Reward per Request', data=metrics_df)
    plt.title(f"Average Reward Comparison (Rate={rate}/s)")
    plt.ylabel("Reward")
    plt.xticks(rotation=20)
    
    plt.subplot(3, 2, 3)
    sns.barplot(x='Model', y='Avg Latency (ms)', data=metrics_df)
    plt.title(f"Average Latency Comparison (Rate={rate}/s)")
    plt.ylabel("Latency (ms)")
    plt.xticks(rotation=20)
    
    plt.subplot(3, 2, 4)
    sns.barplot(x='Model', y='Cloud Usage (%)', data=metrics_df)
    plt.title(f"Cloud Usage Comparison (Rate={rate}/s)")
    plt.ylabel("Cloud Usage (%)")
    plt.xticks(rotation=20)
    
    plt.subplot(3, 2, 5)
    sns.barplot(x='Model', y='Resource Utilization (%)', data=metrics_df)
    plt.title(f"Resource Utilization Comparison (Rate={rate}/s)")
    plt.ylabel("Utilization (%)")
    plt.xticks(rotation=20)
    
    # 第六个子图 - 处理时间对比
    plt.subplot(3, 2, 6)
    sns.barplot(x='Model', y='Avg Processing Time (ms)', data=metrics_df)
    plt.title(f"Average Processing Time Comparison (Rate={rate}/s)")
    plt.ylabel("Processing Time (ms)")
    plt.xticks(rotation=20)
    
    plt.tight_layout()
    plt.savefig(f"{results_folder}/rate{rate}/model_comparison/comprehensive_performance_comparison_all.png", dpi=300)
    plt.close()
    
    # 创建不包含Random模型的版本
    filtered_df = metrics_df[metrics_df['Model'] != 'Random']
    
    # 绘制性能对比柱状图（3x2布局）- 不包含Random模型
    plt.figure(figsize=(18, 12))
    
    # 1. 成功率对比
    plt.subplot(3, 2, 1)
    sns.barplot(x='Model', y='Success Rate', data=filtered_df)
    plt.title(f"Success Rate Comparison (Rate={rate}/s)")
    plt.ylabel("Success Rate")
    plt.ylim(0.7, 1.0)  # 确保Y轴范围合理
    plt.xticks(rotation=20)
    
    # 2. 平均奖励对比
    plt.subplot(3, 2, 2)
    sns.barplot(x='Model', y='Avg Reward per Request', data=filtered_df)
    plt.title(f"Average Reward Comparison (Rate={rate}/s)")
    plt.ylabel("Reward")
    
    # 动态计算Y轴范围
    min_reward = filtered_df['Avg Reward per Request'].min() * 0.995
    max_reward = filtered_df['Avg Reward per Request'].max() * 1.005
    plt.ylim(min_reward, max_reward)
    
    plt.xticks(rotation=20)
    
    # 3. 平均延迟对比
    plt.subplot(3, 2, 3)
    sns.barplot(x='Model', y='Avg Latency (ms)', data=filtered_df)
    plt.title(f"Average Latency Comparison (Rate={rate}/s)")
    plt.ylabel("Latency (ms)")
    
    # 动态计算Y轴范围
    min_latency = filtered_df['Avg Latency (ms)'].min() * 0.995
    max_latency = filtered_df['Avg Latency (ms)'].max() * 1.005
    plt.ylim(min_latency, max_latency)
    
    plt.xticks(rotation=20)
    
    # 4. 云端使用率对比
    plt.subplot(3, 2, 4)
    sns.barplot(x='Model', y='Cloud Usage (%)', data=filtered_df)
    plt.title(f"Cloud Usage Comparison (Rate={rate}/s)")
    plt.ylabel("Cloud Usage (%)")
    
    # 动态计算Y轴范围
    min_cloud = filtered_df['Cloud Usage (%)'].min() * 0.9
    max_cloud = filtered_df['Cloud Usage (%)'].max() * 1.1
    plt.ylim(min_cloud, max_cloud)
    
    plt.xticks(rotation=20)
    
    # 5. 资源利用率对比
    plt.subplot(3, 2, 5)
    sns.barplot(x='Model', y='Resource Utilization (%)', data=filtered_df)
    plt.title(f"Resource Utilization Comparison (Rate={rate}/s)")
    plt.ylabel("Utilization (%)")
    plt.ylim(70, 100)  # 确保Y轴范围合理
    plt.xticks(rotation=20)
    
    # 6. 处理时间对比
    plt.subplot(3, 2, 6)
    sns.barplot(x='Model', y='Avg Processing Time (ms)', data=filtered_df)
    plt.title(f"Average Processing Time Comparison (Rate={rate}/s)")
    plt.ylabel("Processing Time (ms)")
    
    # 动态计算Y轴范围
    min_processing = filtered_df['Avg Processing Time (ms)'].min() * 0.9
    max_processing = filtered_df['Avg Processing Time (ms)'].max() * 1.1
    plt.ylim(min_processing, max_processing)
    
    plt.xticks(rotation=20)
    
    plt.tight_layout()
    plt.savefig(f"{results_folder}/rate{rate}/model_comparison/comprehensive_performance_comparison_filtered.png", dpi=300)
    plt.close()
    
    # 单独绘制处理时间对比图
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Model', y='Avg Processing Time (ms)', data=metrics_df)
    plt.title(f"Average Processing Time Comparison (Rate={rate}/s)")
    plt.ylabel("Processing Time (ms)")
    plt.xticks(rotation=15)
    
    # 在柱子上添加数值标签
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.4f}", 
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center', 
                   xytext=(0, 9), 
                   textcoords='offset points',
                   fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{results_folder}/rate{rate}/model_comparison/processing_time_comparison.png", dpi=300)
    plt.close()

    # 绘制每个模型的机房利用率柱状图
    for model_name, model_metrics in zip(model_names, models_data):
        if model_name == 'Random':  # 跳过随机策略
            continue
            
        # 准备数据
        room_ids = []
        util_values = []
        for room_id, util in model_metrics['room_utilization'].items():
            # if util > 0:  # 只显示有利用率的机房
            #     room_ids.append(room_id)
            #     util_values.append(util)
            room_ids.append(room_id)
            util_values.append(util)
        
        # 排序
        sorted_indices = np.argsort(util_values)[::-1]
        sorted_room_ids = [room_ids[i] for i in sorted_indices]
        sorted_util_values = [util_values[i] for i in sorted_indices]
        
        # 绘制柱状图
        plt.figure(figsize=(12, 6))
        bars = plt.bar(sorted_room_ids, sorted_util_values, color='skyblue')
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.2%}', ha='center', va='bottom')
        
        plt.title(f"Room Utilization - {model_name} (Rate={rate}/s)")
        plt.xlabel("Room ID")
        plt.ylabel("Average Utilization")
        plt.xticks(rotation=45)
        plt.ylim(0, 1.1)  # 利用率范围0-100%
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # 保存图像
        safe_model_name = model_name.replace('+', '_').replace('(', '_').replace(')', '_')
        plt.savefig(f"{results_folder}/rate{rate}/model_comparison/{safe_model_name}_room_utilization.png", dpi=300)
        plt.close()
    
    # 保存性能数据
    metrics_df.to_csv(f"{results_folder}/rate{rate}/model_comparison/performance_metrics.csv", index=False)
    
    return metrics_df

def test_model(agent, env, num_episodes=10):
    metrics = {
        'total_requests': 0,
        'succeed_requests': 0,
        'cloud_requests': 0,
        'total_latency': 0,
        'total_processing': 0,
        'total_reward': 0,
        'total_processing_time': 0,
        'room_utilization': {room_id: [] for room_id in env.nodes['rooms'].keys()}
    }
    
    for _ in range(num_episodes):
        state = env.reset()
        if hasattr(agent, 'reset_hidden_state'):
            agent.reset_hidden_state()
        
        done = False
        while not done:
            start_time = time.time()
            action = agent.get_action(state, epsilon=0.01)
            end_time = time.time()
            processing_time = end_time - start_time
            metrics['total_processing_time'] += processing_time
            
            next_state, reward, done, ep_metrics = env.step(action)
            
            # 更新指标
            for key in metrics:
                if key in ep_metrics:
                    metrics[key] += ep_metrics[key]
            
            # 收集每个机房的利用率数据
            for room_id, room in env.nodes['rooms'].items():
                if room['utilization_history']:
                    # 计算该机房在当前episode的平均利用率
                    avg_util = np.mean(room['utilization_history'])
                    metrics['room_utilization'][room_id].append(avg_util)
            
            state = next_state
    
    # 计算平均值
    for key in metrics:
        if key != 'room_utilization':  # 单独处理利用率数据
            metrics[key] /= num_episodes
    
    # 计算平均处理时间
    metrics['avg_processing_time'] = metrics['total_processing_time'] / metrics['total_requests']
    
    # 计算每个机房的平均利用率
    for room_id, util_list in metrics['room_utilization'].items():
        if util_list:  # 只计算有数据的机房
            metrics['room_utilization'][room_id] = np.mean(util_list)
        else:
            metrics['room_utilization'][room_id] = 0
    
    return metrics

def test_random_policy(env, num_episodes=10):
    metrics = {
        'total_requests': 0,
        'succeed_requests': 0,
        'cloud_requests': 0,
        'total_latency': 0,
        'total_processing': 0,
        'total_reward': 0,
        'total_processing_time': 0,
        'room_utilization': {room_id: [] for room_id in env.nodes['rooms'].keys()}
    }
    
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            start_time = time.time()
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                action = 'cloud'
            else:
                action = random.choice(valid_actions)
            end_time = time.time()
            processing_time = end_time - start_time
            metrics['total_processing_time'] += processing_time
            
            next_state, reward, done, ep_metrics = env.step(action)
            
            # 更新指标
            for key in metrics:
                if key in ep_metrics:
                    metrics[key] += ep_metrics[key]
            
            # 收集每个机房的利用率数据
            for room_id, room in env.nodes['rooms'].items():
                if room['utilization_history']:
                    avg_util = np.mean(room['utilization_history'])
                    metrics['room_utilization'][room_id].append(avg_util)
            
            state = next_state
    
    # 计算平均值
    for key in metrics:
        if key != 'room_utilization':
            metrics[key] /= num_episodes
    
    # 计算平均处理时间
    metrics['avg_processing_time'] = metrics['total_processing_time'] / metrics['total_requests']
    
    # 计算每个机房的平均利用率
    for room_id, util_list in metrics['room_utilization'].items():
        if util_list:
            metrics['room_utilization'][room_id] = np.mean(util_list)
        else:
            metrics['room_utilization'][room_id] = 0
    
    return metrics

def compare_requests_performance(rate):
    """比较所有模型在同一批请求序列上的性能表现（多次实验取平均）"""
    # 创建结果目录
    os.makedirs(f"{results_folder}/rate{rate}/model_comparison/per_request", exist_ok=True)
    
    # 初始化环境
    simulation_time = get_simulation_time(rate)
    
    # 存储所有实验的结果
    all_experiment_results = {}
    
    # 进行多次实验
    for exp_id in range(num_experiments):
        print(f"Running experiment {exp_id+1}/{num_experiments} for rate={rate}")
        
        # 设置随机种子以确保同一批请求
        seed = 42 + exp_id
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        env = ComputingNetworkSimulator(
            'gurobi_solution_service_sources.csv', 
            'gurobi_solution_compute_nodes.csv', 
            rate=rate, 
            simulation_time=simulation_time
        )
        
        # 生成请求序列
        requests = generate_request_sequence(env, rate, simulation_time)
        
        # 测试所有模型
        model_results = {}
        
        # 测试基础GNN模型
        model_type = 'gnn'
        folder_path = get_model_path(model_type, rate)
        agent = GNNAgent(env, device='cuda')
        model_path = os.path.join(folder_path, "gnn_dqn_final.pth")
        if os.path.exists(model_path):
            agent.policy_net.load_state_dict(torch.load(model_path, map_location='cuda'))
            gnn_performance = test_performance_on_requests(env, agent, requests, 'GNN')
            model_results['GNN'] = gnn_performance
        
        # 测试不同时间窗的GNN+LSTM模型
        for lstm_len in lstm_lengths:
            model_type = 'gnn_lstm'
            folder_path = get_model_path(model_type, rate, lstm_len)
            agent = LSTMDQNAgent(env, device='cuda')
            model_path = os.path.join(folder_path, "gnn_lstm_dqn_final.pth")
            if os.path.exists(model_path):
                agent.policy_net.load_state_dict(torch.load(model_path, map_location='cuda'))
                lstm_performance = test_performance_on_requests(env, agent, requests, f'GNN+LSTM({lstm_len})')
                model_results[f'GNN+LSTM({lstm_len})'] = lstm_performance
        
        # 存储本次实验结果（包括请求序列）
        all_experiment_results[exp_id] = {
            'model_results': model_results,
            'requests': requests
        }
    
    # 计算平均结果
    avg_results = {}
    for model_name in all_experiment_results[0]['model_results'].keys():
        # 收集所有实验中该模型的结果
        all_model_results = [exp['model_results'][model_name] for exp in all_experiment_results.values()]
        
        # 计算平均序列长度（取最小长度）
        min_length = min(len(results) for results in all_model_results)
        
        # 初始化平均结果
        avg_performance = []
        for i in range(min_length):
            # 跳过前5个请求
            if i < skip_first_requests:
                continue
                
            # 收集所有实验中该请求位置的指标
            rewards = [results[i]['reward'] for results in all_model_results]
            latencies = [results[i]['latency'] for results in all_model_results]
            processing_times = [results[i]['processing_time'] for results in all_model_results]
            successes = [results[i]['success'] for results in all_model_results]
            
            # 计算平均值
            avg_reward = np.mean(rewards)
            avg_latency = np.mean(latencies)
            avg_processing_time = np.mean(processing_times)
            success_rate = np.mean(successes)
            
            # 保存平均结果
            avg_performance.append({
                'request_idx': i,
                'reward': avg_reward,
                'latency': avg_latency,
                'processing_time': avg_processing_time,
                'success': success_rate
            })
        
        avg_results[model_name] = avg_performance
    
    # 绘制平均结果对比图
    colors = ['b-', 'g-', 'r-', 'c-', 'm-']  # 不同模型使用不同颜色
    large_window_size = 10000  # 移动平均窗口大小
    
    # 奖励对比图（移动平均）- 从窗口大小开始绘制
    plt.figure(figsize=(14, 8))
    for idx, (model_name, performance) in enumerate(avg_results.items()):
        rewards = [req['reward'] for req in performance]
        moving_avg = pd.Series(rewards).rolling(large_window_size, min_periods=1).mean()
        # 从窗口大小开始绘制，避免初始波动
        plt.plot(np.arange(large_window_size-1, len(rewards)), moving_avg[large_window_size-1:], 
                 colors[idx], linewidth=2.5, label=model_name)
    
    plt.title(f"Moving Average Reward Comparison (Rate={rate}/s, {num_experiments} Experiments)")
    plt.xlabel("Request Index")
    plt.ylabel(f"Reward (Moving Avg, Window={large_window_size})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{results_folder}/rate{rate}/model_comparison/per_request/moving_avg_reward_comparison.png", dpi=300)
    plt.close()
    
    # 延迟对比图（移动平均）- 从窗口大小开始绘制
    plt.figure(figsize=(14, 8))
    for idx, (model_name, performance) in enumerate(avg_results.items()):
        latencies = [req['latency'] for req in performance]
        moving_avg = pd.Series(latencies).rolling(large_window_size, min_periods=1).mean()
        # 从窗口大小开始绘制，避免初始波动
        plt.plot(np.arange(large_window_size-1, len(latencies)), moving_avg[large_window_size-1:], 
                 colors[idx], linewidth=2.5, label=model_name)
    
    plt.title(f"Moving Average Latency Comparison (Rate={rate}/s, {num_experiments} Experiments)")
    plt.xlabel("Request Index")
    plt.ylabel(f"Latency (ms) (Moving Avg, Window={large_window_size})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{results_folder}/rate{rate}/model_comparison/per_request/moving_avg_latency_comparison.png", dpi=300)
    plt.close()
    
    # 处理时间对比图（移动平均）- 从窗口大小开始绘制
    plt.figure(figsize=(14, 8))
    for idx, (model_name, performance) in enumerate(avg_results.items()):
        processing_times = [req['processing_time'] * 1000 for req in performance]  # 转换为毫秒
        moving_avg = pd.Series(processing_times).rolling(large_window_size, min_periods=1).mean()
        # 从窗口大小开始绘制，避免初始波动
        plt.plot(np.arange(large_window_size-1, len(processing_times)), moving_avg[large_window_size-1:], 
                 colors[idx], linewidth=2.5, label=model_name)
    
    plt.title(f"Moving Average Processing Time Comparison (Rate={rate}/s, {num_experiments} Experiments)")
    plt.xlabel("Request Index")
    plt.ylabel(f"Processing Time (ms) (Moving Avg, Window={large_window_size})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{results_folder}/rate{rate}/model_comparison/per_request/moving_avg_processing_time_comparison.png", dpi=300)
    plt.close()
    
    # 成功率对比图（移动平均）- 从窗口大小开始绘制
    plt.figure(figsize=(14, 8))
    for idx, (model_name, performance) in enumerate(avg_results.items()):
        success_rates = [req['success'] for req in performance]
        moving_avg = pd.Series(success_rates).rolling(large_window_size, min_periods=1).mean()
        # 从窗口大小开始绘制，避免初始波动
        plt.plot(np.arange(large_window_size-1, len(success_rates)), moving_avg[large_window_size-1:], 
                 colors[idx], linewidth=2.5, label=model_name)
    
    plt.title(f"Moving Average Success Rate Comparison (Rate={rate}/s, {num_experiments} Experiments)")
    plt.xlabel("Request Index")
    plt.ylabel(f"Success Rate (Moving Avg, Window={large_window_size})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{results_folder}/rate{rate}/model_comparison/per_request/moving_avg_success_rate_comparison.png", dpi=300)
    plt.close()
    
    # 按请求类型分析成功率（使用最后一次实验的请求类型信息）
    # 首先确定所有请求类型
    last_requests = all_experiment_results[num_experiments-1]['requests']
    all_types = set(req['type'] for req in last_requests)
    
    # 为每种请求类型和模型收集平均成功率
    success_rates = {}
    for model_name in avg_results.keys():
        success_rates[model_name] = {}
        for req_type in all_types:
            # 收集所有实验中该模型对该类型请求的成功率
            type_success_rates = []
            for exp_id in range(num_experiments):
                # 获取该实验的请求序列
                exp_requests = all_experiment_results[exp_id]['requests']
                # 获取该实验的模型性能
                model_performance = all_experiment_results[exp_id]['model_results'][model_name]
                
                # 过滤该类型的请求（跳过前5个）
                type_indices = [i for i, req in enumerate(exp_requests) 
                                if req['type'] == req_type and i >= skip_first_requests]
                # 计算该模型在这些请求上的成功率
                if type_indices:
                    successes = [model_performance[i]['success'] for i in type_indices]
                    type_success_rates.append(np.mean(successes))
            
            # 计算平均成功率
            success_rates[model_name][req_type] = np.mean(type_success_rates) if type_success_rates else 0
    
    # 绘制成功率对比图
    plt.figure(figsize=(14, 8))
    bar_width = 0.15
    x = np.arange(len(all_types))
    
    for idx, (model_name, rates) in enumerate(success_rates.items()):
        model_rates = [rates[req_type] for req_type in all_types]
        plt.bar(x + idx * bar_width, model_rates, bar_width, label=model_name)
    
    plt.title(f"Success Rate by Request Type (Rate={rate}/s, {num_experiments} Experiments)")
    plt.xlabel("Request Type")
    plt.ylabel("Success Rate")
    plt.xticks(x + bar_width * (len(avg_results) - 1) / 2, all_types)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{results_folder}/rate{rate}/model_comparison/per_request/success_rate_by_type.png", dpi=300)
    plt.close()
    
    # 保存平均结果数据
    for model_name, performance in avg_results.items():
        df = pd.DataFrame(performance)
        safe_model_name = model_name.replace('+', '_').replace('(', '_').replace(')', '_')
        df.to_csv(f"{results_folder}/rate{rate}/model_comparison/per_request/{safe_model_name}_avg_request_performance.csv", index=False)
    
    # 保存最后一次实验的请求序列
    pd.DataFrame(last_requests).to_csv(f"{results_folder}/rate{rate}/model_comparison/per_request/request_sequence.csv", index=False)
    
    return avg_results

def generate_request_sequence(env, rate, simulation_time):
    """生成请求序列"""
    requests = []
    current_time = 0
    request_counter = 0
    request_history = []
    
    # 获取所有机房位置
    room_positions = [room['position'] for room in env.nodes['rooms'].values()]
    
    while current_time < simulation_time:
        inter_arrival = np.random.exponential(1/rate)
        current_time += inter_arrival
        request_counter += 1
        
        # 70%的请求在机房附近生成（热点区域）
        if random.random() < 0.7 and room_positions:
            center_room = random.choice(room_positions)
            position = (
                np.clip(np.random.normal(center_room[0], 10), 0, 100),
                np.clip(np.random.normal(center_room[1], 10), 0, 100)
            )
        else:
            position = (
                np.random.uniform(0, 100),
                np.random.uniform(0, 100)
            )
        
        # 添加位置依赖性
        if request_history and random.random() < 0.6:
            last_position = request_history[-1]['position']
            position = (
                np.clip(last_position[0] + np.random.normal(0, 5), 0, 100),
                np.clip(last_position[1] + np.random.normal(0, 5), 0, 100)
            )
        
        # 根据请求类型分配不同的特性
        request_type = np.random.choice(
            ['safety-critical', 'infotainment', 'adas'], 
            p=[0.2, 0.5, 0.3]
        )
        
        if request_type == 'safety-critical':
            compute_demand = np.clip(np.random.normal(15, 2), 5, 20)
            max_latency = np.random.choice([15, 20, 25])
            base_process = 2
            process_time = np.clip(np.random.normal(base_process, 0.05), 0.5, 4)
        elif request_type == 'infotainment':
            compute_demand = np.clip(np.random.normal(10, 3), 5, 20)
            max_latency = np.random.choice([30, 35, 40])
            base_process = 20
            process_time = np.clip(np.random.normal(base_process, 3), 5, 50)
        else:
            compute_demand = np.clip(np.random.normal(18, 1.5), 5, 20)
            max_latency = np.random.choice([20, 25, 30])
            base_process = 10
            process_time = np.clip(np.random.normal(base_process, 1), 2, 30)
        
        # 机房附近的请求更可能是高计算需求类型
        if room_positions:
            min_dist = min([env._calculate_distance(position, room) for room in room_positions])
            if min_dist < 15:
                if random.random() < 0.7:
                    compute_demand = np.clip(compute_demand * 1.5, 5, 20)
        
        # 找到最近基站
        min_dist = float('inf')
        nearest_bs = None
        for bs_id, bs in env.nodes['base_stations'].items():
            dist = env._calculate_distance(position, bs['position'])
            if dist < min_dist:
                min_dist = dist
                nearest_bs = bs_id
        
        requests.append({
            'req_id': f"REQ_{request_counter}_{request_type[0]}",
            'time': current_time,
            'position': position,
            'base_station': nearest_bs,
            'compute_demand': compute_demand,
            'max_latency': max_latency,
            'process_time': process_time,
            'type': request_type
        })
        
        # 保存到历史用于位置依赖性
        request_history.append({
            'position': position,
            'time': current_time
        })
    
    return requests

def test_performance_on_requests(env, agent, requests, agent_name):
    """在给定的请求序列上测试模型性能"""
    # 重置环境
    state = env.reset()
    if agent_name.startswith('GNN+LSTM'):
        agent.reset_hidden_state()
    
    performance = []
    
    for i, request in enumerate(tqdm(requests, desc=f"Testing {agent_name}")):
        # 设置当前请求
        env.current_time = request['time']
        env.current_request = {
            'position': request['position'],
            'base_station': request['base_station'],
            'compute_demand': request['compute_demand'],
            'max_latency': request['max_latency'],
            'process_time': request['process_time']
        }
        
        # 找到最近基站和本地机房
        nearest_bs = None
        min_dist = float('inf')
        for bs_id, bs in env.nodes['base_stations'].items():
            dist = env._calculate_distance(request['position'], bs['position'])
            if dist < min_dist:
                min_dist = dist
                nearest_bs = bs
        
        home_room_id = nearest_bs['room_id']
        env.current_request['home_room'] = home_room_id
        
        # 获取当前状态
        state = env._get_state()
        
        # 选择动作并记录处理时间
        start_time = time.time()
        action = agent.get_action(state, epsilon=0.01)
        end_time = time.time()
        processing_time = end_time - start_time
        
        # 执行动作
        _, reward, done, metrics = env.step(action)
        
        # 记录性能指标
        performance.append({
            'request_idx': i,
            'time': request['time'],
            'position_x': request['position'][0],
            'position_y': request['position'][1],
            'compute_demand': request['compute_demand'],
            'max_latency': request['max_latency'],
            'type': request['type'],
            'action': action,
            'reward': reward,
            'latency': metrics['last_latency'],
            'success': metrics['last_success'],
            'used_cloud': metrics['used_cloud'],
            'processing_time': processing_time  # 添加处理时间
        })
    
    return performance

def main():
    """主函数：执行所有对比任务"""
    # 设置绘图风格
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 12
    os.makedirs(results_folder, exist_ok=True)
    
    # 用于最终汇总的DataFrame
    summary_data = []
    
    for rate in rates:
        print(f"\n{'='*50}")
        print(f"Processing Rate={rate}/s")
        print(f"{'='*50}")
        
        # 确保结果目录存在
        os.makedirs(f"{results_folder}/rate{rate}", exist_ok=True)
        
        # 1. 绘制所有模型的训练历史对比
        print("Visualizing training histories...")
        plot_all_training_histories(rate)
        
        # 2. 对比模型性能指标
        print("Comparing model performance metrics...")
        metrics_df = compare_models_for_rate(rate)
        
        # 保存到汇总数据
        for _, row in metrics_df.iterrows():
            row_dict = row.to_dict()
            row_dict['rate'] = rate
            summary_data.append(row_dict)
        
        # 3. 在同一请求序列上对比性能（多次实验取平均）
        print("Comparing performance on same request sequence (multiple experiments)...")
        compare_requests_performance(rate)
        
        print(f"Completed processing for rate={rate}/s")
    
    # 保存汇总结果
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f"{results_folder}/performance_summary.csv", index=False)
    
    # 绘制所有rate的处理时间对比
    plt.figure(figsize=(14, 8))
    sns.barplot(x='rate', y='Avg Processing Time (ms)', hue='Model', data=summary_df)
    plt.title("Average Processing Time Comparison Across Rates")
    plt.xlabel("Request Rate (requests/s)")
    plt.ylabel("Processing Time (ms)")
    plt.legend(title='Model')
    plt.tight_layout()
    plt.savefig(f"{results_folder}/processing_time_across_rates.png", dpi=300)
    plt.close()
    
    # 绘制所有rate的成功率对比
    plt.figure(figsize=(14, 8))
    sns.barplot(x='rate', y='Success Rate', hue='Model', data=summary_df)
    plt.title("Success Rate Comparison Across Rates")
    plt.xlabel("Request Rate (requests/s)")
    plt.ylabel("Success Rate")
    plt.legend(title='Model')
    plt.tight_layout()
    plt.savefig(f"{results_folder}/success_rate_across_rates.png", dpi=300)
    plt.close()
    
    # 绘制所有rate的云端使用率对比
    plt.figure(figsize=(14, 8))
    sns.barplot(x='rate', y='Cloud Usage (%)', hue='Model', data=summary_df)
    plt.title("Cloud Usage Comparison Across Rates")
    plt.xlabel("Request Rate (requests/s)")
    plt.ylabel("Cloud Usage (%)")
    plt.legend(title='Model')
    plt.tight_layout()
    plt.savefig(f"{results_folder}/cloud_usage_across_rates.png", dpi=300)
    plt.close()
    
    print("\nAll processing completed. Results saved to", results_folder)

if __name__ == "__main__":
    main()