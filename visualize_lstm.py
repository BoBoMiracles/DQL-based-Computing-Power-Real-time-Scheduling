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
from gnn_lstm_ddqn_agent import GNNLSTMDDQNAgent

# === 全局参数 ===
rates = [0.1, 0.5, 1]  # 不同请求率
lstm_lengths = [3, 5, 10, 20, 30]  # LSTM时间窗长度
FOLDER_MAPPING = {}  # 将在主函数中填充
results_folder = 'results_revision_v1'  # 统一结果目录
num_experiments = 50  # 相同请求序列比较的实验次数
n_episode = 50  # 整体评价对比模型的episode
skip_first_requests = max(lstm_lengths)  # 舍弃lstm未启动的请求的处理结果

# 根据请求率确定仿真时间
def get_simulation_time(rate):
    if rate <= 0.3:
        return 12000  
    elif rate >= 1:
        return 3600  
    return 7200

# 更新模型加载路径的函数
def get_model_path(model_type, rate, lstm_length=None):
    """根据模型类型返回正确的文件夹路径"""
    if model_type == 'gnn':
        return os.path.join(f'models_revision_v1/gnn_model_rate{rate}')
    elif model_type == 'gnn_lstm' and lstm_length:
        return os.path.join(f'models_revision_v1/gnn_lstm{lstm_length}_model_rate{rate}')
    return ''

def plot_all_training_histories(rate):
    """绘制所有模型的训练历史对比（奖励和损失）"""
    os.makedirs(f"{results_folder}/rate{rate}/training_history", exist_ok=True)
    
    # 绘制奖励历史
    plt.figure(figsize=(14, 8))
    
    # 基础GNN模型
    model_type = 'gnn'
    folder_path = get_model_path(model_type, rate)
    rewards_path = os.path.join(folder_path, f"{model_type}_rewards_history.npy")
    
    if os.path.exists(rewards_path):
        rewards = np.load(rewards_path)
        window_size = 100
        if len(rewards) > window_size:
            moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            # 从窗口大小开始绘制，避免初始波动
            plt.plot(np.arange(window_size-1, len(rewards)), moving_avg, 'b-', 
                     linewidth=2.5, label='GNN')
    
    # 不同时间窗的GNN+LSTM模型
    colors = ['g-', 'r-', 'c-', 'm-', 'y-', 'k-']  # 不同时间窗使用不同颜色
    for idx, lstm_len in enumerate(lstm_lengths):
        model_type = 'gnn_lstm'
        folder_path = get_model_path(model_type, rate, lstm_len)
        rewards_path = os.path.join(folder_path, f"gnn_lstm_rewards_history.npy")
        
        if os.path.exists(rewards_path):
            rewards = np.load(rewards_path)
            window_size = 100
            if len(rewards) > window_size:
                moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                # 从窗口大小开始绘制，避免初始波动
                color_idx = idx % len(colors)  # 使用模运算避免索引越界
                plt.plot(np.arange(window_size-1, len(rewards)), moving_avg, colors[color_idx], 
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
    losses_path = os.path.join(folder_path, f"{model_type}_loss_history.npy")
    
    if os.path.exists(losses_path):
        losses = np.load(losses_path)
        # 使用大窗口计算移动平均
        loss_window_size = min(10000, len(losses) // 10)  # 自适应窗口大小
        # loss_window_size = 50
        moving_avg = pd.Series(losses).rolling(loss_window_size, min_periods=1).mean()
        plt.plot(moving_avg, 'b-', linewidth=2.5, label='GNN')
    
    # 不同时间窗的GNN+LSTM模型
    for idx, lstm_len in enumerate(lstm_lengths):
        model_type = 'gnn_lstm'
        folder_path = get_model_path(model_type, rate, lstm_len)
        losses_path = os.path.join(folder_path, f"gnn_lstm_loss_history.npy")
        
        if os.path.exists(losses_path):
            losses = np.load(losses_path)
            loss_window_size = min(10000, len(losses) // 10)  # 自适应窗口大小
            # loss_window_size = 50
            moving_avg = pd.Series(losses).rolling(loss_window_size, min_periods=1).mean()
            color_idx = idx % len(colors)  # 使用模运算避免索引越界
            plt.plot(moving_avg, colors[color_idx], linewidth=2.5, label=f'GNN+LSTM({lstm_len})')
    
    plt.title(f"Training Loss Comparison (Rate={rate}/s)")
    plt.xlabel("Training Step")
    plt.ylabel(f"Loss (Moving Avg, Window={loss_window_size})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{results_folder}/rate{rate}/training_history/all_models_loss_comparison.png", dpi=300)
    plt.close()

def plot_network_with_requests(env, requests, rate, experiment_id):
    """绘制网络拓扑和请求分布在同一张图中"""
    # 设置绘图风格
    plt.figure(figsize=(14, 10))
    
    # 绘制网络拓扑
    # 1. 绘制云端
    plt.scatter(env.cloud_node['position'][0], env.cloud_node['position'][1], 
               c='gold', s=400, marker='*', edgecolors='black', linewidth=2, 
               label='Cloud Node', zorder=10)
    
    # 2. 绘制机房
    for room_id, room in env.nodes['rooms'].items():
        plt.scatter(room['position'][0], room['position'][1], 
                   c='red', s=250, marker='*', edgecolors='black', linewidth=1.5,
                   zorder=9)
        plt.text(room['position'][0] + 2, room['position'][1] + 2, 
                f"R{room_id}", fontsize=10, fontweight='bold', zorder=11)
    
    # 3. 绘制基站
    for bs_id, bs in env.nodes['base_stations'].items():
        plt.scatter(bs['position'][0], bs['position'][1], 
                   c='blue', s=100, marker='s', edgecolors='black', linewidth=1,
                   zorder=8)
    
    # 4. 绘制连接线
    # 基站-机房连接
    for bs_id, bs in env.nodes['base_stations'].items():
        room_id = bs['room_id']
        if room_id in env.nodes['rooms']:
            room = env.nodes['rooms'][room_id]
            plt.plot([bs['position'][0], room['position'][0]], 
                    [bs['position'][1], room['position'][1]], 
                    'gray', alpha=0.5, linewidth=1, zorder=1)
    
    # 机房之间连接
    room_ids = list(env.nodes['rooms'].keys())
    for i in range(len(room_ids)):
        for j in range(i+1, len(room_ids)):
            room1 = env.nodes['rooms'][room_ids[i]]
            room2 = env.nodes['rooms'][room_ids[j]]
            plt.plot([room1['position'][0], room2['position'][0]], 
                    [room1['position'][1], room2['position'][1]], 
                    'lightgray', alpha=0.3, linewidth=0.5, zorder=1)
    
    # 绘制请求分布（按服务类别）
    service_categories = {
        "ultra_low_latency": [],
        "low_latency": [],
        "high_latency_tolerance": []
    }
    
    for req in requests:
        # 将具体服务类型映射到三大类别
        category = env.service_categories.get(req['type'], "high_latency_tolerance")
        service_categories[category].append(req['position'])
    
    # 定义请求类别的样式
    type_colors = plt.cm.Set3(np.linspace(0, 1, 5))
    category_styles = {
        'ultra_low_latency': {'color': type_colors[0], 'marker': 'o', 's': 40, 'alpha': 0.7, 'label': 'Ultra Low Latency'},
        'low_latency': {'color': type_colors[1], 'marker': 's', 's': 35, 'alpha': 0.6, 'label': 'Low Latency'},
        'high_latency_tolerance': {'color': type_colors[2], 'marker': '^', 's': 38, 'alpha': 0.6, 'label': 'High Latency Tolerance'}
    }
    
    for category, positions in service_categories.items():
        if positions:
            style = category_styles[category]
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]
            plt.scatter(x_coords, y_coords, 
                       c=style['color'], marker=style['marker'], s=style['s'],
                       alpha=style['alpha'], edgecolors='black', linewidth=0.5,
                       label=style['label'], zorder=5)
    
    # 设置图形属性
    plt.title(f"Network Topology with Request Distribution\nRate: {rate}/s, Experiment: {experiment_id}", 
              fontsize=14, fontweight='bold')
    plt.xlabel("X Coordinate", fontsize=12)
    plt.ylabel("Y Coordinate", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 设置坐标轴范围
    plt.xlim(0, 115)
    plt.ylim(0, 115)
    
    # 保存图像
    os.makedirs(f"{results_folder}/rate{rate}/network_requests", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{results_folder}/rate{rate}/network_requests/network_requests_exp{experiment_id}.png", 
                dpi=300, bbox_inches='tight')
    plt.close()

def calc_global_avg_util(room_util_dict):
    """从 room_utilization 字典计算全局平均利用率"""
    if not room_util_dict:
        return 0
    # 只计算有数据的机房的平均值
    valid_utils = [v for v in room_util_dict.values() if v > 0]
    if not valid_utils:
        return 0
    return np.mean(valid_utils)

def compare_models_for_rate(rate, episode):
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
    print(f"Testing GNN Model with Rate={rate}/s")
    model_type = 'gnn'
    folder_path = get_model_path(model_type, rate)
    agent = GNNAgent(env, device='cuda')
    model_path = os.path.join(folder_path, "gnn_dqn_final.pth")
    if os.path.exists(model_path):
        agent.policy_net.load_state_dict(torch.load(model_path, map_location='cuda'))
        gnn_metrics = test_model(agent, env, num_episodes=episode)
        models_data.append(gnn_metrics)
        model_names.append('GNN')
    
    # 测试不同时间窗的GNN+LSTM模型
    for lstm_len in lstm_lengths:
        print(f"Testing GNN_LSTM Model with Time Window={lstm_len}, Rate={rate}/s")
        model_type = 'gnn_lstm'
        folder_path = get_model_path(model_type, rate, lstm_len)
        agent = GNNLSTMDDQNAgent(env, device='cuda', sequence_length=lstm_len)
        model_path = os.path.join(folder_path, "gnn_lstm_dqn_final.pth")
        if os.path.exists(model_path):
            agent.policy_net.load_state_dict(torch.load(model_path, map_location='cuda'))
            lstm_metrics = test_model(agent, env, num_episodes=episode)
            models_data.append(lstm_metrics)
            model_names.append(f'GNN+LSTM({lstm_len})')
    
    # 测试随机策略
    print(f"Testing Random policy with Rate={rate}/s")
    random_metrics = test_random_policy(env, num_episodes=episode)
    models_data.append(random_metrics)
    model_names.append('Random')

    # 测试greedy策略
    print(f"Testing Greedy Policy with Rate={rate}/s")
    greedy_metrics = test_greedy_policy(env, num_episodes=episode)
    models_data.append(greedy_metrics)
    model_names.append('Greedy')

    # 测试UAF策略
    print(f"Testing Lowest Utilization First Policy with Rate={rate}/s")
    uaf_metrics = test_luf_policy(env, num_episodes=episode)
    models_data.append(uaf_metrics)
    model_names.append('Lowest Utilization First')

    # 创建性能对比数据
    metrics_df = pd.DataFrame({
        'Model': model_names,
        'Success Rate': [data['succeed_requests'] / data['total_requests'] for data in models_data],
        'Avg Reward per Request': [data['total_reward'] / data['total_requests'] for data in models_data],
        'Avg Latency (ms)': [data['total_latency'] / data['succeed_requests'] if data['succeed_requests'] > 0 else 0 for data in models_data],
        'Cloud Usage (%)': [data['cloud_requests'] / data['total_requests'] * 100 for data in models_data],
        'Resource Utilization (%)': [calc_global_avg_util(data['room_utilization']) * 100 for data in models_data],
        'Avg Compute Demand': [data['total_processing'] / data['total_requests'] for data in models_data],
        'Avg Processing Time (ms)': [data['avg_processing_time'] * 1000 for data in models_data]  # 转换为毫秒
    })
    
    # 确保结果目录存在
    print(f"Visualizing the results of models comparison for rate={rate}/s")
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
    filtered_df = metrics_df[
        (metrics_df['Model'] != 'Random') &
        (metrics_df['Model'] != 'Lowest Utilization First') &
        (metrics_df['Model'] != 'Greedy')
    ]
    
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
    min_utilization = filtered_df['Resource Utilization (%)'].min() * 0.9
    max_utilization = filtered_df['Resource Utilization (%)'].max() * 1.1
    plt.ylim(min_utilization, max_utilization)  # 确保Y轴范围合理
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

    # 修复：绘制每个模型的机房利用率柱状图
    for model_name, model_metrics in zip(model_names, models_data):
        if model_name == 'Random':  # 跳过随机策略
            continue
            
        # 准备数据 - 修复：确保有利用率数据
        room_ids = []
        util_values = []
        
        for room_id, util in model_metrics.get('room_utilization', {}).items():
            # 确保util是数值类型
            if isinstance(util, (int, float)):
                room_ids.append(room_id)
                util_values.append(util)
            elif isinstance(util, list) and len(util) > 0:
                # 如果util是列表，计算平均值
                room_ids.append(room_id)
                util_values.append(np.mean(util))
        
        if not room_ids:  # 如果没有数据，跳过
            continue
            
        # 排序
        sorted_indices = np.argsort(util_values)[::-1]
        sorted_room_ids = [room_ids[i] for i in sorted_indices]
        sorted_util_values = [util_values[i] for i in sorted_indices]
        
        # 绘制柱状图
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(sorted_room_ids)), sorted_util_values, color='skyblue')
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.2%}', ha='center', va='bottom')
        
        plt.title(f"Room Utilization - {model_name} (Rate={rate}/s)")
        plt.xlabel("Room ID")
        plt.ylabel("Average Utilization")
        plt.xticks(range(len(sorted_room_ids)), sorted_room_ids, rotation=45)
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
        if hasattr(agent, 'reset_episode'):
            agent.reset_episode()
        
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
    if metrics['total_requests'] > 0:
        metrics['avg_processing_time'] = metrics['total_processing_time'] / metrics['total_requests']
    else:
        metrics['avg_processing_time'] = 0
    
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
    if metrics['total_requests'] > 0:
        metrics['avg_processing_time'] = metrics['total_processing_time'] / metrics['total_requests']
    else:
        metrics['avg_processing_time'] = 0
    
    # 计算每个机房的平均利用率
    for room_id, util_list in metrics['room_utilization'].items():
        if util_list:
            metrics['room_utilization'][room_id] = np.mean(util_list)
        else:
            metrics['room_utilization'][room_id] = 0
    
    return metrics

def test_greedy_policy(env, num_episodes=10):
    """
    测试 greedy 贪婪策略的性能。
    """
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
            request = env.current_request
            action, _ = env._greedy_policy(request)
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
    if metrics['total_requests'] > 0:
        metrics['avg_processing_time'] = metrics['total_processing_time'] / metrics['total_requests']
    else:
        metrics['avg_processing_time'] = 0
    
    # 计算每个机房的平均利用率
    for room_id, util_list in metrics['room_utilization'].items():
        if util_list:
            metrics['room_utilization'][room_id] = np.mean(util_list)
        else:
            metrics['room_utilization'][room_id] = 0
    
    return metrics

def test_luf_policy(env, num_episodes=10):
    """
    测试最低利用率优先策略 (LUF) 的性能。
    """
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
            request = env.current_request
            action, _ = env._lowest_utilization_policy(request)
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
    if metrics['total_requests'] > 0:
        metrics['avg_processing_time'] = metrics['total_processing_time'] / metrics['total_requests']
    else:
        metrics['avg_processing_time'] = 0
    
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
        print(f"  Generating common request sequence for experiment {exp_id+1}...")
        requests = []
        env.reset() # 重置环境以开始生成
        requests.append(env.current_request)
        done = False
        while not done:
            # 调用模拟器内部的生成器
            req = env._generate_request()
            # 检查是否超出时间
            if env.current_time > simulation_time:
                done = True
            else:
                # 存储 *真实* 的请求对象
                requests.append(req)
        
        print(f"  Generated {len(requests)} requests.")
        
        # 绘制网络拓扑和请求分布图
        if exp_id <= 3:
            plot_network_with_requests(env, requests, rate, exp_id) 

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
            agent = GNNLSTMDDQNAgent(env, device='cuda', sequence_length=lstm_len)
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
            rewards = [results[i]['reward'] for results in all_model_results if i < len(results)]
            latencies = [results[i]['latency'] for results in all_model_results if i < len(results)]
            processing_times = [results[i]['processing_time'] for results in all_model_results if i < len(results)]
            successes = [results[i]['success'] for results in all_model_results if i < len(results)]
            
            if not rewards:  # 如果没有数据，跳过
                continue
                
            # 计算平均值
            avg_reward = np.mean(rewards)
            avg_latency = np.mean(latencies) if latencies else 0
            avg_processing_time = np.mean(processing_times) if processing_times else 0
            success_rate = np.mean(successes) if successes else 0
            
            # 保存平均结果
            avg_performance.append({
                'request_idx': i,
                'reward': avg_reward,
                'latency': avg_latency,
                'processing_time': avg_processing_time,
                'success': success_rate
            })
        
        if avg_performance:  # 只有有数据时才添加
            avg_results[model_name] = avg_performance
    
    # === 简化：按三大服务类别分析 ===
    print("Analyzing performance by service category...")
    
    # 三大服务类别
    service_categories = {
        "ultra_low_latency": ["urgent_braking", "traffic_control"],
        "low_latency": ["collision_avoidance", "sensor_sharing"], 
        "high_latency_tolerance": ["hd_map_update", "infotainment"]
    }
    
    category_summary = {}
    for model_name in avg_results.keys():
        category_summary[model_name] = {}
        for category in service_categories.keys():
            category_summary[model_name][category] = {
                'success_rates': [],
                'latencies': [],
                'count': 0
            }
    
    # 收集每个实验的数据
    for exp_id in range(num_experiments):
        exp_requests = all_experiment_results[exp_id]['requests']
        exp_results = all_experiment_results[exp_id]['model_results']
        
        for model_name, model_performance in exp_results.items():
            for i, request in enumerate(exp_requests):
                if i < skip_first_requests:
                    continue
                    
                # 确定请求所属的类别
                req_category = None
                for category, service_types in service_categories.items():
                    if request['type'] in service_types:
                        req_category = category
                        break
                
                if req_category is None:  # 如果没有匹配的类别，跳过
                    continue
                    
                performance_data = model_performance[i]
                
                # 只统计成功的请求的延迟
                if performance_data['success']:
                    category_summary[model_name][req_category]['latencies'].append(performance_data['latency'])
                
                category_summary[model_name][req_category]['success_rates'].append(performance_data['success'])
                category_summary[model_name][req_category]['count'] += 1
    
    # 计算平均指标
    final_category_summary = {}
    for model_name in category_summary.keys():
        final_category_summary[model_name] = {}
        for category in service_categories.keys():
            data = category_summary[model_name][category]
            if data['count'] > 0:
                final_category_summary[model_name][category] = {
                    'success_rate': np.mean(data['success_rates']),
                    'avg_latency': np.mean(data['latencies']) if data['latencies'] else 0,
                    'request_count': data['count']
                }
    
    # 绘制三大类别的成功率对比图
    plt.figure(figsize=(12, 8))
    categories = list(service_categories.keys())
    bar_width = 0.15
    x = np.arange(len(categories))
    
    for idx, model_name in enumerate(final_category_summary.keys()):
        success_rates = []
        for category in categories:
            if category in final_category_summary[model_name]:
                success_rates.append(final_category_summary[model_name][category]['success_rate'])
            else:
                success_rates.append(0)
        
        plt.bar(x + idx * bar_width, success_rates, bar_width, label=model_name)
    
    plt.title(f"Success Rate by Service Category (Rate={rate}/s)")
    plt.xlabel("Service Category")
    plt.ylabel("Success Rate")
    plt.xticks(x + bar_width * (len(final_category_summary) - 1) / 2, categories)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{results_folder}/rate{rate}/model_comparison/per_request/success_rate_by_category.png", dpi=300)
    plt.close()
    
    # 绘制三大类别的平均延迟对比图
    plt.figure(figsize=(12, 8))
    
    for idx, model_name in enumerate(final_category_summary.keys()):
        latencies = []
        for category in categories:
            if category in final_category_summary[model_name] and final_category_summary[model_name][category]['avg_latency'] > 0:
                latencies.append(final_category_summary[model_name][category]['avg_latency'])
            else:
                latencies.append(0)
        
        plt.bar(x + idx * bar_width, latencies, bar_width, label=model_name)
    
    plt.title(f"Average Latency by Service Category (Rate={rate}/s)")
    plt.xlabel("Service Category")
    plt.ylabel("Latency (ms)")
    plt.xticks(x + bar_width * (len(final_category_summary) - 1) / 2, categories)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{results_folder}/rate{rate}/model_comparison/per_request/latency_by_category.png", dpi=300)
    plt.close()
    
    # 保存类别分析数据
    category_data = []
    for model_name in final_category_summary.keys():
        for category in categories:
            if category in final_category_summary[model_name]:
                data = final_category_summary[model_name][category]
                category_data.append({
                    'Model': model_name,
                    'Service_Category': category,
                    'Success_Rate': data['success_rate'],
                    'Avg_Latency_ms': data['avg_latency'],
                    'Request_Count': data['request_count']
                })
    
    category_df = pd.DataFrame(category_data)
    category_df.to_csv(f"{results_folder}/rate{rate}/model_comparison/per_request/category_analysis.csv", index=False)

    # 修复：绘制移动平均图 - 自适应窗口大小
    colors = ['g-', 'r-', 'c-', 'm-', 'y-', 'k-']
    
    # 奖励对比图（移动平均）
    plt.figure(figsize=(14, 8))
    for idx, (model_name, performance) in enumerate(avg_results.items()):
        if not performance:  # 跳过空数据
            continue
            
        rewards = [req['reward'] for req in performance]
        # 自适应窗口大小
        window_size = min(1000, len(rewards) // 10)  # 最大1000，最小为数据长度的1/10
        if window_size < 10:  # 如果数据太少，使用较小窗口
            window_size = min(5, len(rewards))
        
        if len(rewards) > window_size:
            moving_avg = pd.Series(rewards).rolling(window_size, min_periods=1).mean()
            # 绘制所有数据点
            plt.plot(np.arange(len(rewards)), moving_avg, 
                     colors[idx % len(colors)], linewidth=2.5, label=model_name)
    
    plt.title(f"Moving Average Reward Comparison (Rate={rate}/s, {num_experiments} Experiments)")
    plt.xlabel("Request Index")
    plt.ylabel(f"Reward (Moving Avg, Window={window_size})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{results_folder}/rate{rate}/model_comparison/per_request/moving_avg_reward_comparison.png", dpi=300)
    plt.close()
    
    # 延迟对比图（移动平均）
    plt.figure(figsize=(14, 8))
    for idx, (model_name, performance) in enumerate(avg_results.items()):
        if not performance:
            continue
            
        latencies = [req['latency'] for req in performance]
        window_size = min(1000, len(latencies) // 10)
        if window_size < 10:
            window_size = min(5, len(latencies))
        
        if len(latencies) > window_size:
            moving_avg = pd.Series(latencies).rolling(window_size, min_periods=1).mean()
            plt.plot(np.arange(len(latencies)), moving_avg, 
                     colors[idx % len(colors)], linewidth=2.5, label=model_name)
    
    plt.title(f"Moving Average Latency Comparison (Rate={rate}/s, {num_experiments} Experiments)")
    plt.xlabel("Request Index")
    plt.ylabel(f"Latency (ms) (Moving Avg, Window={window_size})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{results_folder}/rate{rate}/model_comparison/per_request/moving_avg_latency_comparison.png", dpi=300)
    plt.close()
    
    # 处理时间对比图（移动平均）
    plt.figure(figsize=(14, 8))
    for idx, (model_name, performance) in enumerate(avg_results.items()):
        if not performance:
            continue
            
        processing_times = [req['processing_time'] * 1000 for req in performance]  # 转换为毫秒
        window_size = min(1000, len(processing_times) // 10)
        if window_size < 10:
            window_size = min(5, len(processing_times))
        
        if len(processing_times) > window_size:
            moving_avg = pd.Series(processing_times).rolling(window_size, min_periods=1).mean()
            plt.plot(np.arange(len(processing_times)), moving_avg, 
                     colors[idx % len(colors)], linewidth=2.5, label=model_name)
    
    plt.title(f"Moving Average Processing Time Comparison (Rate={rate}/s, {num_experiments} Experiments)")
    plt.xlabel("Request Index")
    plt.ylabel(f"Processing Time (ms) (Moving Avg, Window={window_size})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{results_folder}/rate{rate}/model_comparison/per_request/moving_avg_processing_time_comparison.png", dpi=300)
    plt.close()
    
    # 成功率对比图（移动平均）
    plt.figure(figsize=(14, 8))
    for idx, (model_name, performance) in enumerate(avg_results.items()):
        if not performance:
            continue
            
        success_rates = [req['success'] for req in performance]
        window_size = min(1000, len(success_rates) // 10)
        if window_size < 10:
            window_size = min(5, len(success_rates))
        
        if len(success_rates) > window_size:
            moving_avg = pd.Series(success_rates).rolling(window_size, min_periods=1).mean()
            plt.plot(np.arange(len(success_rates)), moving_avg, 
                     colors[idx % len(colors)], linewidth=2.5, label=model_name)
    
    plt.title(f"Moving Average Success Rate Comparison (Rate={rate}/s, {num_experiments} Experiments)")
    plt.xlabel("Request Index")
    plt.ylabel(f"Success Rate (Moving Avg, Window={window_size})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{results_folder}/rate{rate}/model_comparison/per_request/moving_avg_success_rate_comparison.png", dpi=300)
    plt.close()
    
    # 保存平均结果数据
    for model_name, performance in avg_results.items():
        if performance:  # 只有有数据时才保存
            df = pd.DataFrame(performance)
            safe_model_name = model_name.replace('+', '_').replace('(', '_').replace(')', '_')
            df.to_csv(f"{results_folder}/rate{rate}/model_comparison/per_request/{safe_model_name}_avg_request_performance.csv", index=False)
    
    # 保存最后一次实验的请求序列
    if all_experiment_results:
        last_requests = all_experiment_results[num_experiments-1]['requests']
        pd.DataFrame(last_requests).to_csv(f"{results_folder}/rate{rate}/model_comparison/per_request/request_sequence.csv", index=False)
    
    return avg_results

def test_performance_on_requests(env, agent, requests, agent_name):
    """在给定的请求序列上测试模型性能"""
    # 重置环境
    state = env.reset()
    if hasattr(agent, 'reset_episode'):
        agent.reset_episode()
    
    performance = []
    
    for i, request in enumerate(tqdm(requests, desc=f"Testing {agent_name}")):
        # 设置当前请求
        env.current_time = request['timestamp']
        env.current_request = request
        
        # 获取当前状态
        state = env._get_state()
        
        # 选择动作并记录处理时间
        start_time = time.time()
        action = agent.get_action(state, epsilon=0.01)
        end_time = time.time()
        processing_time = end_time - start_time
        
        # 执行动作
        _, reward, done, metrics = env.step(action, skip_generation=True)
        
        # 记录性能指标
        performance.append({
            'request_idx': i,
            'time': request['timestamp'],
            'position_x': request['position'][0],
            'position_y': request['position'][1],
            'compute_demand': request['compute_demand'],
            'max_latency': request['max_latency'],
            'type': request['type'],
            'action': action,
            'reward': reward,
            'latency': metrics.get('last_latency', 0),
            'success': metrics.get('last_success', False),
            'used_cloud': metrics.get('used_cloud', False),
            'processing_time': processing_time
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
        metrics_df = compare_models_for_rate(rate, episode=n_episode)
        
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
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f"{results_folder}/performance_summary.csv", index=False)

        # 绘制所有rate的平均延迟对比
        plt.figure(figsize=(14, 8))
        sns.barplot(x='rate', y='Avg Latency (ms)', hue='Model', data=summary_df)
        plt.title("Average Latency Comparison Across Rates")
        plt.xlabel("Request Rate (requests/s)")
        plt.ylabel("Average Latency (ms)")
        plt.legend(title='Model')
        plt.tight_layout()
        plt.savefig(f"{results_folder}/average_latency_across_rates.png", dpi=300)
        plt.close()

        # 绘制所有rate的处理时间对比
        plt.figure(figsize=(14, 8))
        sns.barplot(x='rate', y='Avg Reward per Request', hue='Model', data=summary_df)
        plt.title("Average Reward Comparison Across Rates")
        plt.xlabel("Request Rate (requests/s)")
        plt.ylabel("Average Reward")
        plt.legend(title='Model')
        plt.tight_layout()
        plt.savefig(f"{results_folder}/average_reward_across_rates.png", dpi=300)
        plt.close()

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