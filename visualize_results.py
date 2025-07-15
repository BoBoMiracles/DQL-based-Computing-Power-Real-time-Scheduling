import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from simulator import ComputingNetworkSimulator
from gnn_dqn_agent import GNNAgent, StateTransformer
from gnn_lstm_dqn_agent import LSTMDQNAgent
import pandas as pd
from tqdm import tqdm
import os
import random

# 设置绘图风格
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12

def plot_training_history(model_type):
    """绘制训练历史（奖励和损失）"""
    # 加载训练历史
    folder_name = f"{model_type}_model"
    reward_name = f"{model_type}_rewards_history.npy"
    loss_name = f"{model_type}_loss_history.npy"
    reward_path = os.path.join(folder_name, reward_name)
    loss_path = os.path.join(folder_name, loss_name)
    rewards = np.load(reward_path)
    losses = np.load(loss_path)
    
    # 创建结果目录
    os.makedirs("results/training_history", exist_ok=True)
    
    # 绘制奖励曲线
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, label='Episode Reward')
    
    # 计算移动平均
    window_size = 50
    moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    plt.plot(np.arange(window_size-1, len(rewards)), moving_avg, 'r-', label=f'{window_size}-Episode Moving Avg')
    
    plt.title(f"{model_type.upper()} Training Reward History")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/training_history/{model_type}_rewards.png", dpi=300)
    plt.close()
    
    # 损失曲线使用超大窗口的移动平均
    loss_window_size = 10000  # 对70万条数据使用10k窗口
    
    # ==== 标准移动平均损失曲线 ====
    plt.figure(figsize=(12, 6))
    # 计算移动平均 (使用Pandas rolling更高效)
    moving_avg = pd.Series(losses).rolling(loss_window_size, min_periods=1).mean()
    
    plt.plot(moving_avg, 'b-', label=f'Moving Average (Window={loss_window_size})')
    plt.title(f"{model_type.upper()} Training Loss - Moving Average")
    plt.xlabel("Training Step")
    plt.ylabel("Loss (Moving Avg)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/training_history/{model_type}_losses_moving_avg.png", dpi=300)
    plt.close()
    
    # ==== 去除异常值的移动平均损失曲线 ====
    plt.figure(figsize=(12, 6))
    
    # 使用IQR方法识别异常值
    loss_series = pd.Series(losses)
    Q1 = loss_series.quantile(0.25)
    Q3 = loss_series.quantile(0.75)
    IQR = Q3 - Q1
    
    # 创建不含异常值的新序列
    filtered_losses = loss_series[(loss_series >= (Q1 - 1.5 * IQR)) & 
                                 (loss_series <= (Q3 + 1.5 * IQR))]
    
    # 计算过滤后数据的移动平均
    filtered_moving_avg = filtered_losses.rolling(loss_window_size, min_periods=1).mean()
    
    plt.plot(filtered_moving_avg, 'g-', label=f'Filtered Moving Average (Window={loss_window_size})')
    plt.title(f"{model_type.upper()} Training Loss - Filtered Moving Average")
    plt.xlabel("Training Step")
    plt.ylabel("Loss (Filtered Moving Avg)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/training_history/{model_type}_losses_filtered_moving_avg.png", dpi=300)
    plt.close()

def compare_models():
    """比较不同模型的性能"""
    # 加载训练历史
    gnn_rewards = np.load(os.path.join("gnn_model", "gnn_rewards_history.npy"))
    gnn_lstm_rewards = np.load(os.path.join("gnn_lstm_model", "gnn_lstm_rewards_history.npy"))
    
    # 创建结果目录
    os.makedirs("results/model_comparison", exist_ok=True)
    
    # 设置更小的移动平均窗口
    window_size = 20
    
    # 计算移动平均
    gnn_moving_avg = np.convolve(gnn_rewards, np.ones(window_size)/window_size, mode='valid')
    gnn_lstm_moving_avg = np.convolve(gnn_lstm_rewards, np.ones(window_size)/window_size, mode='valid')
    
    # 绘制奖励对比图（突出移动平均）
    plt.figure(figsize=(14, 8))
    
    # 原始奖励数据（半透明虚线）
    plt.plot(gnn_rewards, 'b--', alpha=0.3, label='GNN Raw')
    plt.plot(gnn_lstm_rewards, 'r--', alpha=0.3, label='GNN+LSTM Raw')
    
    # 移动平均线（实线，更粗）
    plt.plot(np.arange(window_size-1, len(gnn_rewards)), gnn_moving_avg, 'b-', linewidth=2.5, label='GNN Moving Avg')
    plt.plot(np.arange(window_size-1, len(gnn_lstm_rewards)), gnn_lstm_moving_avg, 'r-', linewidth=2.5, label='GNN+LSTM Moving Avg')
    
    plt.title("Model Comparison: Training Reward (Highlighting Moving Average)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/model_comparison/reward_comparison_highlight.png", dpi=300)
    plt.close()
    
    # 单独绘制移动平均结果
    plt.figure(figsize=(14, 8))
    
    # 只绘制移动平均线
    plt.plot(np.arange(window_size-1, len(gnn_rewards)), gnn_moving_avg, 'b-', linewidth=2.5, label='GNN Moving Avg')
    plt.plot(np.arange(window_size-1, len(gnn_lstm_rewards)), gnn_lstm_moving_avg, 'r-', linewidth=2.5, label='GNN+LSTM Moving Avg')
    
    plt.title("Model Comparison: Moving Average Reward (Window Size = 20)")
    plt.xlabel("Episode")
    plt.ylabel("Moving Average Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/model_comparison/moving_avg_comparison.png", dpi=300)
    plt.close()
    
    # 加载最终模型进行性能测试
    env = ComputingNetworkSimulator('gurobi_solution_service_sources.csv', 'gurobi_solution_compute_nodes.csv')
    
    # 测试GNN模型
    gnn_agent = GNNAgent(env, device='cpu')
    gnn_agent.policy_net.load_state_dict(torch.load(os.path.join("gnn_model", "gnn_dqn_final.pth"), map_location='cpu'))
    gnn_metrics = test_model(gnn_agent, env)
    
    # 测试GNN+LSTM模型
    gnn_lstm_agent = LSTMDQNAgent(env, device='cpu')
    gnn_lstm_agent.policy_net.load_state_dict(torch.load(os.path.join("gnn_lstm_model", "gnn_lstm_dqn_final.pth"), map_location='cpu'))
    gnn_lstm_metrics = test_model(gnn_lstm_agent, env)
    
    # 测试随机策略
    random_metrics = test_random_policy(env)
    
    # 计算每请求平均奖励
    gnn_avg_reward = gnn_metrics['total_reward'] / gnn_metrics['total_requests']
    gnn_lstm_avg_reward = gnn_lstm_metrics['total_reward'] / gnn_lstm_metrics['total_requests']
    random_avg_reward = random_metrics['total_reward'] / random_metrics['total_requests']
    
    # 创建性能对比数据
    metrics_df = pd.DataFrame({
        'Model': ['GNN', 'GNN+LSTM', 'Random'],
        'Success Rate': [
            gnn_metrics['succeed_requests'] / gnn_metrics['total_requests'],
            gnn_lstm_metrics['succeed_requests'] / gnn_lstm_metrics['total_requests'],
            random_metrics['succeed_requests'] / random_metrics['total_requests']
        ],
        'Avg Reward per Request': [
            gnn_avg_reward,
            gnn_lstm_avg_reward,
            random_avg_reward
        ],
        'Avg Latency (ms)': [
            gnn_metrics['total_latency'] / gnn_metrics['succeed_requests'],
            gnn_lstm_metrics['total_latency'] / gnn_lstm_metrics['succeed_requests'],
            random_metrics['total_latency'] / random_metrics['succeed_requests']
        ],
        'Cloud Usage (%)': [
            gnn_metrics['cloud_requests'] / gnn_metrics['total_requests'] * 100,
            gnn_lstm_metrics['cloud_requests'] / gnn_lstm_metrics['total_requests'] * 100,
            random_metrics['cloud_requests'] / random_metrics['total_requests'] * 100
        ],
        'Resource Utilization (%)': [
            (gnn_metrics['total_processing'] / gnn_metrics['total_requests']) * 100,
            (gnn_lstm_metrics['total_processing'] / gnn_lstm_metrics['total_requests']) * 100,
            (random_metrics['total_processing'] / random_metrics['total_requests']) * 100
        ]
    })
    
    # 绘制性能对比柱状图（3x2布局）
    plt.figure(figsize=(16, 12))
    
    # 成功率
    plt.subplot(3, 2, 1)
    sns.barplot(x='Model', y='Success Rate', data=metrics_df)
    plt.title("Success Rate Comparison")
    plt.ylabel("Success Rate")
    
    # 平均奖励
    plt.subplot(3, 2, 2)
    sns.barplot(x='Model', y='Avg Reward per Request', data=metrics_df)
    plt.title("Average Reward per Request")
    plt.ylabel("Reward")
    
    # 平均延迟
    plt.subplot(3, 2, 3)
    sns.barplot(x='Model', y='Avg Latency (ms)', data=metrics_df)
    plt.title("Average Latency Comparison")
    plt.ylabel("Latency (ms)")
    
    # 云端使用率
    plt.subplot(3, 2, 4)
    sns.barplot(x='Model', y='Cloud Usage (%)', data=metrics_df)
    plt.title("Cloud Usage Comparison")
    plt.ylabel("Cloud Usage (%)")
    
    # 资源利用率
    plt.subplot(3, 2, 5)
    sns.barplot(x='Model', y='Resource Utilization (%)', data=metrics_df)
    plt.title("Resource Utilization Comparison")
    plt.ylabel("Utilization (%)")
    
    plt.tight_layout()
    plt.savefig("results/model_comparison/performance_comparison.png", dpi=300)
    plt.close()
    
    # 单独绘制平均奖励对比图
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='Avg Reward per Request', data=metrics_df, palette='viridis')
    plt.title("Average Reward per Request Comparison")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig("results/model_comparison/avg_reward_comparison.png", dpi=300)
    plt.close()
    
    # 保存性能数据
    metrics_df.to_csv("results/model_comparison/performance_metrics.csv", index=False)
    
    return metrics_df

def test_model(agent, env, num_episodes=10):
    metrics = {
        'total_requests': 0,
        'succeed_requests': 0,
        'cloud_requests': 0,
        'total_latency': 0,
        'total_processing': 0,
        'total_reward': 0,
        'successful_latency': 0,  # 成功请求的延迟总和
        'failed_latency': 0,      # 失败请求的延迟总和
        'success_count': 0        # 成功请求数
    }
    
    for _ in range(num_episodes):
        state = env.reset()
        if isinstance(agent, LSTMDQNAgent):
            agent.reset_hidden_state()
        
        done = False
        while not done:
            action = agent.get_action(state, epsilon=0.01)
            next_state, reward, done, ep_metrics = env.step(action)
            
            # 更新指标
            for key in metrics:
                if key in ep_metrics:
                    metrics[key] += ep_metrics[key]
            
            # 记录成功/失败请求的延迟
            if ep_metrics['last_success']:
                metrics['successful_latency'] += ep_metrics['last_latency']
                metrics['success_count'] += 1
            else:
                metrics['failed_latency'] += ep_metrics['last_latency']
            
            state = next_state
    
    # 计算平均值
    for key in metrics:
        metrics[key] /= num_episodes
    
    # 计算平均成功延迟和平均失败延迟
    if metrics['success_count'] > 0:
        metrics['avg_success_latency'] = metrics['successful_latency'] / metrics['success_count']
    else:
        metrics['avg_success_latency'] = 0
        
    if metrics['total_requests'] - metrics['success_count'] > 0:
        metrics['avg_failed_latency'] = metrics['failed_latency'] / (metrics['total_requests'] - metrics['success_count'])
    else:
        metrics['avg_failed_latency'] = 0
    
    return metrics

def test_random_policy(env, num_episodes=10):
    """测试随机策略性能"""
    metrics = {
        'total_requests': 0,
        'succeed_requests': 0,
        'cloud_requests': 0,
        'total_latency': 0,
        'total_processing': 0,
        'total_reward': 0  # 添加总奖励记录
    }
    
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            # 随机选择合法动作
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                action = 'cloud'
            else:
                action = random.choice(valid_actions)
            
            next_state, reward, done, ep_metrics = env.step(action)
            
            # 更新指标
            for key in metrics:
                if key in ep_metrics:
                    metrics[key] += ep_metrics[key]
            metrics['total_reward'] += reward  # 记录总奖励
            
            state = next_state
    
    # 计算平均值
    for key in metrics:
        metrics[key] /= num_episodes
    
    return metrics

def visualize_request_processing(model_type):
    """可视化请求处理过程"""
    # 创建结果目录
    os.makedirs(f"results/request_processing/{model_type}", exist_ok=True)
    
    # 初始化环境
    env = ComputingNetworkSimulator('gurobi_solution_service_sources.csv', 'gurobi_solution_compute_nodes.csv')
    
    # 加载模型
    if model_type == 'gnn':
        agent = GNNAgent(env, device='cpu')
        model_path = os.path.join("gnn_model", "gnn_dqn_final.pth")
    else:
        agent = LSTMDQNAgent(env, device='cpu')
        model_path = os.path.join("gnn_lstm_model", "gnn_lstm_dqn_final.pth")
    
    agent.policy_net.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    # 重置环境
    state = env.reset()
    if model_type == 'gnn_lstm':
        agent.reset_hidden_state()
    
    # 设置可视化
    env.setup_visualization()
    
    # 运行一个episode并记录
    request_history = []
    done = False
    
    while not done:
        action = agent.get_action(state, epsilon=0.01)
        next_state, reward, done, metrics = env.step(action)
        
        # 记录请求信息
        if env.current_request:
            req = env.current_request.copy()
            req['action'] = action
            req['reward'] = reward
            request_history.append(req)
        
        # 更新可视化
        env.update_visualization()
        
        state = next_state
    
    # 保存请求处理数据
    df = pd.DataFrame(request_history)
    df.to_csv(f"results/request_processing/{model_type}/request_processing.csv", index=False)
    
    # 绘制请求处理分析图
    plt.figure(figsize=(14, 10))
    
    # 请求位置分布
    plt.subplot(2, 2, 1)
    positions = np.array([r['position'] for r in request_history])
    plt.scatter(positions[:, 0], positions[:, 1], c=df['compute_demand'], cmap='viridis')
    plt.colorbar(label='Compute Demand')
    plt.title("Request Position Distribution")
    plt.xlabel("X")
    plt.ylabel("Y")
    
    # 计算需求分布
    plt.subplot(2, 2, 2)
    sns.histplot(df['compute_demand'], bins=20, kde=True)
    plt.title("Compute Demand Distribution")
    plt.xlabel("Compute Demand")
    
    # 动作选择分布
    plt.subplot(2, 2, 3)
    action_counts = df['action'].value_counts()
    plt.pie(action_counts, labels=action_counts.index, autopct='%1.1f%%')
    plt.title("Action Selection Distribution")
    
    # 奖励分布
    plt.subplot(2, 2, 4)
    sns.histplot(df['reward'], bins=20, kde=True)
    plt.title("Reward Distribution")
    plt.xlabel("Reward")
    
    plt.tight_layout()
    plt.savefig(f"results/request_processing/{model_type}/request_analysis.png", dpi=300)
    plt.close()
    
    # 绘制延迟与计算需求的关系
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='compute_demand', y='max_latency', data=df, hue='action', palette='viridis')
    plt.title("Latency vs Compute Demand")
    plt.xlabel("Compute Demand")
    plt.ylabel("Max Allowed Latency")
    plt.legend(title='Action')
    plt.tight_layout()
    plt.savefig(f"results/request_processing/{model_type}/latency_vs_demand.png", dpi=300)
    plt.close()

def visualize_resource_utilization(model_type):
    """可视化资源利用率"""
    # 创建结果目录
    os.makedirs(f"results/resource_utilization/{model_type}", exist_ok=True)
    
    # 初始化环境
    env = ComputingNetworkSimulator('gurobi_solution_service_sources.csv', 'gurobi_solution_compute_nodes.csv')
    
    # 加载模型
    if model_type == 'gnn':
        agent = GNNAgent(env, device='cpu')
        model_path = os.path.join("gnn_model", "gnn_dqn_final.pth")
    else:
        agent = LSTMDQNAgent(env, device='cpu')
        model_path = os.path.join("gnn_lstm_model", "gnn_lstm_dqn_final.pth")
    
    agent.policy_net.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    # 运行测试
    state = env.reset()
    if model_type == 'gnn_lstm':
        agent.reset_hidden_state()
    
    # 收集资源利用率数据
    room_utilizations = {}
    for room_id, room in env.nodes['rooms'].items():
        # 只记录有额外算力的机房
        if room['max_compute'] > 0:
            room_utilizations[room_id] = []
    
    cloud_usage = []
    
    done = False
    while not done:
        action = agent.get_action(state, epsilon=0.01)
        next_state, reward, done, metrics = env.step(action)
        
        # 记录资源利用率
        for room_id, room in env.nodes['rooms'].items():
            # 只记录有额外算力的机房
            if room_id in room_utilizations:
                # 避免除以零错误
                if room['max_compute'] > 0:
                    utilization = 1 - (room['compute'] / room['max_compute'])
                else:
                    utilization = 0
                room_utilizations[room_id].append(utilization)
        
        # 记录云端使用情况
        cloud_usage.append(1 if 'cloud' in action else 0)
        
        state = next_state
    
    # 绘制资源利用率曲线
    plt.figure(figsize=(14, 8))
    for room_id, utils in room_utilizations.items():
        plt.plot(utils, label=f"Room {room_id}")
    
    plt.title(f"{model_type.upper()} Resource Utilization Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Utilization (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/resource_utilization/{model_type}/room_utilization.png", dpi=300)
    plt.close()
    
    # 绘制云端使用率
    plt.figure(figsize=(10, 6))
    plt.plot(cloud_usage, 'r-', label='Cloud Usage')
    plt.title(f"{model_type.upper()} Cloud Usage Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Cloud Usage (1=Used, 0=Not Used)")
    plt.tight_layout()
    plt.savefig(f"results/resource_utilization/{model_type}/cloud_usage.png", dpi=300)
    plt.close()
    
    # 保存资源利用率数据
    util_df = pd.DataFrame(room_utilizations)
    util_df['cloud_usage'] = cloud_usage
    util_df.to_csv(f"results/resource_utilization/{model_type}/resource_utilization.csv", index=False)

def compare_requests_performance():
    """比较两个模型在同一批请求序列上的性能表现"""
    # 创建结果目录
    os.makedirs("results/model_comparison/per_request", exist_ok=True)
    
    # 设置随机种子以确保同一批请求
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 初始化环境
    env = ComputingNetworkSimulator('gurobi_solution_service_sources.csv', 'gurobi_solution_compute_nodes.csv')
    
    # 加载GNN模型
    gnn_agent = GNNAgent(env, device='cpu')
    gnn_agent.policy_net.load_state_dict(torch.load(os.path.join("gnn_model", "gnn_dqn_final.pth"), map_location='cpu'))
    
    # 加载GNN+LSTM模型
    gnn_lstm_agent = LSTMDQNAgent(env, device='cpu')
    gnn_lstm_agent.policy_net.load_state_dict(torch.load(os.path.join("gnn_lstm_model", "gnn_lstm_dqn_final.pth"), map_location='cpu'))
    
    # 生成相同的请求序列（使用新的请求生成逻辑）
    requests = []
    current_time = 0
    request_counter = 0
    request_history = []
    
    # 获取所有机房位置
    room_positions = [room['position'] for room in env.nodes['rooms'].values()]
    
    while current_time < 3600:  # 1小时
        inter_arrival = np.random.exponential(1/env.request_rate)
        current_time += inter_arrival
        request_counter += 1
        
        # 70%的请求在机房附近生成（热点区域）
        if random.random() < 0.7 and room_positions:
            # 随机选择一个机房作为热点中心
            center_room = random.choice(room_positions)
            
            # 在机房周围生成请求（正态分布）
            position = (
                np.clip(np.random.normal(center_room[0], 10), 0, 100),
                np.clip(np.random.normal(center_room[1], 10), 0, 100)
            )
        else:
            # 30%的请求在随机区域生成
            position = (
                np.random.uniform(0, 100),
                np.random.uniform(0, 100)
            )
        
        # 添加位置依赖性：新请求位置靠近前一个请求的概率更高
        if request_history and random.random() < 0.6:
            last_position = request_history[-1]['position']
            position = (
                np.clip(last_position[0] + np.random.normal(0, 5), 0, 100),
                np.clip(last_position[1] + np.random.normal(0, 5), 0, 100)
            )
        
        # 根据请求类型分配不同的特性
        request_type = np.random.choice(
            ['safety-critical', 'infotainment', 'adas'], 
            p=[0.2, 0.5, 0.3]  # 出现概率分布
        )
        
        # 不同请求类型的特性参数
        if request_type == 'safety-critical':
            # 安全攸关请求：高实时性要求，中等算力需求，中等处理时间
            compute_demand = np.clip(np.random.normal(15, 2), 5, 20)  # 平均15，标准差2
            max_latency = np.random.choice([15, 20, 25])  # 严格的延迟要求
            base_process = 2  # 基础处理时间
            process_time = np.clip(np.random.normal(base_process, 0.05), 0.5, 4)  # 2±0.05秒
        elif request_type == 'infotainment':
            # 信息娱乐请求：较低的实时性要求，中等算力需求，较长处理时间
            compute_demand = np.clip(np.random.normal(10, 3), 5, 20)  # 平均10，标准差3
            max_latency = np.random.choice([30, 35, 40])  # 较宽松的延迟要求
            base_process = 20  # 基础处理时间
            process_time = np.clip(np.random.normal(base_process, 3), 5, 50)  # 20±3秒
        else:  # adas (Advanced Driver Assistance Systems)
            # 高级驾驶辅助：中等实时性要求，高算力需求，中等处理时间
            compute_demand = np.clip(np.random.normal(18, 1.5), 5, 20)  # 平均18，标准差1.5
            max_latency = np.random.choice([20, 25, 30])  # 中等延迟要求
            base_process = 10  # 基础处理时间
            process_time = np.clip(np.random.normal(base_process, 1), 2, 30)  # 10±1秒
        
        # 机房附近的请求更可能是高计算需求类型
        if room_positions:
            min_dist = min([env._calculate_distance(position, room) for room in room_positions])
            if min_dist < 15:  # 在机房附近
                # 增加高计算需求请求的概率
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
    
    # 测试两个模型在相同请求序列上的表现
    gnn_performance = test_performance_on_requests(env, gnn_agent, requests, 'GNN')
    gnn_lstm_performance = test_performance_on_requests(env, gnn_lstm_agent, requests, 'GNN+LSTM')
    
    # 创建DataFrame便于绘图
    df_gnn = pd.DataFrame(gnn_performance)
    df_gnn_lstm = pd.DataFrame(gnn_lstm_performance)
    
    # 设置更大的移动平均窗口
    large_window_size = 1000  # 窗口增大到1000
    
    # 奖励对比图（原始数据+移动平均）
    plt.figure(figsize=(14, 8))
    # 原始数据 - 淡色虚线
    plt.plot(df_gnn['reward'], 'b--', alpha=0.2, linewidth=0.8, label='GNN Raw')
    plt.plot(df_gnn_lstm['reward'], 'r--', alpha=0.2, linewidth=0.8, label='GNN+LSTM Raw')
    # 移动平均 - 粗实线
    plt.plot(df_gnn['reward'].rolling(large_window_size).mean(), 'b-', linewidth=2.5, 
             label=f'GNN Moving Avg (Win={large_window_size})')
    plt.plot(df_gnn_lstm['reward'].rolling(large_window_size).mean(), 'r-', linewidth=2.5, 
             label=f'GNN+LSTM Moving Avg (Win={large_window_size})')
    plt.title("Reward Comparison per Request (Highlighting Moving Average)")
    plt.xlabel("Request Index")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/model_comparison/per_request/request_reward_highlight.png", dpi=300)
    plt.close()
    
    # 奖励对比图（仅移动平均）
    plt.figure(figsize=(14, 8))
    plt.plot(df_gnn['reward'].rolling(large_window_size).mean(), 'b-', linewidth=2.5, 
             label=f'GNN Moving Avg (Win={large_window_size})')
    plt.plot(df_gnn_lstm['reward'].rolling(large_window_size).mean(), 'r-', linewidth=2.5, 
             label=f'GNN+LSTM Moving Avg (Win={large_window_size})')
    plt.title("Moving Average Reward Comparison per Request")
    plt.xlabel("Request Index")
    plt.ylabel("Reward (Moving Avg)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/model_comparison/per_request/request_reward_moving_avg.png", dpi=300)
    plt.close()
    
    # 延迟对比图（原始数据+移动平均）
    plt.figure(figsize=(14, 8))
    # 原始数据 - 淡色虚线
    plt.plot(df_gnn['latency'], 'b--', alpha=0.2, linewidth=0.8, label='GNN Raw')
    plt.plot(df_gnn_lstm['latency'], 'r--', alpha=0.2, linewidth=0.8, label='GNN+LSTM Raw')
    # 移动平均 - 粗实线
    plt.plot(df_gnn['latency'].rolling(large_window_size).mean(), 'b-', linewidth=2.5, 
             label=f'GNN Moving Avg (Win={large_window_size})')
    plt.plot(df_gnn_lstm['latency'].rolling(large_window_size).mean(), 'r-', linewidth=2.5, 
             label=f'GNN+LSTM Moving Avg (Win={large_window_size})')
    # 添加最大延迟线
    max_latencies = [request['max_latency'] for request in requests]
    # plt.plot(max_latencies, 'g--', alpha=0.5, label='Max Latency')
    plt.title("Latency Comparison per Request (Highlighting Moving Average)")
    plt.xlabel("Request Index")
    plt.ylabel("Latency (ms)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/model_comparison/per_request/request_latency_highlight.png", dpi=300)
    plt.close()
    
    # 延迟对比图（仅移动平均）
    plt.figure(figsize=(14, 8))
    plt.plot(df_gnn['latency'].rolling(large_window_size).mean(), 'b-', linewidth=2.5, 
             label=f'GNN Moving Avg (Win={large_window_size})')
    plt.plot(df_gnn_lstm['latency'].rolling(large_window_size).mean(), 'r-', linewidth=2.5, 
             label=f'GNN+LSTM Moving Avg (Win={large_window_size})')
    # plt.plot(max_latencies, 'g--', alpha=0.5, label='Max Latency')
    plt.title("Moving Average Latency Comparison per Request")
    plt.xlabel("Request Index")
    plt.ylabel("Latency (ms) (Moving Avg)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/model_comparison/per_request/request_latency_moving_avg.png", dpi=300)
    plt.close()
    
    # 绘制请求处理位置对比
    fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    
    # GNN模型
    ax1 = axes[0]
    for i, request in enumerate(requests):
        color = 'green' if df_gnn.loc[i, 'latency'] <= request['max_latency'] else 'red'
        ax1.scatter(request['position'][0], request['position'][1], color=color, alpha=0.5)
    ax1.set_title("GNN: Request Processing (Green=Success, Red=Failure)")
    ax1.set_ylabel("Y Coordinate")
    
    # GNN+LSTM模型
    ax2 = axes[1]
    for i, request in enumerate(requests):
        color = 'green' if df_gnn_lstm.loc[i, 'latency'] <= request['max_latency'] else 'red'
        ax2.scatter(request['position'][0], request['position'][1], color=color, alpha=0.5)
    ax2.set_title("GNN+LSTM: Request Processing (Green=Success, Red=Failure)")
    ax2.set_xlabel("X Coordinate")
    ax2.set_ylabel("Y Coordinate")
    
    plt.tight_layout()
    plt.savefig("results/model_comparison/per_request/request_position_comparison.png", dpi=300)
    plt.close()
    
    # 绘制请求类型分布
    plt.figure(figsize=(10, 6))
    request_types = [r['type'] for r in requests]
    sns.countplot(x=request_types, order=['safety-critical', 'infotainment', 'adas'])
    plt.title("Request Type Distribution")
    plt.xlabel("Request Type")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("results/model_comparison/per_request/request_type_distribution.png", dpi=300)
    plt.close()
    
    # 按请求类型分析成功率
    type_success = {
        'safety-critical': {'gnn': 0, 'gnn_lstm': 0, 'total': 0},
        'infotainment': {'gnn': 0, 'gnn_lstm': 0, 'total': 0},
        'adas': {'gnn': 0, 'gnn_lstm': 0, 'total': 0}
    }
    
    for i, request in enumerate(requests):
        req_type = request['type']
        type_success[req_type]['total'] += 1
        
        if df_gnn.loc[i, 'latency'] <= request['max_latency']:
            type_success[req_type]['gnn'] += 1
        
        if df_gnn_lstm.loc[i, 'latency'] <= request['max_latency']:
            type_success[req_type]['gnn_lstm'] += 1
    
    # 绘制按请求类型的成功率
    plt.figure(figsize=(12, 6))
    types = list(type_success.keys())
    gnn_success = [type_success[t]['gnn']/type_success[t]['total'] for t in types]
    gnn_lstm_success = [type_success[t]['gnn_lstm']/type_success[t]['total'] for t in types]
    
    x = np.arange(len(types))
    width = 0.35
    
    plt.bar(x - width/2, gnn_success, width, label='GNN')
    plt.bar(x + width/2, gnn_lstm_success, width, label='GNN+LSTM')
    
    plt.title("Success Rate by Request Type")
    plt.xlabel("Request Type")
    plt.ylabel("Success Rate")
    plt.xticks(x, types)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/model_comparison/per_request/success_rate_by_type.png", dpi=300)
    plt.close()
    
    # 保存数据
    df_gnn.to_csv("results/model_comparison/per_request/gnn_request_performance.csv", index=False)
    df_gnn_lstm.to_csv("results/model_comparison/per_request/gnn_lstm_request_performance.csv", index=False)
    
    # 保存请求序列
    pd.DataFrame(requests).to_csv("results/model_comparison/per_request/request_sequence.csv", index=False)
    
    return df_gnn, df_gnn_lstm

    
def test_performance_on_requests(env, agent, requests, agent_name):
    """在给定的请求序列上测试模型性能"""
    # 重置环境
    state = env.reset()
    if agent_name == 'GNN+LSTM':
        agent.reset_hidden_state()
    
    performance = []
    
    for i, request in enumerate(requests):
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
        nearest_bs, _ = env._find_nearest_bs(request['position'])
        home_room_id = nearest_bs['room_id']
        env.current_request['home_room'] = home_room_id
        
        # 获取当前状态
        state = env._get_state()
        
        # 选择动作
        action = agent.get_action(state, epsilon=0.01)  # 使用小量探索
        
        # 执行动作
        _, reward, _, metrics = env.step(action)
        
        # 记录性能指标
        performance.append({
            'request_idx': i,
            'time': request['time'],
            'position_x': request['position'][0],
            'position_y': request['position'][1],
            'compute_demand': request['compute_demand'],
            'max_latency': request['max_latency'],
            'action': action,
            'reward': reward,
            'latency': metrics['last_latency'],
            'success': metrics['last_success'],
            'used_cloud': metrics['used_cloud']
        })
        
        # 更新环境状态以反映处理完成
        env._process_pending_events()
    
    return performance

def main():
    """主函数：执行所有可视化任务"""
    # 确保结果目录存在
    os.makedirs("results", exist_ok=True)
    
    print("Visualizing training history...")
    plot_training_history('gnn')
    plot_training_history('gnn_lstm')
    
    print("Comparing models...")
    compare_models()
    
    print("Comparing per-request performance...")
    compare_requests_performance()  
    
    print("Visualizing request processing...")
    visualize_request_processing('gnn')
    visualize_request_processing('gnn_lstm')
    
    print("Visualizing resource utilization...")
    visualize_resource_utilization('gnn')
    visualize_resource_utilization('gnn_lstm')
    
    print("All visualizations completed. Results saved to 'results' directory.")


if __name__ == "__main__":
    main()