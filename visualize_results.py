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
    
    # 绘制损失曲线
    plt.figure(figsize=(12, 6))
    plt.plot(losses, label='Training Loss')
    
    # 计算移动平均
    window_size = 100
    moving_avg = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
    plt.plot(np.arange(window_size-1, len(losses)), moving_avg, 'r-', label=f'{window_size}-Step Moving Avg')
    
    plt.title(f"{model_type.upper()} Training Loss History")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/training_history/{model_type}_losses.png", dpi=300)
    plt.close()

def compare_models():
    """比较不同模型的性能"""
    # 加载训练历史
    gnn_rewards = np.load(os.path.join("gnn_model", "gnn_rewards_history.npy"))
    gnn_lstm_rewards = np.load(os.path.join("gnn_lstm_model", "gnn_lstm_rewards_history.npy"))
    
    # 创建结果目录
    os.makedirs("results/model_comparison", exist_ok=True)
    
    # 绘制奖励对比
    plt.figure(figsize=(12, 6))
    plt.plot(gnn_rewards, label='GNN')
    plt.plot(gnn_lstm_rewards, label='GNN+LSTM')
    
    # 计算移动平均
    window_size = 50
    gnn_moving_avg = np.convolve(gnn_rewards, np.ones(window_size)/window_size, mode='valid')
    gnn_lstm_moving_avg = np.convolve(gnn_lstm_rewards, np.ones(window_size)/window_size, mode='valid')
    
    plt.plot(np.arange(window_size-1, len(gnn_rewards)), gnn_moving_avg, 'b-', linewidth=2)
    plt.plot(np.arange(window_size-1, len(gnn_lstm_rewards)), gnn_lstm_moving_avg, 'r-', linewidth=2)
    
    plt.title("Model Comparison: Training Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/model_comparison/reward_comparison.png", dpi=300)
    plt.close()
    
    # 加载最终模型进行性能测试
    env = ComputingNetworkSimulator('base_stations.csv', 'rooms.csv')
    
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
    
    # 创建性能对比数据
    metrics_df = pd.DataFrame({
        'Model': ['GNN', 'GNN+LSTM', 'Random'],
        'Success Rate': [
            gnn_metrics['succeed_requests'] / gnn_metrics['total_requests'],
            gnn_lstm_metrics['succeed_requests'] / gnn_lstm_metrics['total_requests'],
            random_metrics['succeed_requests'] / random_metrics['total_requests']
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
    
    # 绘制性能对比柱状图
    plt.figure(figsize=(14, 10))
    
    # 成功率
    plt.subplot(2, 2, 1)
    sns.barplot(x='Model', y='Success Rate', data=metrics_df)
    plt.title("Success Rate Comparison")
    plt.ylabel("Success Rate")
    
    # 平均延迟
    plt.subplot(2, 2, 2)
    sns.barplot(x='Model', y='Avg Latency (ms)', data=metrics_df)
    plt.title("Average Latency Comparison")
    plt.ylabel("Latency (ms)")
    
    # 云端使用率
    plt.subplot(2, 2, 3)
    sns.barplot(x='Model', y='Cloud Usage (%)', data=metrics_df)
    plt.title("Cloud Usage Comparison")
    plt.ylabel("Cloud Usage (%)")
    
    # 资源利用率
    plt.subplot(2, 2, 4)
    sns.barplot(x='Model', y='Resource Utilization (%)', data=metrics_df)
    plt.title("Resource Utilization Comparison")
    plt.ylabel("Utilization (%)")
    
    plt.tight_layout()
    plt.savefig("results/model_comparison/performance_comparison.png", dpi=300)
    plt.close()
    
    # 保存性能数据
    metrics_df.to_csv("results/model_comparison/performance_metrics.csv", index=False)
    
    return metrics_df

def test_model(agent, env, num_episodes=10):
    """测试模型性能"""
    metrics = {
        'total_requests': 0,
        'succeed_requests': 0,
        'cloud_requests': 0,
        'total_latency': 0,
        'total_processing': 0
    }
    
    for _ in range(num_episodes):
        state = env.reset()
        if isinstance(agent, LSTMDQNAgent):
            agent.reset_hidden_state()
        
        done = False
        while not done:
            action = agent.get_action(state, epsilon=0.01)  # 使用小量探索
            next_state, reward, done, ep_metrics = env.step(action)
            
            # 更新指标
            for key in metrics:
                metrics[key] += ep_metrics[key]
            
            state = next_state
    
    # 计算平均值
    for key in metrics:
        metrics[key] /= num_episodes
    
    return metrics

def test_random_policy(env, num_episodes=10):
    """测试随机策略性能"""
    metrics = {
        'total_requests': 0,
        'succeed_requests': 0,
        'cloud_requests': 0,
        'total_latency': 0,
        'total_processing': 0
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
                metrics[key] += ep_metrics[key]
            
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
    env = ComputingNetworkSimulator('base_stations.csv', 'rooms.csv')
    
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
    env = ComputingNetworkSimulator('base_stations.csv', 'rooms.csv')
    
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

def main():
    """主函数：执行所有可视化任务"""
    # 确保结果目录存在
    os.makedirs("results", exist_ok=True)
    
    print("Visualizing training history...")
    plot_training_history('gnn')
    plot_training_history('gnn_lstm')
    
    print("Comparing models...")
    compare_models()
    
    print("Visualizing request processing...")
    visualize_request_processing('gnn')
    visualize_request_processing('gnn_lstm')
    
    print("Visualizing resource utilization...")
    visualize_resource_utilization('gnn')
    visualize_resource_utilization('gnn_lstm')
    
    print("All visualizations completed. Results saved to 'results' directory.")

if __name__ == "__main__":
    main()