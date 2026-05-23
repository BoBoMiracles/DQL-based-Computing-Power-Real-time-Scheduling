import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pandas as pd
from tqdm import tqdm
import random
import time
import argparse 
from simulator import ComputingNetworkSimulator
from gnn_dqn_agent import GNNAgent, StateTransformer
from gnn_lstm_ddqn_agent import GNNLSTMDDQNAgent
from gnn_transformer_ddqn_agent import GNNTransformerDDQNAgent

# === 全局参数 ===
rates = [1, 5, 8]  # 不同请求率
# rates = [0.1, 0.5]
# 分别定义 LSTM 和 Transformer 的时间窗长度，方便独立控制
lstm_lengths = [3, 5, 10, 20, 30]  
# lstm_lengths = [3, 5]  
transformer_lengths = [3, 5, 10, 20, 30]
# transformer_lengths = [3, 5]
FOLDER_MAPPING = {}  # 将在主函数中填充
# results_folder = 'results_revision_transformer_v5'
base_folder = 'models_revision_v3' # 统一的模型根目录
result_folder = os.path.join('results', 'results_revision_v3')
num_experiments = 10  # 相同请求序列比较的实验次数
n_episode = 20  # 整体评价对比模型的episode
period_count = 2  # 测试时间内的潮汐周期数，需要根据测试时间进行设定
skip_first_requests = max(max(lstm_lengths), max(transformer_lengths))  # 舍弃未启动的请求的处理结果
PLOT_SKIP_REQUESTS = 100

def set_global_seed(seed):
    """统一设置所有随机种子，确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# 根据请求率确定仿真时间
def get_simulation_time(rate):
    if rate <= 1: return 3600
    elif rate >= 8: return 600 
    return 900

# 方案1：扩大颜色选择范围（使用 tab20 调色板，足够区分20条线）
UNIQUE_COLORS = plt.cm.tab20.colors 

# 方案2：语义化颜色映射（长度 -> 颜色）
LENGTH_COLOR_MAP = {
    3:  'tab:red',      # 短视：红色
    5:  'tab:orange',   # 短期：橙色
    10: 'tab:green',    # 中期：绿色
    20: 'tab:blue',     # 长期：蓝色
    30: 'tab:purple',   # 超长：紫色
    # 如果有其他长度，会自动回退到默认色
}

class BaselineAgent:
    """将 Simulator 中的基准策略包装成 Agent 接口"""
    def __init__(self, env, strategy_type):
        self.env = env
        self.strategy_type = strategy_type

    def get_action(self, state, epsilon=0.0):
        if self.strategy_type == 'Random':
            valid = self.env.get_valid_actions()
            return random.choice(valid) if valid else 'cloud'
        elif self.strategy_type == 'LUF':
            # 调用 Simulator 内部的最低利用率优先策略
            action, _ = self.env._lowest_utilization_policy(self.env.current_request)
            return action
        elif self.strategy_type == 'All-Cloud':
            # 全部卸载到云端
            return 'cloud'
        elif self.strategy_type == 'All-Local':
            # 找到离当前请求最近的基站，并返回其所属的本地机房
            nearest_bs, _ = self.env._find_nearest_bs(self.env.current_request['position'])
            return nearest_bs['room_id']
        return 'cloud'

    def reset_episode(self):
        """Baseline 策略通常是无状态的，此处仅为接口兼容"""
        pass

def should_skip_request(request_idx, skip_threshold, current_time, total_time):
    """统一的冷启动判断逻辑"""
    # 基于请求索引跳过
    if request_idx < skip_threshold:
        return True
    return False

# 更新模型加载路径的函数
def get_model_path(model_type, rate, length=None, tidal_flow=False):
    """根据模型类型和潮汐模式返回正确的文件夹路径"""
    tidal_suffix = "_tidal" if tidal_flow else ""
    
    if model_type == 'gnn':
        return os.path.join(base_folder, f'gnn_model_rate{rate}{tidal_suffix}')
    elif model_type == 'gnn_lstm' and length:
        return os.path.join(base_folder, f'gnn_lstm{length}_model_rate{rate}{tidal_suffix}')
    elif model_type == 'gnn_transformer' and length:
        return os.path.join(base_folder, f'gnn_transformer{length}_model_rate{rate}{tidal_suffix}')
    return ''

def get_style_for_model(model_name, length=None, scheme='semantic'):
    """
    根据模型名称和长度获取绘图样式 (color, linestyle, label)
    
    Args:
        model_name (str): 模型名称 (e.g., 'gnn', 'GNN+LSTM(5)', 'All-Cloud')
        length (int): 时间窗长度 (仅用于训练历史绘图)
        scheme (str): 'unique' (方案1: 全不相同) 或 'semantic' (方案2: 相同长度同色)
    """
    # --- 统一解析逻辑 ---
    # 尝试从名称中解析长度
    if length is None and '(' in model_name and ')' in model_name:
        try:
            length = int(model_name.split('(')[1].split(')')[0])
        except:
            length = 0
            
    is_transformer = 'Trans' in model_name or 'transformer' in model_name
    is_lstm = 'LSTM' in model_name or 'lstm' in model_name
    is_gnn = model_name.lower() == 'gnn' or model_name == 'GNN'
    
    # 定义 Baseline 模型及其低对比度/柔和颜色
    baseline_styles = {
        'Random': '#B0B0B0',                   # 浅灰色 (用于随机)
        'All-Cloud': '#B0C4DE',                # 浅灰蓝色 (LightSteelBlue)
        'All-Local': '#DEB887',                # 浅棕褐色 (BurlyWood)
        'Lowest Utilization First': '#A8D08D'  # 浅灰绿色 (Muted Green)
    }
    is_baseline = model_name in baseline_styles
    
    # --- 方案 1: 所有线颜色都不一样 (Unique) ---
    if scheme == 'unique':
        if is_gnn:
            return 'black', '-', model_name
        if is_baseline:
            return baseline_styles[model_name], '-', model_name
            
        # 基于长度和类型的索引计算
        color_idx = 0
        if length:
            color_idx = (length * 2 + (1 if is_transformer else 0)) % len(UNIQUE_COLORS)
        
        return UNIQUE_COLORS[color_idx], '-', model_name

    # --- 方案 2: 相同时间窗同色，实线/虚线区分 (Semantic) ---
    else:
        # 1. 确定颜色
        if is_gnn:
            color = 'black'
        elif is_baseline:
            color = baseline_styles[model_name]
        elif length in LENGTH_COLOR_MAP:
            color = LENGTH_COLOR_MAP[length]
        else:
            color = 'gray' # 未知长度默认灰色
            
        # 2. 确定线型与标签
        if is_transformer:
            linestyle = '--'  # Transformer 用虚线
            label = f'GNN+Trans({length})'
        elif is_lstm:
            linestyle = '-'   # LSTM 用实线
            label = f'GNN+LSTM({length})'
        elif is_baseline:
            # Baseline 使用点划线或实线均可，这里使用实线配合低对比度颜色让其作为背景参考
            linestyle = '-'
            label = model_name
        else:
            linestyle = '-'   # GNN 用实线
            label = model_name
            
        return color, linestyle, label

def plot_all_training_histories(rate, compare_mode='all', style_scheme='semantic', tidal_flow=False):
    """
    绘制所有模型的训练历史对比
    style_scheme: 'semantic' (方案2: 同长同色) 或 'unique' (方案1: 全不同)
    """
    os.makedirs(f"{results_folder}/rate{rate}/training_history", exist_ok=True)
    
    # --- 1. 奖励历史 ---
    plt.figure(figsize=(14, 8))
    
    # 辅助函数：绘制单条线
    def plot_line(m_type, length=None):
        folder_path = get_model_path(m_type, rate, length, tidal_flow=tidal_flow)
        r_path = os.path.join(folder_path, f"{m_type}_rewards_history.npy")
        
        if os.path.exists(r_path):
            rewards = np.load(r_path)
            window_size = 100
            if len(rewards) > window_size:
                moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                
                # 获取样式
                color, style, label = get_style_for_model(m_type, length, scheme=style_scheme)
                # 如果是语义模式，覆盖label以显示清晰的模型名
                final_label = f"{'GNN+Trans' if 'transformer' in m_type else 'GNN+LSTM'}({length})" if length else "GNN"
                
                plt.plot(np.arange(window_size-1, len(rewards)), moving_avg, 
                         color=color, linestyle=style, linewidth=2.5 if length is None else 2.0, 
                         label=final_label)

    # 绘制 GNN
    plot_line('gnn')
    
    # 绘制 LSTM
    if compare_mode in ['lstm', 'all']:
        for length in lstm_lengths:
            plot_line('gnn_lstm', length)
    
    # 绘制 Transformer
    if compare_mode in ['transformer', 'all']:
        for length in transformer_lengths:
            plot_line('gnn_transformer', length)

    plt.title(f"Training Reward Comparison (Rate={rate}/s)", fontsize=25)
    plt.xlabel("Episode", fontsize=23)
    plt.ylabel("Reward (Moving Avg)", fontsize=23)
    plt.legend(fontsize=20)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{results_folder}/rate{rate}/training_history/all_models_reward_comparison.png", dpi=300)
    plt.close()
    
    # --- 2. 损失历史 (逻辑同上，略作简化) ---
    plt.figure(figsize=(14, 8))
    
    def plot_loss_line(m_type, length=None):
        folder_path = get_model_path(m_type, rate, length, tidal_flow=tidal_flow)
        l_path = os.path.join(folder_path, f"{m_type}_loss_history.npy")
        
        if os.path.exists(l_path):
            losses = np.load(l_path)
            limit = 200000
            losses = losses[:limit]
            loss_window_size = min(10000, len(losses) // 10)
            moving_avg = pd.Series(losses).rolling(loss_window_size, min_periods=1).mean()
            
            color, style, label = get_style_for_model(m_type, length, scheme=style_scheme)
            final_label = f"{'GNN+Trans' if 'transformer' in m_type else 'GNN+LSTM'}({length})" if length else "GNN"
            
            plt.plot(moving_avg, color=color, linestyle=style, linewidth=2.0, label=final_label)

    plot_loss_line('gnn')
    if compare_mode in ['lstm', 'all']:
        for length in lstm_lengths: plot_loss_line('gnn_lstm', length)
    if compare_mode in ['transformer', 'all']:
        for length in transformer_lengths: plot_loss_line('gnn_transformer', length)

    plt.title(f"Training Loss Comparison (Rate={rate}/s)", fontsize=25)
    plt.xlabel("Training Step", fontsize=23)
    plt.ylabel("Loss (Moving Avg)", fontsize=23)
    plt.legend(fontsize=20)
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

def compare_models_for_rate(rate, episode, compare_mode='all', tidal_flow=False):
    """
    比较指定请求率下所有模型的性能
    compare_mode: 'lstm', 'transformer', 'all'
    """
    # 在函数开始设置随机种子
    base_seed = 42

    # 初始化环境
    simulation_time = get_simulation_time(rate)
    set_global_seed(base_seed)
    env = ComputingNetworkSimulator(
        'gurobi_solution_service_sources_sim.csv', 
        'gurobi_solution_compute_nodes_sim.csv', 
        rate=rate, 
        simulation_time=simulation_time,
        tidal_flow=tidal_flow, 
        tidal_period_count=period_count
    )
    
    # 测试所有模型
    models_data = []
    model_names = []
    skip_requests = max(max(lstm_lengths), max(transformer_lengths))

    # Helper function to run evaluation with consistent seeding
    def run_eval(agent, name):
        set_global_seed(base_seed) # 关键：模型评估前重置种子
        metrics = evaluate_policy(env, agent, num_episodes=episode, model_name=name, skip_first_requests=skip_requests)
        models_data.append(metrics)
        model_names.append(name)

    # --- 1. 测试基础GNN模型 ---
    print(f"[Testing] GNN Model (Rate={rate}/s)")
    folder_path = get_model_path('gnn', rate, tidal_flow=tidal_flow)
    model_path = os.path.join(folder_path, "gnn_dqn_final.pth")
    if os.path.exists(model_path):
        agent = GNNAgent(env, device='cuda')
        agent.policy_net.load_state_dict(torch.load(model_path, map_location='cuda'))
        run_eval(agent, 'GNN')
    
    # --- 2. 测试 GNN+LSTM 模型 ---
    if compare_mode in ['lstm', 'all']:
        for lstm_len in lstm_lengths:
            name = f'GNN+LSTM({lstm_len})'
            print(f"[Testing] {name}")
            folder = get_model_path('gnn_lstm', rate, lstm_len, tidal_flow=tidal_flow)
            path = os.path.join(folder, "gnn_lstm_dqn_final.pth")
            if os.path.exists(path):
                agent = GNNLSTMDDQNAgent(env, device='cuda', sequence_length=lstm_len)
                agent.policy_net.load_state_dict(torch.load(path, map_location='cuda'))
                run_eval(agent, name)

    # --- 3. 测试 GNN+Transformer 模型 ---
    if compare_mode in ['transformer', 'all']:
        for trans_len in transformer_lengths:
            name = f'GNN+Trans({trans_len})'
            print(f"[Testing] {name}")
            folder = get_model_path('gnn_transformer', rate, trans_len, tidal_flow=tidal_flow)
            path = os.path.join(folder, "gnn_transformer_dqn_final.pth")
            if os.path.exists(path):
                agent = GNNTransformerDDQNAgent(env, device='cuda', sequence_length=trans_len)
                agent.policy_net.load_state_dict(torch.load(path, map_location='cuda'))
                run_eval(agent, name)
    
    # --- 4. 测试 Baseline 策略 ---
    print(f"[Testing] Random Policy (Rate={rate}/s)")
    run_eval(lambda x: ('cloud', 0), 'Random')

    print(f"[Testing] Greedy Policy (Rate={rate}/s)")
    run_eval(env._greedy_policy, 'Greedy')

    print(f"[Testing] LUF Policy (Rate={rate}/s)")
    run_eval(env._lowest_utilization_policy, 'Lowest Utilization First')

    # 创建性能对比数据 - 使用新的指标名称
    metrics_df = pd.DataFrame({
        'Model': model_names,
        'Success Rate': [data.get('success_rate', 0) for data in models_data], 
        'Avg Reward': [data.get('avg_reward_stable', 0) for data in models_data],
        'Avg Latency All (ms)': [data.get('avg_latency_all', 0) for data in models_data],
        'Avg Latency Success (ms)': [data.get('avg_latency_success', 0) for data in models_data],
        'Cloud Usage (%)': [data.get('cloud_requests', 0) / data.get('total_requests', 1) * 100 for data in models_data],
        'Resource Utilization (%)': [
            np.mean(list(data.get('room_utilization', {}).values())) * 100 if data.get('room_utilization') else 0 
            for data in models_data
        ],
        'Processing Time (ms)': [data.get('avg_processing_time', 0) * 1000 for data in models_data]
    })
    
    # 确保结果目录存在
    print(f"Visualizing the results of models comparison for rate={rate}/s")
    os.makedirs(f"{results_folder}/rate{rate}/model_comparison", exist_ok=True)
    
    # 绘制性能对比柱状图（3x2布局）- 包含所有模型
    plt.figure(figsize=(18, 16))
    
    plot_specs = [
        (1, 'Success Rate', 'Success Rate'),
        (2, 'Avg Reward', 'Average Reward'),
        (3, 'Avg Latency All (ms)', 'Latency (All)'),
        (4, 'Avg Latency Success (ms)', 'Latency (Success)'),
        (5, 'Cloud Usage (%)', 'Cloud Usage'),
        (6, 'Resource Utilization (%)', 'Resource Util'),
        (7, 'Processing Time (ms)', 'Processing Time')
    ]
    
    for idx, col, title in plot_specs:
        plt.subplot(4, 2, idx)
        sns.barplot(x='Model', y=col, data=metrics_df)
        plt.title(title)
        plt.xticks(rotation=30, ha='right')
        # 给 Processing Time 添加数值标签
        if 'Processing' in title:
            ax = plt.gca()
            for p in ax.patches:
                ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{results_folder}/rate{rate}/model_comparison/comprehensive_performance_all.png", dpi=300)
    plt.close()
    
    # 创建不包含baseline模型的版本
    # filtered_df = metrics_df[~metrics_df['Model'].isin(['Random', 'Greedy', 'Lowest Utilization First'])]
    filtered_df = metrics_df[~metrics_df['Model'].isin(['Random', 'All-Cloud', 'All-Local', 'Lowest Utilization First'])]
    
    # 绘制性能对比柱状图（3x2布局）- 不包含baseline模型
    plt.figure(figsize=(18, 16))
    for idx, col, title in plot_specs:
        plt.subplot(4, 2, idx)
        sns.barplot(x='Model', y=col, data=filtered_df)
        plt.title(title)
        
        # 动态设置 Y 轴范围以突出差异
        vals = filtered_df[col]
        if len(vals) > 0:
            if 'Rate' in col or 'Usage' in col or 'Util' in col:
                # 百分比类数据，范围稍微放宽
                ymin, ymax = vals.min() * 0.9, min(vals.max() * 1.05, 100)
                plt.ylim(ymin, ymax)
            elif 'Reward' in col:
                # 负数奖励处理
                range_span = vals.max() - vals.min()
                plt.ylim(vals.min() - range_span*0.05, vals.max() + range_span*0.05)
            elif 'Latency' in col:
                plt.ylim(vals.min() * 0.95, vals.max() * 1.05)

        plt.xticks(rotation=30, ha='right')

    plt.tight_layout()
    plt.savefig(f"{results_folder}/rate{rate}/model_comparison/comprehensive_performance_filtered.png", dpi=300)
    plt.close()

    # 绘制每个模型的机房利用率柱状图
    for model_name, model_metrics in zip(model_names, models_data):
        # if model_name in ['Random', 'Greedy', 'Lowest Utilization First']:  # 跳过baseline策略
        #     continue
            
        # 准备数据
        room_ids = []
        util_values = []
        
        room_utilization = model_metrics.get('room_utilization', {})
        for room_id, util in room_utilization.items():
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

def evaluate_policy(env, policy_fn, num_episodes, model_name, skip_first_requests=0):
    metrics = {
        'total_requests': 0, 'succeed_requests': 0, 'cloud_requests': 0,
        'total_latency': 0, 'success_latency': 0, 'total_processing_time': 0,
        'total_reward': 0, 'room_utilization': {id: [] for id in env.nodes['rooms']},
        'stable_requests': 0, 'stable_latency': 0, 'stable_success_latency': 0,
        'stable_reward': 0, 'stable_success_count': 0,
        'success_rate': 0 # 确保 key 存在
    }

    for _ in tqdm(range(num_episodes), desc=f"Ep {model_name}", leave=False):
        state = env.reset()
        if hasattr(policy_fn, 'reset_episode'): policy_fn.reset_episode()
        
        done = False
        req_count = 0
        ep_stable = {'count':0, 'reward':0, 'latency':0, 'success':0, 's_latency':0}

        while not done:
            start = time.time()
            if hasattr(policy_fn, 'get_action'): action = policy_fn.get_action(state, epsilon=0.0)
            elif callable(policy_fn):
                if model_name == 'Random':
                    valid = env.get_valid_actions()
                    action = random.choice(valid) if valid else 'cloud'
                else: action, _ = policy_fn(env.current_request)
            metrics['total_processing_time'] += (time.time() - start)
            
            next_state, reward, done, info = env.step(action)
            
            # 冷启动过滤
            is_stable = req_count >= skip_first_requests
            
            if is_stable:
                ep_stable['count'] += 1
                ep_stable['reward'] += reward
                ep_stable['latency'] += info.get('last_latency', 0)
                if info.get('last_success', False):
                    ep_stable['success'] += 1
                    ep_stable['s_latency'] += info.get('last_latency', 0)

            # 更新 Total
            metrics['total_requests'] += 1
            metrics['total_reward'] += reward
            if info.get('used_cloud'): metrics['cloud_requests'] += 1
            
            state = next_state
            req_count += 1
        
        # 累加稳定期数据
        metrics['stable_requests'] += ep_stable['count']
        metrics['stable_reward'] += ep_stable['reward']
        metrics['stable_latency'] += ep_stable['latency']
        metrics['stable_success_count'] += ep_stable['success']
        metrics['stable_success_latency'] += ep_stable['s_latency']

        # 利用率采样
        if env.current_time > 1e-6:
            for rid, r in env.nodes['rooms'].items():
                util = r['cumulative_compute_time'] / (r['max_compute'] * env.current_time) if r['max_compute']>0 else 0
                metrics['room_utilization'][rid].append(min(util, 1.0))
    
    # 计算最终平均值
    if metrics['stable_requests'] > 0:
        metrics['avg_reward_stable'] = metrics['stable_reward'] / metrics['stable_requests']
        metrics['avg_latency_all'] = metrics['stable_latency'] / metrics['stable_requests']
        metrics['success_rate'] = metrics['stable_success_count'] / metrics['stable_requests']
        metrics['avg_latency_success'] = metrics['stable_success_latency'] / metrics['stable_success_count'] if metrics['stable_success_count'] > 0 else 0
    else:
        metrics.update({'avg_reward_stable':0, 'avg_latency_all':0, 'success_rate':0, 'avg_latency_success':0})
    
    metrics['avg_processing_time'] = metrics['total_processing_time'] / max(metrics['total_requests'], 1)
    for rid in metrics['room_utilization']:
        metrics['room_utilization'][rid] = np.mean(metrics['room_utilization'][rid]) if metrics['room_utilization'][rid] else 0
    
    print(f"模型 {model_name} 统计信息:")
    print(f"  - 稳定期请求数: {metrics['stable_requests']}")
    print(f"  - 所有请求平均延迟: {metrics['avg_latency_all']:.2f}ms")
    print(f"  - 成功请求平均延迟: {metrics['avg_latency_success']:.2f}ms")
    print(f"  - 成功率: {metrics['success_rate']:.2%}")
    print(f"  - 稳定期平均奖励: {metrics['avg_reward_stable']:.3f}")

    return metrics

def compare_requests_performance(rate, compare_mode='all', tidal_flow=False):
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
        
        # 设置随机种子
        set_global_seed(42 + exp_id)
        
        env = ComputingNetworkSimulator(
            'gurobi_solution_service_sources_sim.csv', 
            'gurobi_solution_compute_nodes_sim.csv', 
            rate=rate, 
            simulation_time=simulation_time,
            tidal_flow=tidal_flow, 
            tidal_period_count=period_count
        )
        
        # 生成请求序列
        print(f"  Generating common request sequence...")
        requests = []
        env.reset() 
        requests.append(env.current_request.copy())
        while True:
            req = env._generate_request()
            if env.current_time > env.total_time: break
            requests.append(req.copy())
        
        # 绘制网络拓扑和请求分布图
        if exp_id <= 3:
            plot_network_with_requests(env, requests, rate, exp_id) 

        # 测试所有模型
        model_results = {}
        model_utils = {} # 新增：存储每个模型的整个周期利用率

        # Helper for test
        def run_test_reqs(agent, name):
            # 这里的种子确保 Agent 决策的随机性（如 epsilon-greedy）在相同请求序列下一致
            set_global_seed(42 + exp_id) 
            return test_performance_on_requests(env, agent, requests, name)
        
        # === 测试 Baseline 策略 ===
        print(f"  Testing Baseline policies on common sequence...")
        
        # 1. Random 策略
        perf_rand, util_rand = run_test_reqs(BaselineAgent(env, 'Random'), 'Random')
        model_results['Random'] = perf_rand
        model_utils['Random'] = util_rand
        
        # 2. All-Cloud 策略
        perf_cloud, util_cloud = run_test_reqs(BaselineAgent(env, 'All-Cloud'), 'All-Cloud')
        model_results['All-Cloud'] = perf_cloud
        model_utils['All-Cloud'] = util_cloud
        
        # 3. All-Local 策略
        perf_local, util_local = run_test_reqs(BaselineAgent(env, 'All-Local'), 'All-Local')
        model_results['All-Local'] = perf_local
        model_utils['All-Local'] = util_local
        
        # 4. Lowest Utilization First 策略 (UTF/LUF)
        perf_luf, util_luf = run_test_reqs(BaselineAgent(env, 'LUF'), 'Lowest Utilization First')
        model_results['Lowest Utilization First'] = perf_luf
        model_utils['Lowest Utilization First'] = util_luf

        # GNN
        folder = get_model_path('gnn', rate, tidal_flow=tidal_flow)
        if os.path.exists(os.path.join(folder, "gnn_dqn_final.pth")):
            agent = GNNAgent(env, device='cuda')
            agent.policy_net.load_state_dict(torch.load(os.path.join(folder, "gnn_dqn_final.pth")))
            perf, util = run_test_reqs(agent, 'GNN')
            model_results['GNN'] = perf
            model_utils['GNN'] = util
            
        # LSTM
        if compare_mode in ['lstm', 'all']:
            for length in lstm_lengths:
                folder = get_model_path('gnn_lstm', rate, length, tidal_flow=tidal_flow)
                if os.path.exists(os.path.join(folder, "gnn_lstm_dqn_final.pth")):
                    agent = GNNLSTMDDQNAgent(env, device='cuda', sequence_length=length)
                    agent.policy_net.load_state_dict(torch.load(os.path.join(folder, "gnn_lstm_dqn_final.pth")))
                    name = f'GNN+LSTM({length})'
                    perf, util = run_test_reqs(agent, name)
                    model_results[name] = perf
                    model_utils[name] = util
        
        # Transformer
        if compare_mode in ['transformer', 'all']:
            for length in transformer_lengths:
                folder = get_model_path('gnn_transformer', rate, length, tidal_flow=tidal_flow)
                if os.path.exists(os.path.join(folder, "gnn_transformer_dqn_final.pth")):
                    agent = GNNTransformerDDQNAgent(env, device='cuda', sequence_length=length)
                    agent.policy_net.load_state_dict(torch.load(os.path.join(folder, "gnn_transformer_dqn_final.pth")))
                    name = f'GNN+Trans({length})'
                    perf, util = run_test_reqs(agent, name)
                    model_results[name] = perf
                    model_utils[name] = util

        # 存储本次实验结果 (保持原有逻辑不变)
        all_experiment_results[exp_id] = {
            'model_results': model_results,
            'model_utils': model_utils,
            'requests': requests
        }
    
    # 计算平均结果
    avg_results = {}
    per_req_summary_df = pd.DataFrame() # 初始化为空 DataFrame
    
    if not all_experiment_results:
        return {}, per_req_summary_df

    first_models = all_experiment_results[0]['model_results']
    per_req_summary_list = []

    for model_name in first_models.keys():
        exp_data_list = [exp['model_results'][model_name] for exp in all_experiment_results.values()]
        # 获取该模型在所有实验中的利用率列表
        exp_util_list = [exp['model_utils'][model_name] for exp in all_experiment_results.values()]
        
        min_len = min(len(d) for d in exp_data_list)
        
        avg_perf = []
        total_rewards = []
        total_latencies = []
        success_latencies = []
        all_process_times = [] # 用于计算平均处理时长
        success_count = 0
        cloud_usage_count = 0
        total_count = 0

        for i in range(min_len):
            if i < skip_first_requests: continue
            
            rewards = [d[i]['reward'] for d in exp_data_list]
            latencies = [d[i]['latency'] for d in exp_data_list]
            suc_latencies = [d[i]['latency'] for d in exp_data_list if d[i]['success']]
            process_times = [d[i]['processing_time'] for d in exp_data_list]
            successes = [d[i]['success'] for d in exp_data_list]
            is_cloud_list = [d[i].get('is_cloud', False) for d in exp_data_list]
            
            avg_perf.append({
                'request_idx': i,
                'reward': np.mean(rewards),
                'latency_all': np.mean(latencies),
                'latency_success': np.mean(suc_latencies) if suc_latencies else None,
                'processing_time': np.mean(process_times),
                'success': np.mean(successes)
            })

            # 汇总用于生成 CSV
            total_rewards.append(np.mean(rewards))
            total_latencies.append(np.mean(latencies))
            all_process_times.extend(process_times) # 收集所有处理时间
            if suc_latencies: success_latencies.extend(suc_latencies)
            success_count += sum(successes)
            cloud_usage_count += sum(is_cloud_list)
            total_count += len(successes)

        if avg_perf:
            avg_results[model_name] = avg_perf
            cloud_usage_pct = (cloud_usage_count / total_count * 100) if total_count > 0 else 0
            # 2. 修改 CSV 汇总数据，增加利用率和处理时长
            per_req_summary_list.append({
                'Model': model_name,
                'Success Rate': success_count / total_count if total_count > 0 else 0,
                'Avg Reward': np.mean(total_rewards),
                'Avg Latency All (ms)': np.mean(total_latencies),
                'Avg Latency Success (ms)': np.mean(success_latencies) if success_latencies else 0,
                'Processing Time (ms)': np.mean(all_process_times) * 1000, # 转换为ms，列名匹配metrics_df
                'Resource Utilization (%)': np.mean(exp_util_list) * 100, # 长期平均利用率
                'Cloud Usage (%)': cloud_usage_pct
            })
    
    # 生成 DataFrame 用于返回
    per_req_summary_df = pd.DataFrame(per_req_summary_list)
    
    per_req_summary_df.to_csv(
        f"{results_folder}/rate{rate}/model_comparison/per_request/per_request_performance_metrics.csv", 
        index=False
    )

    # === 按三大服务类别分析 ===
    print("Analyzing performance by service category...")
    
    service_categories = {
        "ultra_low_latency": ["urgent_braking", "traffic_control"],
        "low_latency": ["collision_avoidance", "sensor_sharing"], 
        "high_latency_tolerance": ["hd_map_update", "infotainment"]
    }
    
    type_to_category = {}
    for cat, types in service_categories.items():
        for t in types: type_to_category[t] = cat

    analysis_records = []
    all_models_set = set()

    for exp_id in range(num_experiments):
        exp_data = all_experiment_results[exp_id]
        requests_seq = exp_data['requests']
        model_results = exp_data['model_results']
        
        for model_name, perf_list in model_results.items():
            all_models_set.add(model_name)
            for item in perf_list:
                req_idx = item['request_idx']
                if req_idx < skip_first_requests: continue
                if req_idx >= len(requests_seq): continue
                    
                req_type = requests_seq[req_idx]['type']
                category = type_to_category.get(req_type)
                
                if category:
                    analysis_records.append({
                        'Model': model_name,
                        'Category': category,
                        'Success': 1 if item['success'] else 0,
                        'Latency': item['latency'],
                        'Latency_Success': item['latency'] if item['success'] else None
                    })

    # 定义要过滤的 baseline 模型列表
    baseline_models = ['Random', 'All-Cloud', 'All-Local', 'Lowest Utilization First']

    if analysis_records:
        df_analysis = pd.DataFrame(analysis_records)
        custom_palette = {m: get_style_for_model(m, scheme='semantic')[0] for m in all_models_set}
        
        # 循环两次：一次绘制全量模型，一次绘制过滤后的强化学习模型
        for is_filtered in [False, True]:
            suffix = "_filtered" if is_filtered else ""
            plot_df = df_analysis[~df_analysis['Model'].isin(baseline_models)] if is_filtered else df_analysis
            
            if plot_df.empty: continue
            
            # A. Success Rate Plot
            plt.figure(figsize=(14, 8))
            ax = sns.barplot(data=plot_df, x='Category', y='Success', hue='Model', palette=custom_palette, ci=None)
            y_min = plot_df.groupby(['Category', 'Model'])['Success'].mean().min()
            ax.set_ylim(max(0, y_min * 0.98), 1.0) 
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            plt.tight_layout()
            plt.savefig(f"{results_folder}/rate{rate}/model_comparison/per_request/success_rate_by_category{suffix}.png", dpi=300)
            plt.close()
            
            # B. Latency Comparison
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 8))
            
            # 左图：所有请求的平均延迟
            sns.barplot(data=plot_df, x='Category', y='Latency', hue='Model', 
                        palette=custom_palette, ax=ax1, ci=None)
            ax1.set_title("Avg Latency (All Requests)")
            ax1.set_ylabel("Latency (ms)")
            
            mean_latencies = plot_df.groupby(['Category', 'Model'])['Latency'].mean()
            if not mean_latencies.empty:
                ax1.set_ylim(mean_latencies.min() * 0.9, mean_latencies.max() * 1.05)
            if ax1.get_legend(): ax1.legend_.remove()
            
            # 右图：成功请求的平均延迟
            sns.barplot(data=plot_df, x='Category', y='Latency_Success', hue='Model', 
                        palette=custom_palette, ax=ax2, ci=None)
            ax2.set_title("Avg Latency (Successful Requests Only)")
            ax2.set_ylabel("Latency (ms)")
            
            mean_success_latencies = plot_df.groupby(['Category', 'Model'])['Latency_Success'].mean()
            if not mean_success_latencies.empty:
                ax2.set_ylim(mean_success_latencies.min() * 0.9, mean_success_latencies.max() * 1.05)
            
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            
            plt.tight_layout()
            plt.savefig(f"{results_folder}/rate{rate}/model_comparison/per_request/latency_by_category{suffix}.png", dpi=300)
            plt.close()
            
        # 保存 CSV (全量数据)
        summary_df = df_analysis.groupby(['Model', 'Category']).agg(
            Success_Rate=('Success', 'mean'),
            Avg_Latency_All=('Latency', 'mean'),
            Avg_Latency_Success=('Latency_Success', 'mean'),
            Count=('Success', 'count')
        ).reset_index()
        summary_df.to_csv(f"{results_folder}/rate{rate}/model_comparison/per_request/category_analysis.csv", index=False)
    
    # === 绘制 per request 曲线图 ===
    metrics_to_plot = [
        ('reward', 'Reward', 'Avg Reward'),
        ('latency_all', 'Latency (ms)', 'Avg Latency (All)'),
        ('latency_success', 'Latency (ms)', 'Avg Latency (Success)'),
        ('processing_time', 'Time (ms)', 'Avg Processing Time'),
        ('success', 'Rate', 'Success Rate')
    ]
    
    # 同样循环两次，绘制曲线图
    for is_filtered in [False, True]:
        suffix = "_filtered" if is_filtered else ""
        current_results = {k: v for k, v in avg_results.items() if k not in baseline_models} if is_filtered else avg_results
        
        if not current_results: continue

        for metric_key, ylabel, title in metrics_to_plot:
            plt.figure(figsize=(16, 8))
            has_data = False
            for model_name, perf in current_results.items():
                if len(perf) <= PLOT_SKIP_REQUESTS: continue
                
                plot_perf = perf[PLOT_SKIP_REQUESTS:]
                data = [p[metric_key] for p in plot_perf]
                indices = [p['request_idx'] for p in plot_perf]
                
                if metric_key == 'processing_time': data = [d * 1000 for d in data]
                
                clean_data_series = pd.Series(data)
                
                if not clean_data_series.dropna().empty:
                    has_data = True
                    ws = min(1000, max(5, len(data)//10)) 
                    ma = clean_data_series.rolling(ws, min_periods=ws//2).mean()
                    
                    color, style, _ = get_style_for_model(model_name, scheme='semantic')
                    plt.plot(indices, ma, color=color, linestyle=style, label=model_name)
            
            if has_data:
                plt.title(f"{title} (Rate={rate}/s) - Moving Avg (Window={ws})")
                plt.xlabel("Request Index")
                plt.ylabel(ylabel)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
                plt.tight_layout()
                safe_title = title.lower().replace(' ', '_').replace('(', '').replace(')', '')
                plt.savefig(f"{results_folder}/rate{rate}/model_comparison/per_request/{safe_title}{suffix}.png", dpi=300)
            plt.close()
    
    # 保存原始平均结果 (全量数据)
    for model_name, performance in avg_results.items():
        if performance:
            df = pd.DataFrame(performance)
            safe_model_name = model_name.replace('+', '_').replace('(', '_').replace(')', '_')
            df.to_csv(f"{results_folder}/rate{rate}/model_comparison/per_request/{safe_model_name}_avg_request_performance.csv", index=False)
    
    # === 绘制联合箱线图 ===
    print("Generating combined boxplots...")
    
    for is_filtered in [False, True]:
        suffix = "_filtered" if is_filtered else ""
        current_results = {k: v for k, v in avg_results.items() if k not in baseline_models} if is_filtered else avg_results
        
        if not current_results: continue
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 8))
        
        metrics_config = [
            ('reward', 'Reward Distribution', axes[0]),
            ('latency_all', 'Latency (All) Distribution', axes[1]),
            ('latency_success', 'Latency (Success Only) Distribution', axes[2])
        ]
        
        for metric_key, title, ax in metrics_config:
            plot_data = []
            labels = []
            colors = []
            
            for model_name, perf in current_results.items():
                if not perf: continue
                
                raw_values = [p[metric_key] for p in perf]
                
                if 'latency' in metric_key:
                    values = [v for v in raw_values if v is not None and v > 0]
                else:
                    values = [v for v in raw_values if v is not None]
                    
                if values:
                    plot_data.append(values)
                    labels.append(model_name)
                    c, _, _ = get_style_for_model(model_name, scheme='semantic')
                    colors.append(c)
            
            if plot_data:
                bplot = ax.boxplot(plot_data, patch_artist=True, labels=labels, showfliers=False)
                for patch, color in zip(bplot['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.6)
                
                ax.set_title(title, fontsize=14)
                ax.grid(True, axis='y', alpha=0.3)
                ax.set_xticklabels(labels, rotation=30, ha='right')
                
                if 'Latency' in title: ax.set_ylabel("Time (ms)")
                elif 'Reward' in title: ax.set_ylabel("Scaled Reward")

        plt.tight_layout()
        plt.savefig(f"{results_folder}/rate{rate}/model_comparison/per_request/combined_boxplots{suffix}.png", dpi=300)
        plt.close()

    # [修改] 返回 per_req_summary_df 供 main 函数使用
    return avg_results, per_req_summary_df

def test_performance_on_requests(env, agent, requests, agent_name):
    """在给定的请求序列上测试模型性能"""
    state = env.reset()
    if hasattr(agent, 'reset_episode'):
        agent.reset_episode()
    
    performance = []
    
    for i, request in enumerate(tqdm(requests, desc=f"Testing {agent_name}", leave=False)):
        env.current_time = request['timestamp']
        env.current_request = request

        # 强制 Simulator 检查当前时间点有哪些任务完成了，并释放资源
        env._process_pending_events()

        # 2. 获取状态 (此时获取的状态才是资源释放后的正确状态)
        state = env._get_state()
        
        # 3. 决策
        start = time.time()
        action = agent.get_action(state, epsilon=0.0) # 测试时 epsilon=0
        process_time = time.time() - start
        
        _, reward, _, metrics = env.step(action, skip_generation=True)
        
        performance.append({
            'request_idx': i,
            'time': request['timestamp'],
            'action': action,
            'reward': reward,
            'latency': metrics.get('last_latency', 0),
            'success': metrics.get('last_success', False),
            'processing_time': process_time,
            'is_cloud': metrics.get('used_cloud', False)
        })
    
    # 2. 新增：返回性能列表的同时，返回该 Episode 的全局平均利用率
    final_utilization = env.get_global_average_utilization()
    
    return performance, final_utilization

def main():
    """主函数：执行所有对比任务"""
    parser = argparse.ArgumentParser(description='Visualize model performance')
    parser.add_argument('--mode', type=str, default='all', choices=['lstm', 'transformer', 'all'],
                        help='Compare mode: lstm, transformer, or all')
    parser.add_argument('--tidal_flow', action='store_true',
                        help='Enable tidal flow pattern in request generation')
    args = parser.parse_args()
    
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 12

    global results_folder
    tidal_suffix = "_tidal" if args.tidal_flow else ""
    results_folder = result_folder + tidal_suffix
    os.makedirs(results_folder, exist_ok=True)
    
    summary_data = []
    
    for rate in rates:
        print(f"\n{'='*50}")
        print(f"Processing Rate={rate}/s | Mode={args.mode} | Tidal Flow={args.tidal_flow}")
        print(f"{'='*50}")
        
        os.makedirs(f"{results_folder}/rate{rate}", exist_ok=True)
        
        print("Visualizing training histories...")
        plot_all_training_histories(rate, compare_mode=args.mode, tidal_flow=args.tidal_flow)
        
        # print("Comparing model performance metrics (Macro evaluation)...")
        # 运行宏观评估以生成单独的文件夹图表
        # compare_models_for_rate(rate, episode=n_episode, compare_mode=args.mode, tidal_flow=args.tidal_flow)
        
        print("Comparing performance on same request sequence (Micro evaluation)...")
        # 使用 per-request 的结果来填充最终的 summary_data
        _, req_summary_df = compare_requests_performance(rate, compare_mode=args.mode, tidal_flow=args.tidal_flow)
        
        for _, row in req_summary_df.iterrows():
            row_dict = row.to_dict()
            row_dict['rate'] = rate
            summary_data.append(row_dict)
        
        print(f"Completed processing for rate={rate}/s")

    # 保存汇总结果
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f"{results_folder}/performance_summary.csv", index=False)

        print("Generating summary plots across all rates...")

        # 1. 构建统一的颜色调色板 (确保与之前的图颜色一致)
        unique_models = summary_df['Model'].unique()
        custom_palette = {}
        for model_name in unique_models:
            # 调用之前的 get_style_for_model 函数获取颜色
            # 注意：Baseline 模型需要在这里处理一下，如果没有在 get_style_for_model 定义，给个默认色
            if model_name in ['Random', 'Greedy', 'Lowest Utilization First']:
                if model_name == 'Random': color = 'gray'
                elif model_name == 'Greedy': color = 'brown'
                else: color = 'olive'
            else:
                color, _, _ = get_style_for_model(model_name, scheme='semantic')
            custom_palette[model_name] = color

        # 2. 定义需要绘制的指标配置列表
        # 格式: (DataFrame列名, 图表标题, 文件名后缀)
        metrics_config = [
            ('Avg Latency All (ms)', 'Average Latency (All Requests)', 'avg_latency_all'),
            ('Avg Latency Success (ms)', 'Average Latency (Success Requests)', 'avg_latency_success'),
            ('Avg Reward', 'Average Reward', 'avg_reward'), 
            ('Processing Time (ms)', 'Average Processing Time', 'processing_time'),
            ('Success Rate', 'Success Rate', 'success_rate'),
            ('Cloud Usage (%)', 'Cloud Usage', 'cloud_usage'),
            ('Resource Utilization (%)', 'Resource Utilization', 'resource_utilization')
        ]

        # 3. 封装绘图函数
        def plot_summary(df, suffix_folder=""):
            for col, title, fname in metrics_config:
                # 检查列是否存在，防止报错
                if col not in df.columns:
                    continue

                plt.figure(figsize=(16, 8)) # 稍微加宽以适应图例
                
                # 绘图
                ax = sns.barplot(x='rate', y=col, hue='Model', data=df, palette=custom_palette)
                
                # 设置标题和标签
                plt.title(f"{title} Comparison Across Rates")
                plt.xlabel("Request Rate (requests/s)")
                plt.ylabel(title)
                plt.legend(title='Model', bbox_to_anchor=(1.01, 1), loc='upper left')
                
                # [修改] 动态调整 Y 轴范围，不再强制 0-100%
                if not df[col].empty:
                    y_min = df[col].min()
                    y_max = df[col].max()
                    
                    # 针对百分比数据（Success Rate, Usage, Utilization），如果是0-1小数或0-100
                    is_percentage = 'Rate' in col or 'Usage' in col or 'Utilization' in col
                    
                    if is_percentage:
                        # 确定上下界，保留一定边距
                        padding = (y_max - y_min) * 0.1 if y_max != y_min else 0.05
                        # 如果下限很接近0，就锁定0，否则用数据最小值
                        lower_bound = 0 if y_min < (0.1 if y_max <= 1 else 10) else max(0, y_min - padding)
                        upper_bound = y_max + padding
                        # 封顶
                        if y_max <= 1.0: upper_bound = min(upper_bound, 1.05)
                        elif y_max <= 100: upper_bound = min(upper_bound, 105)
                        
                        plt.ylim(lower_bound, upper_bound)
                    else:
                        # 非百分比数据（Reward, Latency, Time）
                        padding = (y_max - y_min) * 0.1 if y_max != y_min else abs(y_max) * 0.1
                        plt.ylim(y_min - padding, y_max + padding)

                plt.tight_layout()
                
                # 保存
                save_path = f"{results_folder}/{fname}_across_rates{suffix_folder}.png"
                plt.savefig(save_path, dpi=300)
                plt.close()

        # --- 第一组图：包含所有模型 ---
        plot_summary(summary_df)

        # --- 第二组图：仅 RL 模型 (去除基线) ---
        # 这样可以更清晰地看到 GNN, LSTM, Transformer 之间的细微差异
        # baseline_models = ['Random', 'Greedy', 'Lowest Utilization First']
        baseline_models = ['Random', 'All-Cloud', 'All-Local', 'Lowest Utilization First']
        rl_df = summary_df[~summary_df['Model'].isin(baseline_models)].copy()
        
        if not rl_df.empty:
            print("Generating filtered summary plots (RL models only)...")
            plot_summary(rl_df, suffix_folder="_filtered")

    print(f"\nAll processing completed. Results saved to {results_folder}")

if __name__ == "__main__":
    main()