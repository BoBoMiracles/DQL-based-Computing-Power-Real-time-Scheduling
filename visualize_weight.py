import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pandas as pd
from tqdm import tqdm
import pickle
import re
import json
from collections import defaultdict

# === 全局参数 ===
results_folder = 'weight_experiment_analysis'
os.makedirs(results_folder, exist_ok=True)

# 设置绘图风格
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12

def extract_experiment_data(base_dir="weight_experiment_results"):
    """从实验目录中提取所有数据"""
    experiment_data = []
    
    # 遍历所有实验目录
    for exp_dir in os.listdir(base_dir):
        exp_path = os.path.join(base_dir, exp_dir)
        
        # 跳过非目录项（如文件）
        if not os.path.isdir(exp_path) or not exp_dir.startswith('exp'):
            continue
            
        # 遍历每个实验中的模型类型
        for model_dir in os.listdir(exp_path):
            model_path = os.path.join(exp_path, model_dir)
            
            # 检查是否是模型目录
            if not os.path.isdir(model_path) or not model_dir.startswith('model_'):
                continue
                
            # 提取模型类型
            if 'gnn_lstm3' in model_dir:
                model_type = 'GNN+LSTM3'
            elif 'gnn_lstm10' in model_dir:
                model_type = 'GNN+LSTM10'
            elif 'gnn' in model_dir:
                model_type = 'GNN'
            else:
                continue
                
            # 遍历到达率目录
            for rate_dir in os.listdir(model_path):
                rate_path = os.path.join(model_path, rate_dir)
                
                if not os.path.isdir(rate_path) or not rate_dir.startswith('rate_'):
                    continue
                    
                # 提取到达率
                arrival_rate = float(rate_dir.split('_')[1])
                
                # 遍历权重配置目录
                for weight_dir in os.listdir(rate_path):
                    weight_path = os.path.join(rate_path, weight_dir)
                    
                    if not os.path.isdir(weight_path) or not weight_dir.startswith('weights_'):
                        continue
                        
                    # 提取权重配置
                    weights_str = weight_dir.split('_')[1:]
                    reward_weights = tuple(float(w) for w in weights_str)
                    
                    # 检查是否有训练历史文件
                    history_file = os.path.join(weight_path, "training_history.pkl")
                    if not os.path.exists(history_file):
                        continue
                        
                    # 加载训练历史
                    with open(history_file, 'rb') as f:
                        history = pickle.load(f)
                    
                    # 提取最终性能指标（最后10个episode的平均值）
                    final_metrics = {
                        'model_type': model_type,
                        'arrival_rate': arrival_rate,
                        'reward_weights': reward_weights,
                        'final_reward': np.mean(history['rewards'][-10:]),
                        'avg_success_rate': np.mean(history['success_rates'][-10:]),
                        'avg_cloud_rate': np.mean(history['cloud_usage'][-10:]),
                        'avg_latency': np.mean(history['avg_latencies'][-10:])
                    }
                    
                    experiment_data.append(final_metrics)
    
    return experiment_data

def create_summary_dataframe(experiment_data):
    """将实验数据转换为DataFrame"""
    df = pd.DataFrame(experiment_data)
    
    # 为权重配置创建可读标签
    df['weights_label'] = df['reward_weights'].apply(
        lambda x: f"W={x[0]}, L={x[1]}, U={x[2]}, C={x[3]}"
    )
    
    return df

def plot_performance_by_weights(df, metric, title_suffix, ylabel, filename):
    """绘制不同权重配置下的性能对比"""
    plt.figure(figsize=(14, 8))
    
    # 为每种模型类型和到达率创建子图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for i, rate in enumerate([2.0, 10.0]):
        rate_data = df[df['arrival_rate'] == rate]
        
        # 为每种模型类型绘制数据
        for model_type in ['GNN', 'GNN+LSTM3', 'GNN+LSTM10']:
            model_data = rate_data[rate_data['model_type'] == model_type]
            
            # 按权重配置排序
            model_data = model_data.sort_values('weights_label')
            
            axes[i].plot(
                range(len(model_data)), 
                model_data[metric], 
                'o-', 
                label=model_type,
                linewidth=2,
                markersize=8
            )
        
        axes[i].set_title(f'Arrival Rate = {rate}/s')
        axes[i].set_xlabel('Weight Configuration')
        axes[i].set_ylabel(ylabel)
        axes[i].set_xticks(range(len(model_data)))
        axes[i].set_xticklabels(model_data['weights_label'].tolist(), rotation=45, ha='right')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle(f'{title_suffix} by Weight Configuration')
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, filename), dpi=300, bbox_inches='tight')
    plt.close()

def plot_bar_comparison(df, metric, title_suffix, ylabel, filename):
    """绘制柱状图比较不同模型和权重配置的性能"""
    plt.figure(figsize=(16, 10))
    
    # 创建分组柱状图
    pivot_df = df.pivot_table(
        index='weights_label', 
        columns=['model_type', 'arrival_rate'], 
        values=metric
    )
    
    # 为每种到达率创建子图
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    
    for i, rate in enumerate([2.0, 10.0]):
        # 提取该到达率的数据
        rate_data = {}
        for model_type in ['GNN', 'GNN+LSTM3', 'GNN+LSTM10']:
            rate_data[model_type] = pivot_df[(model_type, rate)].values
        
        # 创建柱状图
        x = np.arange(len(pivot_df.index))
        width = 0.25
        
        for j, (model_type, values) in enumerate(rate_data.items()):
            axes[i].bar(
                x + j * width, 
                values, 
                width, 
                label=model_type
            )
        
        axes[i].set_title(f'Arrival Rate = {rate}/s')
        axes[i].set_xlabel('Weight Configuration')
        axes[i].set_ylabel(ylabel)
        axes[i].set_xticks(x + width)
        axes[i].set_xticklabels(pivot_df.index, rotation=45, ha='right')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'{title_suffix} Comparison')
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, filename), dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_curves(base_dir="weight_experiment_results"):
    """绘制训练曲线对比（奖励和损失）"""
    # 创建训练曲线目录
    training_curve_dir = os.path.join(results_folder, "training_curves")
    os.makedirs(training_curve_dir, exist_ok=True)
    
    # 按权重配置分组
    weight_configs = set()
    for exp_dir in os.listdir(base_dir):
        exp_path = os.path.join(base_dir, exp_dir)
        
        # 跳过非目录项
        if not os.path.isdir(exp_path) or not exp_dir.startswith('exp'):
            continue
            
        # 遍历每个实验中的模型类型
        for model_dir in os.listdir(exp_path):
            model_path = os.path.join(exp_path, model_dir)
            
            # 跳过非目录项
            if not os.path.isdir(model_path) or not model_dir.startswith('model_'):
                continue
                
            # 遍历到达率目录
            for rate_dir in os.listdir(model_path):
                rate_path = os.path.join(model_path, rate_dir)
                
                # 跳过非目录项
                if not os.path.isdir(rate_path) or not rate_dir.startswith('rate_'):
                    continue
                    
                # 遍历权重配置目录
                for weight_dir in os.listdir(rate_path):
                    weight_path = os.path.join(rate_path, weight_dir)
                    
                    # 跳过非目录项
                    if not os.path.isdir(weight_path) or not weight_dir.startswith('weights_'):
                        continue
                        
                    weights_str = weight_dir.split('_')[1:]
                    weight_config = tuple(float(w) for w in weights_str)
                    weight_configs.add(weight_config)
    
    # 为每种权重配置绘制训练曲线
    for weight_config in weight_configs:
        weights_str = "_".join([f"{w:.1f}" for w in weight_config])
        weights_label = f"W={weight_config[0]}, L={weight_config[1]}, U={weight_config[2]}, C={weight_config[3]}"
        
        # 为每种到达率创建图表
        for rate in [2.0, 10.0]:
            # 创建奖励曲线图
            plt.figure(figsize=(14, 10))
            
            # 绘制每种模型的奖励曲线
            for model_type, color in zip(['GNN', 'GNN+LSTM3', 'GNN+LSTM10'], ['blue', 'green', 'red']):
                # 查找对应的训练历史文件
                history_files = []
                for exp_dir in os.listdir(base_dir):
                    exp_path = os.path.join(base_dir, exp_dir)
                    
                    # 跳过非目录项
                    if not os.path.isdir(exp_path) or not exp_dir.startswith('exp'):
                        continue
                        
                    # 根据模型类型确定目录名
                    if model_type == 'GNN':
                        model_dir_name = "model_gnn"
                    elif model_type == 'GNN+LSTM3':
                        model_dir_name = "model_gnn_lstm3"
                    elif model_type == 'GNN+LSTM10':
                        model_dir_name = "model_gnn_lstm10"
                    else:
                        continue
                        
                    model_path = os.path.join(exp_path, model_dir_name)
                    
                    if not os.path.exists(model_path):
                        continue
                        
                    rate_dir_name = f"rate_{rate}"
                    rate_path = os.path.join(model_path, rate_dir_name)
                    
                    if not os.path.exists(rate_path):
                        continue
                        
                    weight_dir_name = f"weights_{weights_str}"
                    weight_path = os.path.join(rate_path, weight_dir_name)
                    
                    if not os.path.exists(weight_path):
                        continue
                        
                    history_file = os.path.join(weight_path, "training_history.pkl")
                    if os.path.exists(history_file):
                        history_files.append(history_file)
                
                if not history_files:
                    print(f"警告: 未找到 {model_type} 模型在权重 {weights_str} 和到达率 {rate} 的训练数据")
                    continue
                    
                # 加载训练历史并绘制曲线
                with open(history_files[0], 'rb') as f:
                    history = pickle.load(f)
                
                # 绘制奖励曲线
                rewards = history['rewards']
                episodes = range(1, len(rewards) + 1)
                
                # 使用移动平均平滑曲线
                window_size = 10
                moving_avg = pd.Series(rewards).rolling(window_size, min_periods=1).mean()
                
                plt.plot(episodes, moving_avg, color=color, label=model_type, linewidth=2)
            
            plt.title(f'Training Reward Curves\n{weights_label}, Rate={rate}/s')
            plt.xlabel('Episode')
            plt.ylabel('Reward (Moving Average)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 保存奖励曲线图像
            safe_weights_str = weights_str.replace('.', '_')
            filename = f"reward_curve_rate{rate}_weights{safe_weights_str}.png"
            plt.savefig(os.path.join(training_curve_dir, filename), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 创建损失曲线图
            plt.figure(figsize=(14, 10))
            
            # 绘制每种模型的损失曲线
            for model_type, color in zip(['GNN', 'GNN+LSTM3', 'GNN+LSTM10'], ['blue', 'green', 'red']):
                # 查找对应的训练历史文件
                history_files = []
                for exp_dir in os.listdir(base_dir):
                    exp_path = os.path.join(base_dir, exp_dir)
                    
                    # 跳过非目录项
                    if not os.path.isdir(exp_path) or not exp_dir.startswith('exp'):
                        continue
                        
                    # 根据模型类型确定目录名
                    if model_type == 'GNN':
                        model_dir_name = "model_gnn"
                    elif model_type == 'GNN+LSTM3':
                        model_dir_name = "model_gnn_lstm3"
                    elif model_type == 'GNN+LSTM10':
                        model_dir_name = "model_gnn_lstm10"
                    else:
                        continue
                        
                    model_path = os.path.join(exp_path, model_dir_name)
                    
                    if not os.path.exists(model_path):
                        continue
                        
                    rate_dir_name = f"rate_{rate}"
                    rate_path = os.path.join(model_path, rate_dir_name)
                    
                    if not os.path.exists(rate_path):
                        continue
                        
                    weight_dir_name = f"weights_{weights_str}"
                    weight_path = os.path.join(rate_path, weight_dir_name)
                    
                    if not os.path.exists(weight_path):
                        continue
                        
                    history_file = os.path.join(weight_path, "training_history.pkl")
                    if os.path.exists(history_file):
                        history_files.append(history_file)
                
                if not history_files:
                    print(f"警告: 未找到 {model_type} 模型在权重 {weights_str} 和到达率 {rate} 的训练数据")
                    continue
                    
                # 加载训练历史并绘制曲线
                with open(history_files[0], 'rb') as f:
                    history = pickle.load(f)
                
                # 绘制损失曲线
                if 'losses' in history and history['losses']:
                    losses = history['losses']
                    # 使用大窗口计算移动平均以平滑曲线
                    window_size = 1000
                    moving_avg = pd.Series(losses).rolling(window_size, min_periods=1).mean()
                    
                    # 绘制移动平均损失
                    steps = range(len(moving_avg))
                    plt.plot(steps, moving_avg, color=color, label=model_type, linewidth=2)
                else:
                    print(f"警告: {model_type} 模型在权重 {weights_str} 和到达率 {rate} 的训练数据中没有损失记录")
            
            plt.title(f'Training Loss Curves\n{weights_label}, Rate={rate}/s')
            plt.xlabel('Training Step')
            plt.ylabel('Loss (Moving Average)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 保存损失曲线图像
            filename = f"loss_curve_rate{rate}_weights{safe_weights_str}.png"
            plt.savefig(os.path.join(training_curve_dir, filename), dpi=300, bbox_inches='tight')
            plt.close()

def create_radar_chart(df, filename):
    """创建雷达图比较不同模型和权重配置的综合性能"""
    from math import pi
    
    # 创建雷达图目录
    radar_dir = os.path.join(results_folder, "radar")
    os.makedirs(radar_dir, exist_ok=True)
    
    # 只使用三个指标：奖励、云端使用率、延迟
    metrics = ['final_reward', 'avg_cloud_rate', 'avg_latency']
    metric_labels = ['Reward', 'Cloud Usage', 'Latency']
    
    # 为每个指标计算自定义范围，以扩大区别
    metric_ranges = {}
    
    # 奖励：越高越好，使用全局范围但稍微扩大
    reward_min = df['final_reward'].min() * 0.9  # 稍微低于最小值
    reward_max = df['final_reward'].max() * 1.1  # 稍微高于最大值
    metric_ranges['final_reward'] = (reward_min, reward_max)
    
    # 云端使用率：越低越好，使用全局范围但稍微扩大
    cloud_min = df['avg_cloud_rate'].min() * 0.9
    cloud_max = df['avg_cloud_rate'].max() * 1.1
    metric_ranges['avg_cloud_rate'] = (cloud_min, cloud_max)
    
    # 延迟：越低越好，使用全局范围但稍微扩大
    latency_min = df['avg_latency'].min() * 0.9
    latency_max = df['avg_latency'].max() * 1.1
    metric_ranges['avg_latency'] = (latency_min, latency_max)
    
    # 为每种权重配置创建雷达图
    for weight_config in df['reward_weights'].unique():
        weight_data = df[df['reward_weights'] == weight_config]
        weights_str = "_".join([f"{w:.1f}" for w in weight_config])
        weights_label = f"W={weight_config[0]}, L={weight_config[1]}, U={weight_config[2]}, C={weight_config[3]}"
        
        # 为每种到达率创建雷达图
        for rate in [2.0, 10.0]:
            rate_data = weight_data[weight_data['arrival_rate'] == rate]
            
            if rate_data.empty:
                continue
                
            # 创建雷达图
            categories = metric_labels
            N = len(categories)
            
            # 计算每个角度的坐标
            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles += angles[:1]  # 闭合图形
            
            plt.figure(figsize=(10, 10))
            ax = plt.subplot(111, polar=True)
            
            # 绘制每种模型的雷达图
            colors = ['blue', 'green', 'red']
            for i, (_, row) in enumerate(rate_data.iterrows()):
                # 归一化每个指标到0-1范围
                reward_norm = (row['final_reward'] - metric_ranges['final_reward'][0]) / (
                    metric_ranges['final_reward'][1] - metric_ranges['final_reward'][0])
                
                # 云端使用率需要反向（越低越好）
                cloud_norm = 1 - (row['avg_cloud_rate'] - metric_ranges['avg_cloud_rate'][0]) / (
                    metric_ranges['avg_cloud_rate'][1] - metric_ranges['avg_cloud_rate'][0])
                
                # 延迟需要反向（越低越好）
                latency_norm = 1 - (row['avg_latency'] - metric_ranges['avg_latency'][0]) / (
                    metric_ranges['avg_latency'][1] - metric_ranges['avg_latency'][0])
                
                values = [reward_norm, cloud_norm, latency_norm]
                values += values[:1]  # 闭合图形
                
                ax.plot(angles, values, color=colors[i], linewidth=2, label=row['model_type'])
                ax.fill(angles, values, color=colors[i], alpha=0.25)
            
            # 添加类别标签
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            
            # 添加径向标签
            ax.set_rlabel_position(0)
            plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=10)
            ax.set_ylim(0, 1)
            
            plt.title(f'Performance Radar Chart\n{weights_label}, Rate={rate}/s', size=15, y=1.05)
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            
            # 保存图像到radar文件夹
            safe_weights_str = weights_str.replace('.', '_')
            radar_filename = f"radar_rate{rate}_weights{safe_weights_str}.png"
            plt.savefig(os.path.join(radar_dir, radar_filename), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 为每个雷达图创建数据表格
            table_data = []
            for _, row in rate_data.iterrows():
                table_data.append([
                    row['model_type'],
                    f"{row['final_reward']:.2f}",
                    f"{row['avg_cloud_rate']:.3f}",
                    f"{row['avg_latency']:.2f}"
                ])
            
            # 创建并保存数据表格
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.axis('tight')
            ax.axis('off')
            table = ax.table(
                cellText=table_data,
                colLabels=['Model', 'Reward', 'Cloud Usage', 'Latency'],
                cellLoc='center',
                loc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            plt.title(f'Performance Data\n{weights_label}, Rate={rate}/s')
            
            # 保存数据表格
            table_filename = f"table_rate{rate}_weights{safe_weights_str}.png"
            plt.savefig(os.path.join(radar_dir, table_filename), dpi=300, bbox_inches='tight')
            plt.close()

def main():
    """主函数：执行所有分析任务"""
    print("开始分析奖励函数参数实验结果...")
    
    # 1. 提取实验数据
    print("提取实验数据...")
    experiment_data = extract_experiment_data()
    
    # 2. 创建DataFrame
    print("创建数据摘要...")
    df = create_summary_dataframe(experiment_data)
    df.to_csv(os.path.join(results_folder, "experiment_summary.csv"), index=False)
    
    # 3. 绘制性能对比图
    print("绘制性能对比图...")
    
    # 成功率对比
    plot_performance_by_weights(
        df, 
        'avg_success_rate', 
        'Success Rate', 
        'Success Rate', 
        'success_rate_by_weights.png'
    )
    
    # 奖励对比
    plot_performance_by_weights(
        df, 
        'final_reward', 
        'Final Reward', 
        'Reward', 
        'reward_by_weights.png'
    )
    
    # 云端使用率对比
    plot_performance_by_weights(
        df, 
        'avg_cloud_rate', 
        'Cloud Usage Rate', 
        'Cloud Usage Rate', 
        'cloud_usage_by_weights.png'
    )
    
    # 延迟对比
    plot_performance_by_weights(
        df, 
        'avg_latency', 
        'Average Latency', 
        'Latency (ms)', 
        'latency_by_weights.png'
    )
    
    # 4. 绘制柱状图比较
    plot_bar_comparison(
        df, 
        'avg_success_rate', 
        'Success Rate', 
        'Success Rate', 
        'success_rate_comparison.png'
    )
    
    plot_bar_comparison(
        df, 
        'final_reward', 
        'Final Reward', 
        'Reward', 
        'reward_comparison.png'
    )
    
    # 5. 绘制训练曲线
    print("绘制训练曲线...")
    plot_training_curves()
    
    # 6. 创建雷达图
    print("创建雷达图...")
    create_radar_chart(df, "performance_radar.png")
    
    # 7. 创建总结报告
    print("创建总结报告...")
    with open(os.path.join(results_folder, "summary_report.txt"), 'w') as f:
        f.write("奖励函数参数实验总结报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("实验配置:\n")
        f.write(f"- 模型类型: GNN, GNN+LSTM3, GNN+LSTM10\n")
        f.write(f"- 到达率: 2.0/s, 10.0/s\n")
        f.write(f"- 奖励权重配置: 5种不同组合\n\n")
        
        f.write("性能指标摘要:\n")
        for rate in [2.0, 10.0]:
            rate_data = df[df['arrival_rate'] == rate]
            f.write(f"\n到达率 {rate}/s:\n")
            
            for model_type in ['GNN', 'GNN+LSTM3', 'GNN+LSTM10']:
                model_data = rate_data[rate_data['model_type'] == model_type]
                f.write(f"  {model_type}:\n")
                f.write(f"    平均成功率: {model_data['avg_success_rate'].mean():.3f}\n")
                f.write(f"    平均奖励: {model_data['final_reward'].mean():.3f}\n")
                f.write(f"    平均云端使用率: {model_data['avg_cloud_rate'].mean():.3f}\n")
                f.write(f"    平均延迟: {model_data['avg_latency'].mean():.3f}\n")
    
    print(f"分析完成! 结果保存在: {results_folder}")

if __name__ == "__main__":
    main()