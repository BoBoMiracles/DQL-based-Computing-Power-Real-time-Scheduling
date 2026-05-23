import torch
from simulator import ComputingNetworkSimulator
from gnn_dqn_agent import GNNAgent
from gnn_lstm_dqn_agent import GNNLSTMDQNAgent
from gnn_lstm_ddqn_agent import GNNLSTMDDQNAgent
from gnn_transformer_ddqn_agent import GNNTransformerDDQNAgent
import numpy as np
import time
import argparse
import os
import matplotlib.pyplot as plt

episodes = 200  # 总训练轮次

def save_topology_map(env, save_folder):
    """
    绘制并保存当前环境的拓扑结构图到指定文件夹
    """
    print(f"Generating topology map to {save_folder}...")
    
    # 使用 Agg 后端，防止在无显示器的服务器上报错
    plt.switch_backend('Agg') 
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    rooms = env.nodes['rooms']
    bss = env.nodes['base_stations']
    
    # 1. 绘制连接线 (灰色虚线)
    for bs_id, bs in bss.items():
        room_id = bs['room_id']
        if room_id in rooms:
            room = rooms[room_id]
            ax.plot([bs['position'][0], room['position'][0]], 
                    [bs['position'][1], room['position'][1]], 
                    color='gray', alpha=0.3, linewidth=0.5, zorder=1)
    
    # 2. 绘制基站 (蓝色点)
    if bss:
        bs_x = [bs['position'][0] for bs in bss.values()]
        bs_y = [bs['position'][1] for bs in bss.values()]
        ax.scatter(bs_x, bs_y, c='blue', s=10, alpha=0.6, label='Base Station', zorder=2)
    
    # 3. 绘制机房 (红色方块)
    if rooms:
        room_x = [r['position'][0] for r in rooms.values()]
        room_y = [r['position'][1] for r in rooms.values()]
        ax.scatter(room_x, room_y, c='red', s=50, marker='s', edgecolors='black', label='Edge Room', zorder=3)
    
    # 4. 绘制云端 (如果存在)
    if hasattr(env, 'cloud_node'):
        cloud = env.cloud_node
        ax.scatter(cloud['position'][0], cloud['position'][1], c='orange', s=100, marker='*', label='Cloud', zorder=4)

    # 5. 绘制边界
    rect = plt.Rectangle(
        (env.min_x, env.min_y), 
        env.max_x - env.min_x, 
        env.max_y - env.min_y, 
        fill=False, color='green', linestyle='--', label='Boundary'
    )
    ax.add_patch(rect)
    
    # 设置标题和标签
    ax.set_title(f'Network Topology (BS: {len(bss)}, Rooms: {len(rooms)})')
    ax.set_xlabel('Relative X (km)')
    ax.set_ylabel('Relative Y (km)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # 保存图片
    save_path = os.path.join(save_folder, 'topology_map.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig) # 关闭图表释放内存
    print(f"Topology map saved to {save_path}")

def print_topology_stats(env):
    """
    打印过滤和坐标转换后，用于仿真的基站和机房数量
    """
    num_bss = len(env.nodes['base_stations'])
    num_rooms = len(env.nodes['rooms'])
    
    print("\n--- 拓扑结构统计 (过滤后) ---")
    print(f"✅ 过滤后用于训练的基站数量 (Base Stations): {num_bss}")
    print(f"✅ 过滤后用于训练的机房数量 (Compute Rooms): {num_rooms}")
    print(f"仿真地图大小: {env.max_x - env.min_x:.2f} x {env.max_y - env.min_y:.2f} (已归一化到 100x100)")
    print("-----------------------------\n")
    
    return num_bss, num_rooms

def train(model_type, device, lstm_len, arr_rate, sim_time, folder_name, resume_epoch=0, update_type='soft', tidal_flow=False):
    """
    训练GNN, GNN+LSTM, 或 GNN+Transformer智能体。
    
    Args:
        model_type (str): 模型类型 ('gnn' 或 'gnn_lstm')。
        device (str): 训练设备 ('cuda' 或 'cpu')。
        lstm_len (int): 仅用于GNN+LSTM模型的序列长度。
        arr_rate (float): 环境请求到达率。
        sim_time (int): 仿真时间。
        resume_epoch (int): 从第N个epoch的模型继续训练 (N > 0)。
    """
    # 0. 初始化环境，确定仿真环境的请求到达率和文件夹名称
    env = ComputingNetworkSimulator('gurobi_solution_service_sources_sim.csv', 
                                       'gurobi_solution_compute_nodes_sim.csv', 
                                       rate = arr_rate, 
                                       simulation_time = sim_time,
                                       tidal_flow=tidal_flow)
    rate = env.base_rate

    tidal_suffix = "_tidal" if tidal_flow else ""
    
    # 根据模型类型选择智能体和文件夹名称
    if model_type == 'gnn_lstm':
        agent = GNNLSTMDDQNAgent(env, device=device, sequence_length=lstm_len, update_type=update_type)
        length = agent.sequence_length
        folder_name = f'{folder_name}/gnn_lstm{length}_model_rate{rate}{tidal_suffix}'
        print(f"Training GNN+LSTM (Len={length}) model at rate {rate}{tidal_suffix}...")
    elif model_type == 'gnn':
        agent = GNNAgent(env, device=device, update_type=update_type)
        folder_name = f'{folder_name}/gnn_model_rate{rate}{tidal_suffix}'
        print(f"Training GNN model at rate {rate}{tidal_suffix}...")
    elif model_type == 'gnn_transformer': 
        agent = GNNTransformerDDQNAgent(env=env, sequence_length=lstm_len, device=device, update_type=update_type)
        length = agent.sequence_length
        folder_name = f'{folder_name}/gnn_transformer{length}_model_rate{rate}{tidal_suffix}'
        print(f"Training GNN+Transformer (Len={length}) model at rate {rate}{tidal_suffix}...")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    os.makedirs(folder_name, exist_ok=True)

    num_bss, num_rooms = print_topology_stats(env)

    try:
        save_topology_map(env, folder_name)
    except Exception as e:
        print(f"Warning: Failed to save topology map. Error: {e}")

    # 1. 训练参数和断点续训准备
    epsilon_start = 1.0
    epsilon_end = 0.01
    ANNEALING_STEPS = 300000
    
    start_epoch = 0
    rewards_history = []
    loss_history = []
    global_step_counter = 0 # 用于控制 epsilon 衰减和学习率调度
    initial_epsilon = epsilon_start

    # 新增：完整的断点续训逻辑
    if resume_epoch > 0:
        # 构建状态文件路径
        state_file = os.path.join(folder_name, f"{model_type}_agent_state_ep{resume_epoch}.pth")
        reward_file = os.path.join(folder_name, f'{model_type}_rewards_history.npy')
        loss_file = os.path.join(folder_name, f'{model_type}_loss_history.npy')
        step_file = os.path.join(folder_name, f'{model_type}_global_step.npy')
        
        if os.path.exists(state_file):
            print(f"🔄 RESUME: Loading complete training state from {state_file}")
            
            # 加载智能体完整状态
            saved_state = agent.load_state(state_file, env=env)
            
            # 加载训练历史
            if os.path.exists(reward_file):
                rewards_history = np.load(reward_file).tolist()
                # 只保留resume_epoch之前的历史
                rewards_history = rewards_history[:resume_epoch]
                
            if os.path.exists(step_file):
                # 从文件恢复步数计数器（最准确）
                file_global_step = int(np.load(step_file))
                print(f"Restored global_step_counter from file: {file_global_step}")
                
                # 比较两个来源的步数，确保一致性
                state_global_step = saved_state.get('steps_done', 0)
                if file_global_step != state_global_step:
                    print(f"⚠️ Step count mismatch: file={file_global_step}, state={state_global_step}")
                    # 优先使用文件中的步数（更可靠）
                    global_step_counter = file_global_step
                else:
                    global_step_counter = file_global_step
            else:
                # 回退到从state中恢复
                global_step_counter = saved_state.get('steps_done', 0)

            if os.path.exists(loss_file):
                # 3. 恢复 Loss 历史：精确截断
                loss_history = np.load(loss_file).tolist()
                
                # 核心改进：Loss 的条数应该等于模型更新的总次数 (global_step_counter)
                if len(loss_history) > global_step_counter:
                    loss_history = loss_history[:global_step_counter]
                    print(f"✂️ Scaled loss_history to match global_step: {len(loss_history)}")
                else:
                    print(f"Restored loss_history: {len(loss_history)} items")
            
            # 从保存的状态中恢复训练参数
            global_step_counter = saved_state.get('steps_done', 0)
            start_epoch = resume_epoch
            
            # 计算续训时的epsilon（保持连续性）
            if global_step_counter > 0:
                initial_epsilon = max(epsilon_end, 
                    epsilon_start - (epsilon_start - epsilon_end) * min(1.0, global_step_counter / ANNEALING_STEPS))
            
            print(f"✅ RESUME: Successfully resumed from episode {start_epoch}")
            print(f"   Global steps: {global_step_counter}, Memory size: {len(agent.memory)}")
            print(f"   Initial epsilon: {initial_epsilon:.4f}")
            
        else:
            print(f"❌ RESUME: State file not found at {state_file}")
            print("Starting training from scratch...")
            resume_epoch = 0
            start_epoch = 0
    
    start_time = time.time()

    # 2. 训练循环 (从 start_epoch 开始)
    for ep in range(start_epoch, episodes):
        state = env.reset()
        
        # 重置序列模型的内部缓冲区/状态
        if hasattr(agent, 'reset_episode'):
            agent.reset_episode()
            
        total_reward = 0
        done = False
        episode_losses = []
        
        while not done:
            epsilon = max(epsilon_end, 
                      epsilon_start - (epsilon_start - epsilon_end) * min(1.0, global_step_counter / ANNEALING_STEPS))
            action = agent.get_action(state, epsilon)
            next_state, reward, done, metrics = env.step(action)
            total_reward += reward
            
            # 记忆步骤
            if model_type in ['gnn_lstm', 'gnn_transformer']:
                agent.remember_step(state, action, reward, next_state, done)
            else:
                agent.remember(state, action, reward, next_state, done)
            
            # 模型更新 (现在包含软更新)
            # 仅当经验回放缓冲区有足够经验时才训练
            min_memory = agent.batch_size
            
            if len(agent.memory) > min_memory: 
                loss = agent.update_model()
                if loss is not None:
                    episode_losses.append(loss)
                    loss_history.append(loss)
                    global_step_counter += 1
                    # 更新智能体的步数计数器
                    if hasattr(agent, 'steps_done'):
                        agent.steps_done = global_step_counter
                
            state = next_state
        
        # 3. 记录和保存
        avg_episode_loss = np.mean(episode_losses) if episode_losses else 0
        rewards_history.append(total_reward) # 记录当前轮次的奖励
            
        # 调度器更新 (如果使用 ReduceLROnPlateau，需要在外部调用)
        if episode_losses and hasattr(agent, 'scheduler') and isinstance(agent.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            agent.scheduler.step(avg_episode_loss)
            
        elapsed = time.time() - start_time
        
        # 打印轮次从 ep + 1 开始，保持连续性
        len_str = str(lstm_len) if model_type != 'gnn' else "0"
        
        print(f"[{model_type} | Rate:{rate}{tidal_suffix} | Len:{len_str}] "  # <--- 新增：模型类型和时间窗
              f"Ep {ep + 1}/{episodes} | "
              f"Reward: {total_reward:.1f} | "
              f"Loss: {avg_episode_loss:.3f} | "
              f"Epsilon: {epsilon:.4f} | "
              f"Step: {global_step_counter} | "
              f"Time: {elapsed:.1f}s")
        
        # --- 在保存部分 (Save Checkpoint) ---
        if (ep + 1) % 50 == 0:
            current_epoch = ep + 1
            
            # 1. 保存全量状态 (用于断点续训 - 必须)
            state_name = f"{model_type}_agent_state_ep{current_epoch}.pth"
            state_path = os.path.join(folder_name, state_name)
            agent.save_state(state_path)

            # 2. 保存纯模型参数 (用于快速推理/画图 - 可选，建议保留)
            if current_epoch < episodes:
                model_name = f"{model_type}_dqn_ep{current_epoch}.pth"
                model_path = os.path.join(folder_name, model_name)
                torch.save(agent.policy_net.state_dict(), model_path)

            # 保存历史数据
            np.save(os.path.join(folder_name, f'{model_type}_rewards_history.npy'), np.array(rewards_history))
            np.save(os.path.join(folder_name, f'{model_type}_loss_history.npy'), np.array(loss_history))
            np.save(os.path.join(folder_name, f'{model_type}_global_step.npy'), np.array(global_step_counter))
            
            print(f"💾 Checkpoint saved: Ep {current_epoch}")

            # 3. 删除旧文件 (清理逻辑)
            # 保护第 500 轮不被删除
            prev_epoch = current_epoch - 50
            if prev_epoch > 0 and prev_epoch != episodes: # 确保不删第500轮
                
                # 删除旧的全量状态
                prev_state_file = os.path.join(folder_name, f"{model_type}_agent_state_ep{prev_epoch}.pth")
                if os.path.exists(prev_state_file):
                    try:
                        os.remove(prev_state_file)
                        print(f"🗑️ Cleaned up old state: Ep {prev_epoch}")
                    except OSError:
                        pass
                
                # 删除旧的纯模型文件 (同步删除，避免堆积)
                prev_model_file = os.path.join(folder_name, f"{model_type}_dqn_ep{prev_epoch}.pth")
                if os.path.exists(prev_model_file):
                    try:
                        os.remove(prev_model_file)
                        # print(f"🗑️ Cleaned up old model: Ep {prev_epoch}")
                    except OSError:
                        pass

    # 4. 训练结束，保存最终模型和历史
    print("\n--- Training Finished ---")
    final_name = f"{model_type}_dqn_final.pth"
    final_path = os.path.join(folder_name, final_name)
    torch.save(agent.policy_net.state_dict(), final_path)

    # 保存最终训练状态
    # final_state_path = os.path.join(folder_name, f"{model_type}_agent_state_final.pth")
    # agent.save_state(final_state_path)
    
    print(f"✅ Final model (pure weights) saved as: {final_name}")
    print(f"✅ Final training state (full) kept as: {model_type}_agent_state_ep{episodes}.pth")


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Train DQN agent for computing network scheduling')
    parser.add_argument('--model', type=str, default='gnn', choices=['gnn', 'gnn_lstm', 'gnn_transformer'],
                        help='Model type: gnn or gnn_lstm (default: gnn)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device to use for training (default: cuda)')
    parser.add_argument('--resume_epoch', type=int, default=0,
                        help='Epoch number to resume training from (e.g., 200). 0 means start from scratch.')
    parser.add_argument('--update_type', type=str, default='soft', choices=['soft', 'hard'],
                        help='Target network update type: soft or hard (default: soft)')
    parser.add_argument('--tidal_flow', action='store_true',
                        help='Enable tidal flow pattern in request generation')
    args = parser.parse_args()
    
    # 开始训练
    rate = (1, 5, 8) # 您代码中定义的请求率列表
    t = 0
    folder = 'models_revision_v3'
    tidal_s = "_tidal" if args.tidal_flow else ""
    
    # 保持原有的多配置训练逻辑
    if args.model == 'gnn_transformer':
        # Transformer 训练逻辑
        for r in rate:
            # 根据请求率动态调整仿真时间
            t = 1200 if r >= 8 else (7200 if r <= 1 else 1800)    
            # 可以在这里定义要尝试的序列长度，例如 [5, 20]
            for length in (20, 30): 
                print(f"\n--- Starting GNN_Transformer{tidal_s} Training ---")
                print(f"Config: Time Window = {length}, Arrival Rate = {r}{tidal_s}, Sim Time = {t}")
                train(model_type=args.model, 
                      device=args.device, 
                      lstm_len=length, 
                      arr_rate=r, 
                      sim_time=t, 
                      folder_name=folder, 
                      resume_epoch=args.resume_epoch, 
                      update_type=args.update_type,
                      tidal_flow=args.tidal_flow)
    
    elif args.model == 'gnn_lstm':  
        for r in rate:
            # 根据请求率设置仿真时间
            t = 1200 if r >= 8 else (7200 if r <= 1 else 1800)    
            for length in (20, 30):
                print(f"\n--- Starting GNN_LSTM{tidal_s} Training ---")
                print(f"Config: Time Window = {length}, Arrival Rate = {r}{tidal_s}, Sim Time = {t}")
                train(model_type=args.model, 
                      device=args.device, 
                      lstm_len=length, 
                      arr_rate=r, 
                      sim_time=t, 
                      folder_name=folder, 
                      resume_epoch=args.resume_epoch, 
                      update_type=args.update_type,
                      tidal_flow=args.tidal_flow)
    
    elif args.model == 'gnn': 
        for r in rate:
            # 根据请求率设置仿真时间
            t = 1200 if r >= 8 else (7200 if r <= 1 else 1800)    
            print(f"\n--- Starting GNN{tidal_s} Training ---")
            print(f"Config: Arrival Rate = {r}{tidal_s}, Sim Time = {t}")
            train(model_type=args.model, 
                  device=args.device, 
                  lstm_len=0, 
                  arr_rate=r, 
                  sim_time=t, 
                  folder_name=folder, 
                  resume_epoch=args.resume_epoch, 
                  update_type=args.update_type,
                  tidal_flow=args.tidal_flow)