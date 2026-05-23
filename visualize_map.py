import matplotlib.pyplot as plt
import networkx as nx
from simulator import ComputingNetworkSimulator
import numpy as np

def visualize_topology_and_verify(bs_csv, room_csv):
    """
    实例化模拟器并可视化当前的物理拓扑结构
    """
    print("Initializing Simulator for Visualization...")
    # 使用较小的参数初始化，只为了获取静态数据
    env = ComputingNetworkSimulator(
        bs_csv_path=bs_csv,
        room_csv_path=room_csv,
        rate=1, 
        simulation_time=10
    )
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 1. 提取数据
    rooms = env.nodes['rooms']
    bss = env.nodes['base_stations']
    
    # 2. 绘制连接线 (基站 -> 所属机房)
    print("Drawing connections...")
    for bs_id, bs in bss.items():
        room_id = bs['room_id']
        if room_id in rooms:
            room = rooms[room_id]
            # 绘制灰色虚线
            ax.plot([bs['position'][0], room['position'][0]], 
                    [bs['position'][1], room['position'][1]], 
                    color='gray', alpha=0.3, linewidth=0.5, zorder=1)
    
    # 3. 绘制基站 (Base Stations)
    bs_x = [bs['position'][0] for bs in bss.values()]
    bs_y = [bs['position'][1] for bs in bss.values()]
    ax.scatter(bs_x, bs_y, c='blue', s=10, alpha=0.6, label='Base Station', zorder=2)
    
    # 4. 绘制机房 (Compute Nodes)
    room_x = [r['position'][0] for r in rooms.values()]
    room_y = [r['position'][1] for r in rooms.values()]
    ax.scatter(room_x, room_y, c='red', s=50, marker='s', edgecolors='black', label='Edge Data Center', zorder=3)
    
    # 5. 绘制边界框
    ax.add_patch(plt.Rectangle(
        (env.min_x, env.min_y), 
        env.max_x - env.min_x, 
        env.max_y - env.min_y, 
        fill=False, color='green', linestyle='--', label='Simulation Boundary'
    ))
    
    # 设置图表
    ax.set_title(f'Network Topology (BS: {len(bss)}, Rooms: {len(rooms)})')
    ax.set_xlabel('Relative X (km)')
    ax.set_ylabel('Relative Y (km)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal') # 保持比例，确保地图不拉伸
    
    plt.tight_layout()
    plt.show()
    
    # 打印统计信息
    print(f"\nTopology Stats:")
    print(f"Total Base Stations: {len(bss)}")
    print(f"Total Compute Rooms: {len(rooms)}")
    print(f"Map Size: {env.max_x - env.min_x:.2f} km * {env.max_y - env.min_y:.2f} km")
    
    # 验证是否有孤立基站
    isolated_bs = [bs_id for bs_id, bs in bss.items() if bs['room_id'] not in rooms]
    if isolated_bs:
        print(f"WARNING: Found {len(isolated_bs)} base stations with missing home rooms! IDs: {isolated_bs[:5]}...")
    else:
        print("All base stations are correctly connected to valid rooms.")

if __name__ == "__main__":
    # 替换为你的文件名
    visualize_topology_and_verify('gurobi_solution_service_sources_real.csv', 'gurobi_solution_compute_nodes_real.csv') # 假设你的基站和机房都在这一个文件，或者分别传入