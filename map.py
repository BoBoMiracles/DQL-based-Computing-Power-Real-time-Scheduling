import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.colors as mcolors
from geopy.distance import geodesic

def visualize_network_topology(bs_csv_path, room_csv_path, output_file="network_topology.png"):
    # 加载数据
    bs_df = pd.read_csv(bs_csv_path)
    room_df = pd.read_csv(room_csv_path)
    
    # 过滤经度范围 (113.5 < x_coord < 114.5)
    bs_df = bs_df[(bs_df['x_coord'] > 113.5) & (bs_df['x_coord'] < 114.5)]
    room_df = room_df[(room_df['x_coord'] > 113.5) & (room_df['x_coord'] < 114.5)]
    
    # 计算经纬度范围
    min_lat = min(bs_df['y_coord'].min(), room_df['y_coord'].min())
    max_lat = max(bs_df['y_coord'].max(), room_df['y_coord'].max())
    min_lon = min(bs_df['x_coord'].min(), room_df['x_coord'].min())
    max_lon = max(bs_df['x_coord'].max(), room_df['x_coord'].max())
    
    # 创建图形
    plt.figure(figsize=(15, 12))
    ax = plt.gca()
    
    # 设置边界
    padding = 0.1 * (max_lon - min_lon)
    ax.set_xlim(min_lon - padding, max_lon + padding)
    ax.set_ylim(min_lat - padding, max_lat + padding)
    
    # 添加标题和标签
    plt.title("Computing Network Topology (113.5 < Longitude < 114.5)", fontsize=16)
    plt.xlabel("Longitude (x_coord)", fontsize=12)
    plt.ylabel("Latitude (y_coord)", fontsize=12)
    
    # 绘制基站
    for _, row in bs_df.iterrows():
        plt.scatter(
            row['x_coord'], row['y_coord'],
            s=50, color='blue', marker='^', alpha=0.7,
            edgecolor='black', linewidth=0.5
        )
        # 添加基站ID标签
        # plt.text(
        #     row['x_coord'] + 0.001, row['y_coord'] + 0.001,
        #     f"BS{row['id']}", fontsize=8, ha='left', va='bottom'
        # )
    
    # 绘制机房 - 根据算力板数量调整大小和颜色
    max_boards = room_df['allocated_boards'].max()
    min_boards = room_df['allocated_boards'].min()
    
    # 创建颜色映射
    norm = Normalize(vmin=min_boards, vmax=max_boards)
    cmap = plt.cm.viridis
    
    for _, row in room_df.iterrows():
        # 计算大小和颜色
        size = 100 + 500 * (row['allocated_boards'] - min_boards) / (max_boards - min_boards + 1e-5)
        color = cmap(norm(row['allocated_boards']))
        
        plt.scatter(
            row['x_coord'], row['y_coord'],
            s=size, color=color, marker='s', alpha=0.8,
            edgecolor='black', linewidth=1.5
        )
        # 添加机房ID和算力板数量标签
        # plt.text(
        #     row['x_coord'] + 0.001, row['y_coord'] + 0.001,
        #     f"Room{row['id']}\n({row['allocated_boards']} boards)",
        #     fontsize=9, ha='left', va='bottom'
        # )
    
    # 添加连接线 - 基站到归属机房
    for _, bs_row in bs_df.iterrows():
        # 查找匹配的归属机房 - 同时匹配ID和位置
        home_room_mask = (
            (room_df['id'] == bs_row['home_compute_node_id']) &
            (abs(room_df['y_coord'] - bs_row['home_y_coord']) < 1e-5) &
            (abs(room_df['x_coord'] - bs_row['home_x_coord']) < 1e-5)
        )
        
        home_room = room_df[home_room_mask]
        
        if not home_room.empty:
            home_room = home_room.iloc[0]
            plt.plot(
                [bs_row['x_coord'], home_room['x_coord']],
                [bs_row['y_coord'], home_room['y_coord']],
                'g-', linewidth=0.8, alpha=0.5
            )
    
    # 添加连接线 - 基站到分配机房
    for _, bs_row in bs_df.iterrows():
        # 查找匹配的分配机房 - 同时匹配ID和位置
        assigned_room_mask = (
            (room_df['id'] == bs_row['assigned_compute_node_id']) &
            (abs(room_df['y_coord'] - bs_row['assigned_y_coord']) < 1e-5) &
            (abs(room_df['x_coord'] - bs_row['assigned_x_coord']) < 1e-5)
        )
        
        assigned_room = room_df[assigned_room_mask]
        
        if not assigned_room.empty:
            assigned_room = assigned_room.iloc[0]
            plt.plot(
                [bs_row['x_coord'], assigned_room['x_coord']],
                [bs_row['y_coord'], assigned_room['y_coord']],
                'r--', linewidth=0.8, alpha=0.7
            )
    
    # 添加图例
    legend_elements = [
        plt.Line2D([0], [0], marker='^', color='w', label='Base Station',
                   markerfacecolor='blue', markersize=10),
        plt.Line2D([0], [0], marker='s', color='w', label='Compute Room',
                   markerfacecolor='gray', markersize=10),
        plt.Line2D([0], [0], color='green', lw=2, label='Home Room Connection'),
        plt.Line2D([0], [0], color='red', lw=2, linestyle='--', label='Assigned Room Connection')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # 添加颜色条表示算力板数量
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7)
    cbar.set_label('Allocated Compute Boards', fontsize=12)
    
    # 添加比例尺
    scale_lon = min_lon + 0.1 * (max_lon - min_lon)
    scale_lat = min_lat + 0.05 * (max_lat - min_lat)
    scale_km = 10  # 10公里比例尺
    
    # 计算10公里在经度上的大致距离
    # 使用geodesic计算实际距离
    point1 = (scale_lat, scale_lon)
    point2 = (scale_lat, scale_lon + 0.1)  # 初始猜测
    actual_dist = geodesic(point1, point2).km
    
    # 调整经度差以达到10公里
    lon_delta = 10 * 0.1 / actual_dist
    
    # plt.plot(
    #     [scale_lon, scale_lon + lon_delta],
    #     [scale_lat, scale_lat],
    #     'k-', linewidth=2
    # )
    # plt.text(
    #     scale_lon + lon_delta/2, scale_lat - 0.005,
    #     '10 km', fontsize=10, ha='center'
    # )
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Network topology visualization saved to {output_file}")

# 使用示例
if __name__ == "__main__":
    bs_file = "gurobi_solution_service_sources_real.csv"
    room_file = "gurobi_solution_compute_nodes_real.csv"
    visualize_network_topology(bs_file, room_file)