import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

def generate_network_data(num_rooms=10, num_base_stations=50, p_main_station=0.2, coordinate=80):
    """生成机房和基站数据
    参数：
        num_rooms: 机房数量 (默认10)
        num_base_stations: 基站总数 (默认50)
        p_main_station: 基站中主基站的比例 (默认20%)
        coordinate: 生成的机房和基站的坐标范围 (默认0-80)
    """
    # 生成随机机房
    rooms = []
    for i in range(num_rooms):
        # 决定是否分配额外算力 (30%概率分配)
        has_extra_compute = random.choices([0, 1], weights=[0.7, 0.3])[0]
        extra_compute = random.randint(100, 300) if has_extra_compute else 0
        
        room = {
            "room_id": f"room{i+1:02d}",
            "pos_x": round(random.uniform(0, coordinate), 2),
            "pos_y": round(random.uniform(0, coordinate), 2),
            "has_extra_compute": has_extra_compute,
            "extra_compute_power": extra_compute
        }
        rooms.append(room)
    rooms_df = pd.DataFrame(rooms)

    # 生成基站数据
    base_stations = []
    for j in range(num_base_stations):
        # 随机生成基站位置
        bs_x = round(random.uniform(0, coordinate), 2)
        bs_y = round(random.uniform(0, coordinate), 2)
        
        # 找到最近机房
        min_dist = float('inf')
        nearest_room = None
        for _, room in rooms_df.iterrows():
            dist = np.sqrt((bs_x - room['pos_x'])**2 + (bs_y - room['pos_y'])**2)
            if dist < min_dist:
                min_dist = dist
                nearest_room = room
        
        # 生成基站属性
        is_main = random.choices([0, 1], weights=[1-p_main_station, p_main_station])[0]  # 30%概率是主基站
        base_station = {
            "bs_id": f"bs{j+1:03d}",
            "pos_x": bs_x,
            "pos_y": bs_y,
            "room_id": nearest_room['room_id'],
            "type": is_main,
            "compute_power": random.randint(50, 200) if is_main else 0,
            "bandwidth": random.choice([800, 1000, 1200]),
            "latency": random.randint(3, 8)
        }
        base_stations.append(base_station)
    
    return rooms_df, pd.DataFrame(base_stations)

def visualize_generated_data(rooms_df, base_stations_df):
    plt.figure(figsize=(10, 10))
    
    # 绘制机房
    plt.scatter(rooms_df['pos_x'], rooms_df['pos_y'], 
                c='red', s=200, marker='s', label='Rooms')
    for _, row in rooms_df.iterrows():
        plt.text(row['pos_x'], row['pos_y']+2, row['room_id'], ha='center')
    
    # 绘制基站
    main_bs = base_stations_df[base_stations_df['type'] == 1]
    normal_bs = base_stations_df[base_stations_df['type'] == 0]
    plt.scatter(main_bs['pos_x'], main_bs['pos_y'], 
                c='blue', s=80, label='Main BS')
    plt.scatter(normal_bs['pos_x'], normal_bs['pos_y'],
                c='grey', s=50, label='Normal BS')
    
    # 绘制连接线
    for _, bs in base_stations_df.iterrows():
        room = rooms_df[rooms_df['room_id'] == bs['room_id']].iloc[0]
        plt.plot([bs['pos_x'], room['pos_x']], 
                 [bs['pos_y'], room['pos_y']], 
                 'g-', alpha=0.1)
    
    plt.title("Generated Network Topology")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # 生成示例数据（20个机房，100个基站）
    rooms_df, base_stations_df = generate_network_data(20, 100, 0.2, 80)

    # 创建简化版的基站文件（只包含要求的字段）
    simplified_base_stations = base_stations_df[['bs_id', 'pos_x', 'pos_y', 'room_id']]
    
    # 保存为CSV文件
    rooms_df.to_csv("rooms.csv", index=False)
    simplified_base_stations.to_csv("base_stations.csv", index=False)

    print("生成数据示例：")
    print("\n机房数据：")
    print(rooms_df.head(3))
    print("\n基站数据：")
    print(simplified_base_stations.sample(3))

    visualize_generated_data(rooms_df, base_stations_df)