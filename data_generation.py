import pandas as pd

if __name__ == "__main__":
    # 示例数据（保存为base_stations.csv和rooms.csv）
    bs_data = {
        'bs_id': ['bs1', 'bs2', 'bs3', 'bs4'],
        'type': [0, 1, 0, 1],
        'pos_x': [10, 18, 20, 32],
        'pos_y': [10, 24, 20, 6],
        'room_id': ['roomA', 'roomA', 'roomB', 'roomB'],
        'compute_power': [100, 0, 80, 0],
        'bandwidth': [1000, 800, 1200, 900],
        'latency': [5, 8, 6, 7]
    }
    pd.DataFrame(bs_data).to_csv('base_stations.csv', index=False)
    
    room_data = {
        'room_id': ['roomA', 'roomB'],
        'pos_x': [15, 25],
        'pos_y': [15, 25],
        'total_bandwidth': [5000, 6000]
    }
    pd.DataFrame(room_data).to_csv('rooms.csv', index=False)

