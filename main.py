import pandas as pd
import environmental_simulation
from environmental_simulation import ComputingNetworkSimulator
import random
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # 初始化环境
    env = ComputingNetworkSimulator('base_stations.csv', 'rooms.csv')
    env.setup_visualization()

    # 运行演示
    state = env.reset()
    valid_bs = ['bs1', 'bs3']  # 根据实际数据定义可用主基站
    
    try:
        state = env.reset()
        for _ in range(100):  # 运行100步
            action = {
                'target_type': random.choice(['base_stations']),
                'target_id': random.choice(['bs1', 'bs3'])
            }
            next_state, reward, done, _ = env.step(action)
            env.update_visualization()
            if done:
                break
            plt.pause(0.1)  # 控制刷新率
    finally:
        plt.ioff()
        plt.show()

    
    