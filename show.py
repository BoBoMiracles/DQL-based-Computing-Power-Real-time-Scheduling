import pandas as pd
import simulator
from simulator import ComputingNetworkSimulator
import random
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # 初始化环境
    env = ComputingNetworkSimulator('base_stations.csv', 'rooms.csv')
    env.setup_visualization()

    # 运行演示
    state = env.reset()
    
    try:
        state = env.reset()
        for _ in range(100):  # 运行100步
            
            #初步考虑可选动作范围
            #1.随机选择所有可选动作
            valid_actions = env.get_valid_actions()
            if valid_actions:  # 确保有可用动作
                action = random.choice(valid_actions)
            else:
                action = {'target_type': 'cloud', 'target_id': 'cloud'}  # 默认回退到云端
            
            #2.智能动作采样（考虑资源可用性）
            #action = env.sample_action()
            
            next_state, reward, done, _ = env.step(action)
            env.update_visualization()
            if done:
                break
            plt.pause(0.1)  # 控制刷新率
    finally:
        plt.ioff()
        plt.show()

    
    