import torch
from environmental_simulation import ComputingNetworkSimulator
from dqn_agent import GNNAgent
from utils import StateTransformer

def train():
    # 初始化环境
    env = ComputingNetworkSimulator('base_stations.csv', 'rooms.csv')
    agent = GNNAgent(env, device='cpu')
    
    # 训练参数
    episodes = 1000
    target_update = 10  # 目标网络更新间隔
    
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # 选择动作
            action = agent.get_action(state, epsilon=0.1)
            
            # 执行动作
            next_state, reward, done, _ = env.step(action)
            
            # 存储经验
            agent.memory.append( (state, action, reward, next_state, done) )
            
            # 更新模型
            agent.update_model()
            
            # 状态转移
            state = next_state
            total_reward += reward
        
        # 更新目标网络
        if ep % target_update == 0:
            agent.update_target_net()
        
        print(f"Episode {ep+1}, Reward: {total_reward:.1f}")

if __name__ == "__main__":
    train()