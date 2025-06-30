from simulator import ComputingNetworkSimulator
from dqn_agent import GNNAgent
import torch

def run():
    # 初始化
    env = ComputingNetworkSimulator('base_stations.csv', 'rooms.csv')
    agent = GNNAgent(env)
    agent.policy_net.load_state_dict(torch.load('gnn_dqn.pth'))
    
    # 运行
    state = env.reset()
    done = False
    while not done:
        action = agent.get_action(state, epsilon=0.01)  # 小概率探索
        next_state, reward, done, _ = env.step(action)
        env.update_visualization()
        state = next_state

if __name__ == "__main__":
    run()