from environment_simulation import ComputingNetworkSimulator
from utils import StateTransformer
from gnn_model import GNNPolicy
import torch
import torch.nn.functional as F


# 测试前向传播
def test_forward_pass():
    env = ComputingNetworkSimulator('base_stations.csv', 'rooms.csv')
    state = env.reset()
    transformer = StateTransformer(env)
    graph_data = transformer.state_to_graph(state)
    
    print(f"节点数量: {graph_data.num_nodes}")
    print(f"边数量: {graph_data.num_edges}")
    print(f"节点特征维度: {graph_data.num_node_features}")
    print(f"边特征维度: {graph_data.num_edge_features}")
    
    model = GNNPolicy()
    
    # 测试各层输出
    x = model.node_enc(graph_data.x)
    print(f"节点编码后形状: {x.shape}")
    
    edge_attr = model.edge_enc(graph_data.edge_attr)
    print(f"边编码后形状: {edge_attr.shape}")
    
    # 第一层GAT
    x = model.conv1(x, graph_data.edge_index, edge_attr=edge_attr)
    x = F.relu(x)
    print(f"第一层GAT后形状: {x.shape}")
    
    # 第二层GAT
    x = model.conv2(x, graph_data.edge_index, edge_attr=edge_attr)
    x = F.relu(x)
    print(f"第二层GAT后形状: {x.shape}")
    
    # 全局池化
    global_feat = model.pool(x, graph_data.batch)
    print(f"全局池化后形状: {global_feat.shape}")  # 应为 [1, hidden_dim]
    
    # 全局特征复制
    global_feat_per_node = global_feat[graph_data.batch]
    print(f"复制到节点后形状: {global_feat_per_node.shape}")
    
    # 特征拼接
    x = torch.cat([x, global_feat_per_node], dim=1)
    print(f"特征拼接后形状: {x.shape}")
    
    # 最终输出
    output = model.q_head(x)
    print(f"最终输出形状: {output.shape}")

if __name__ == "__main__":
    test_forward_pass()