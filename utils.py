import torch
from torch_geometric.data import Data, Batch

class StateTransformer:
    """环境状态到图数据的转换器"""
    def __init__(self, env):
        self.env = env
        self.node_types = {'cloud':0, 'room':1, 'main':2, 'normal':3}
    
    def state_to_graph(self, state):
        """将环境状态转换为图数据"""
        node_feats = []
        edge_indices = []
        edge_feats = []
        
        # 添加节点
        node_feats.append(self._get_node_feature(self.env.cloud_node))  # 云端
        for room in self.env.nodes['rooms'].values():
            node_feats.append(self._get_node_feature(room))
        for bs in self.env.nodes['base_stations'].values():
            node_feats.append(self._get_node_feature(bs))
        
        # 构建边
        # 机房-云端连接
        for i, room in enumerate(self.env.nodes['rooms'].values(), start=1):
            edge_indices.append([0, i])  # cloud到room
            edge_indices.append([i, 0])  # room到cloud (双向连接)
            edge_feats.append(self._get_edge_feature(self.env.cloud_node, room))
            edge_feats.append(self._get_edge_feature(room, self.env.cloud_node))
        
        # 基站-机房连接
        bs_start_idx = 1 + len(self.env.nodes['rooms'])
        for i, bs in enumerate(self.env.nodes['base_stations'].values(), start=bs_start_idx):
            room_idx = 1 + list(self.env.nodes['rooms'].keys()).index(bs['room_id'])
            edge_indices.append([i, room_idx])
            edge_indices.append([room_idx, i])  # 双向连接
            edge_feats.append(self._get_edge_feature(bs, self.env.nodes['rooms'][bs['room_id']]))
            edge_feats.append(self._get_edge_feature(self.env.nodes['rooms'][bs['room_id']], bs))
        
        # 创建批次信息 - 所有节点属于同一个图
        batch = torch.zeros(len(node_feats), dtype=torch.long)
        
        return Data(
            x=torch.stack(node_feats),
            edge_index=torch.tensor(edge_indices).t().contiguous(),
            edge_attr=torch.stack(edge_feats),
            batch=batch  # 添加批次信息
        )
    
    def _get_node_feature(self, node):
        """节点特征向量"""
        feat = [
            node['position'][0]/100,  # 归一化坐标
            node['position'][1]/100,
            node.get('compute',0)/200 if node['type']!='cloud' else 0,  # 算力利用率
            node.get('bandwidth',0)/8000,  # 带宽利用率
            node.get('latency',0)/35,       # 基础延迟
            self.node_types[node['type']]/3  # 节点类型
        ]
        return torch.tensor(feat, dtype=torch.float32)
    
    def _get_edge_feature(self, src, dst):
        """边特征向量"""
        dx = src['position'][0] - dst['position'][0]
        dy = src['position'][1] - dst['position'][1]
        distance = (dx**2 + dy**2)**0.5 / 141.42  # 最大可能距离(100√2)
        
        # 连接类型权重
        conn_type = 1.0 if ('cloud' in [src['type'], dst['type']]) else 0.5
        return torch.tensor([distance, conn_type], dtype=torch.float32)