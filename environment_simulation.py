import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D 
from collections import deque
from torch_geometric.data import Data
import random

# ======================
# 模拟环境核心类
# ======================
class ComputingNetworkSimulator:
    def __init__(self, bs_csv_path, room_csv_path):
        # 加载原始数据
        self.bs_df = pd.read_csv(bs_csv_path)
        self.bs_df['room_id'] = self.bs_df['room_id'].astype(str)  # 统一类型
        self.room_df = pd.read_csv(room_csv_path)
        self.request_artists = []
        
        # 添加云端节点
        self.cloud_node = {
            'node_id': 'cloud',
            'type': 'cloud',
            'compute': float('inf'),
            'position': (90, 90),  # 云端位置固定
            'bandwidth': float('inf'),
            'latency': 5  # 基础延迟
        }
        
        # 构建节点字典
        self.nodes = {
            'rooms': {row['room_id']: self._create_room_node(row) 
                     for _, row in self.room_df.iterrows()},
            'base_stations': {row['bs_id']: self._create_bs_node(row)
                            for _, row in self.bs_df.iterrows()},
            'cloud': {'cloud': self.cloud_node} 
        }
        
        # 初始化动态状态
        self._reset_dynamic_state()
        
        # 请求生成参数
        self.request_rate = 0.5  # 每秒请求数
        self.current_time = 0
        self.request_counter = 0

    def _create_room_node(self, row):
        """创建机房节点"""
        return {
            'node_id': row['room_id'],
            'type': 'room',
            'position': (row['pos_x'], row['pos_y']),
            'compute': 200,  # 机房自身无算力
            'bandwidth': row['total_bandwidth'],
            'latency': 0,
            'connected_bs': [bs for bs in self.bs_df[self.bs_df.room_id==row['room_id']]['bs_id']]
        }

    def _create_bs_node(self, row):
        """创建基站节点（最终版本）"""
        return {
            'node_id': str(row['bs_id']),
            'type': 'main' if row['type'] == 1 else 'normal',
            'position': (row['pos_x'], row['pos_y']),
            'compute': row['compute_power'] if row['type'] == 1 else 0,
            'max_compute': row['compute_power'] if row['type'] == 1 else 0,
            'bandwidth': row['bandwidth'],
            'latency': row['latency'],
            'room_id': str(row['room_id'])  # 确保该字段存在
        }

    def _reset_dynamic_state(self):
        """重置动态状态"""
        self.request_history = deque(maxlen=1000)
        self.metrics = {
            'total_requests': 0,
            'succeed_requests': 0,
            'cloud_requests': 0,
            'total_latency': 0
        }
        
        # 初始化算力
        for bs in self.nodes['base_stations'].values():
            if bs['type'] == 'main':
                bs['compute'] = bs['max_compute']

    def _generate_request(self):
        """生成新的计算请求（增加调试输出）"""
        self.request_counter += 1
        inter_arrival = np.random.exponential(1/self.request_rate)
        self.current_time += inter_arrival
        
        # 在随机区域生成请求
        position = (
            np.random.uniform(0, 80),
            np.random.uniform(0, 80)
        )
        
        req = {
            'req_id': f"REQ_{self.request_counter}",
            'timestamp': self.current_time,
            'position': position,
            'compute_demand': np.random.randint(1, 10),
            'max_latency': np.random.choice([15, 25, 35])
        }
        # print(f"Debug - 生成请求: Position={position}, Demand={req['compute_demand']}")
        return req

    def _calculate_path_latency(self, position, target_node):
        """计算请求位置到目标节点的路径延迟"""
        # 构造虚拟请求节点
        virtual_node = {
            'node_id': 'virtual_request',
            'position': position,
            'latency': 0,
            'type': 'virtual'
        }
        
        # 找到最近基站
        min_latency = float('inf')
        for bs in self.nodes['base_stations'].values():
            latency = self._calculate_latency(virtual_node, bs)
            if latency < min_latency:
                min_latency = latency
                nearest_bs = bs
        
        # 基站到目标路径
        if target_node['node_id'] == 'cloud':
            # 到云端的路径：基站 -> 机房 -> 云端
            room = self.nodes['rooms'][nearest_bs['room_id']]
            return (min_latency +
                    self._calculate_latency(nearest_bs, room) +
                    self._calculate_latency(room, self.cloud_node))
        else:
            # 本地路径：基站 -> 目标节点
            return min_latency + self._calculate_latency(nearest_bs, target_node)

    def _calculate_latency(self, src, dst):
        """计算两个节点之间的延迟（增强健壮性版本）"""
        try:
            src_pos = src['position']
            dst_pos = dst['position']
            src_lat = src.get('latency', 0)
            dst_lat = dst.get('latency', 0)
        except KeyError as e:
            raise KeyError(f"Missing required key in node: {e}")

        # 云端延迟计算
        if src.get('node_id', '') == 'cloud' or dst.get('node_id', '') == 'cloud':
            base_latency = src_lat + dst_lat + 10
            distance = np.linalg.norm(np.array(src_pos) - np.array(dst_pos)) * 0.2
            return base_latency + distance * 0.5
        
        # 普通节点延迟
        distance = np.linalg.norm(np.array(src_pos) - np.array(dst_pos)) * 111
        return src_lat + dst_lat + distance * 0.1

    def step(self, action):
        """执行调度动作"""
        req = self._generate_request()
        self.request_history.append(req)
        self.metrics['total_requests'] += 1
        
        target_node = self.nodes[action['target_type']][action['target_id']]
        
        # 计算延迟路径
        path_latency = self._calculate_path_latency(req['position'], target_node)
        total_latency = path_latency + 0.1 * req['compute_demand']  # 处理延迟
        
        # 执行资源分配
        if self._allocate_resource(target_node, req['compute_demand']):
            reward = self._calculate_reward(req, total_latency, is_cloud=(action['target_type']=='cloud'))
            self.metrics['succeed_requests'] += 1
            if action['target_type'] == 'cloud':
                self.metrics['cloud_requests'] += 1
        else:
            reward = -5  # 分配失败惩罚
            
        self.metrics['total_latency'] += total_latency
        next_state = self._get_state()
        done = self.current_time > 3600  # 模拟1小时
        
        return next_state, reward, done, {}

    def _calculate_path_latency(self, position, target_node):
        """计算请求位置到目标节点的路径延迟"""
        # 找到最近基站
        min_latency = float('inf')
        for bs in self.nodes['base_stations'].values():
            latency = self._calculate_latency(
                {'position': position, 'latency': 0},
                bs
            )
            if latency < min_latency:
                min_latency = latency
                nearest_bs = bs
        
        # 基站到目标路径
        if target_node['type'] == 'cloud':
            # 到云端的路径：基站 -> 机房 -> 云端
            room = self.nodes['rooms'][nearest_bs['room_id']]
            return (min_latency +
                    self._calculate_latency(nearest_bs, room) +
                    self._calculate_latency(room, self.cloud_node))
        else:
            # 本地路径：基站 -> 目标节点
            return min_latency + self._calculate_latency(nearest_bs, target_node)

    def _allocate_resource(self, target_node, demand):
        """执行资源分配"""
        if target_node['type'] == 'cloud':
            return True  # 云端资源无限
            
        if target_node['compute'] >= demand:
            target_node['compute'] -= demand
            return True
        return False

    def _calculate_reward(self, req, latency, is_cloud=False):
        """计算奖励值"""
        base = 20 if not is_cloud else 10  # 云奖励减半
        latency_penalty = max(0, latency - req['max_latency']) * (-0.5)
        return base + latency_penalty

    def _get_state(self):
        """获取当前状态图"""
        node_features = []
        edge_index = []
        
        # 添加云端节点
        node_features.append([
            self.cloud_node['position'][0],
            self.cloud_node['position'][1],
            9999,  # 特殊标记
            self.cloud_node['latency'],
            0  # 占位
        ])
        
        # 添加机房节点
        for room in self.nodes['rooms'].values():
            node_features.append([
                room['position'][0],
                room['position'][1],
                -1,  # 机房特殊标记
                room['bandwidth'],
                0
            ])
        
        # 添加基站节点
        for bs in self.nodes['base_stations'].values():
            node_features.append([
                bs['position'][0],
                bs['position'][1],
                bs['compute'] / (bs['max_compute'] + 1e-5),  # 归一化
                bs['bandwidth'],
                bs['latency']
            ])
        
        # 构建连接边（全连接逻辑）
        # 1. 所有机房互相连接
        room_ids = list(self.nodes['rooms'].keys())
        for i in range(len(room_ids)):
            for j in range(i+1, len(room_ids)):
                src = room_ids[i]
                dst = room_ids[j]
                edge_index.append([self._node_id_to_idx(src), self._node_id_to_idx(dst)])
                edge_index.append([self._node_id_to_idx(dst), self._node_id_to_idx(src)])
        
        # 2. 机房与云端连接
        cloud_idx = 0
        for room in self.nodes['rooms'].values():
            room_idx = self._node_id_to_idx(room['node_id'])
            edge_index.append([room_idx, cloud_idx])
            edge_index.append([cloud_idx, room_idx])
        
        # 3. 机房内部连接
        for room in self.nodes['rooms'].values():
            room_idx = self._node_id_to_idx(room['node_id'])
            for bs_id in room['connected_bs']:
                bs_idx = self._node_id_to_idx(bs_id)
                edge_index.append([room_idx, bs_idx])
                edge_index.append([bs_idx, room_idx])
                
    
        state = {
            'x': ...,
            'edge_index': ...,
            'valid_actions': self.get_valid_actions_mask()  # 新增
        }

        # return {
        #     'x': torch.tensor(node_features, dtype=torch.float32),
        #     'edge_index': torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        # }
        return state
    
    def get_valid_actions_mask(self):
        """获取合法动作的布尔掩码"""
        mask = []
        # 基站部分
        for bs in self.nodes['base_stations'].values():
            mask.append(bs['type'] == 'main' and bs['compute'] > 0)
        # 云端
        mask.append(True)
        return torch.tensor(mask, dtype=torch.bool)

    def get_valid_actions(self):
        """获取当前可用的合法动作集合"""
        valid_actions = []
        
        # 添加所有可用主基站
        for bs_id, bs in self.nodes['base_stations'].items():
            if bs['type'] == 'main' and bs['compute'] > 0:  # 只包含有算力的主基站
                valid_actions.append({
                    'target_type': 'base_stations',
                    'target_id': bs_id
                })
        
        # 添加云端选项
        valid_actions.append({
            'target_type': 'cloud',
            'target_id': 'cloud'
        })
        
        return valid_actions

    def sample_action(self):
        """智能动作采样（考虑资源可用性）"""
        available_actions = []
        
        # 有算力的主基站
        for bs_id in self.main_base_stations:
            bs = self.nodes['base_stations'][bs_id]
            if bs['compute'] > 5:  # 保留最小算力缓冲
                available_actions.append({
                    'target_type': 'base_stations',
                    'target_id': bs_id
                })
        
        # 总是包含云端选项
        available_actions.append({
            'target_type': 'cloud',
            'target_id': 'cloud'
        })
        
        return random.choice(available_actions) if available_actions else {
            'target_type': 'cloud',
            'target_id': 'cloud'
        }

    def _node_id_to_idx(self, node_id):
        """转换节点ID到特征矩阵索引"""
        if node_id == 'cloud':
            return 0
        if node_id in self.nodes['rooms']:
            return 1 + list(self.nodes['rooms'].keys()).index(node_id)
        return 1 + len(self.nodes['rooms']) + list(self.nodes['base_stations'].keys()).index(node_id)

    def reset(self):
        """重置环境"""
        self._reset_dynamic_state()
        self.current_time = 0
        return self._get_state()

    def setup_visualization(self):
        """初始化可视化画布"""
        plt.ion()  # 开启交互模式
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 10))
        
        # 主视图设置
        self.ax1.set_title("Network Topology")
        self.ax1.set_xlim(0, 95)
        self.ax1.set_ylim(0, 95)
        
        # 资源状态视图设置
        self.ax2.set_title("Resource Utilization")
        self.ax2.set_xlim(0, 1)
        self.ax2.set_ylim(0, len(self.nodes['base_stations'])+2)
        self.ax2.axis('off')
        
        # 初始化绘图元素
        self._init_visual_elements()
        
    def _init_visual_elements(self):
        """创建所有可视化元素"""
        # ===== 主视图元素 =====
        self.bs_artists = {}
        self.room_artists = {}
        
        # 创建专门用于图例的虚拟artist
        legend_elements = [
            Line2D([0], [0], marker='*', color='w', label='Cloud Node',
                markerfacecolor='gold', markersize=15),
            Line2D([0], [0], marker='*', color='w', label='Room',
                markerfacecolor='red', markersize=10),
            Line2D([0], [0], marker='s', color='w', label='Main BS',
                markerfacecolor='blue', markersize=8),
            Line2D([0], [0], marker='s', color='w', label='Slave BS',
                markerfacecolor='green', markersize=8)
        ]
        
        # 添加图例（只显示一次）
        self.ax1.legend(handles=legend_elements, 
                    loc='upper right',
                    bbox_to_anchor=(1.25, 1),
                    fontsize=8)
        
        # 绘制云端
        self.cloud_artist = self.ax1.scatter(
            [self.cloud_node['position'][0]], [self.cloud_node['position'][1]],
            c='gold', s=250, marker='*'  # 移除了label参数
        )
        
        # 绘制机房
        for room in self.nodes['rooms'].values():
            sc = self.ax1.scatter(
                room['position'][0], room['position'][1],
                c='red', s=100, marker='*'  # 移除了label参数
            )
            self.room_artists[room['node_id']] = sc
        
        # 绘制基站
        for bs in self.nodes['base_stations'].values():
            if bs['type'] == 'main':
                color, marker = 'blue', 's'
            else:
                color, marker = 'green', 's'
                
            sc = self.ax1.scatter(
                bs['position'][0], bs['position'][1],
                c=color, s=30, marker=marker  # 移除了label参数
            )
            self.bs_artists[bs['node_id']] = sc
        
        # 绘制连接线
        self.line_artists = []
        # 机房-云端连接
        # for room in self.nodes['rooms'].values():
        #     line, = self.ax1.plot(
        #         [room['position'][0], self.cloud_node['position'][0]],
        #         [room['position'][1], self.cloud_node['position'][1]],
        #         'y--', alpha=0.3
        #     )
        #     self.line_artists.append(line)

        # 绘制机房-主基站连接线（修正访问方式）
        for bs in self.nodes['base_stations'].values():
            # 通过基站节点中的room_id获取机房
            room = self.nodes['rooms'][bs['room_id']]
            line, = self.ax1.plot(
                [room['position'][0], bs['position'][0]],
                [room['position'][1], bs['position'][1]],
                'g-', alpha=0.2
            )
            self.line_artists.append(line)
        
        # ===== 资源视图 =====
        labels = ['Cloud'] + [bs['node_id'] for bs in self.nodes['base_stations'].values() if bs['type']=='main']
        n_bars = len(labels)
        
        # 设置 y 轴范围（关键！）
        self.ax2.set_ylim(-0.5, n_bars - 0.5)  # 上下留出空间
        self.ax2.set_xlim(-0.5, 1.2)  # 左侧留空间给标签，右侧留空间给数值
        
        # 初始化柱状图和文本
        self.util_bars = []
        self.util_texts = []
        for i, label in enumerate(labels):
            bar = self.ax2.barh(i, 0, height=0.6)  # y=i
            self.ax2.text(-0.1, i, label, ha='right', va='center', fontsize=10)  # 左侧标签
            util_text = self.ax2.text(0, i, "", ha='left', va='center', fontsize=9)  # 利用率文本
            self.util_bars.append(bar)
            self.util_texts.append(util_text)
        
        # 将统计文本放在柱状图上方
        self.stats_text = self.ax2.text(
            0.5, len(labels) + 0.5, 
            "Total Requests: 0\nSuccess Rate: 0%", 
            ha='center'
        )

        
    def update_visualization(self):
        """动态更新可视化"""
        # ===== 更新主视图 =====
        # 自动计算范围（包含所有元素）
        all_x = [
            r['position'][0] for r in self.request_history
        ] + [
            n['position'][0] for n in self.nodes['rooms'].values()
        ] + [
            n['position'][0] for n in self.nodes['base_stations'].values()
        ] + [
            self.cloud_node['position'][0]  # 包含云端节点
        ]
        
        all_y = [
            r['position'][1] for r in self.request_history
        ] + [
            n['position'][1] for n in self.nodes['rooms'].values()
        ] + [
            n['position'][1] for n in self.nodes['base_stations'].values()
        ] + [
            self.cloud_node['position'][1]  # 包含云端节点
        ]

        # 设置动态范围（扩展边界）
        padding = 5  # ← 调整边距大小
        x_min = min(all_x) - padding if all_x else 0
        x_max = max(all_x) + padding if all_x else 60
        y_min = min(all_y) - padding if all_y else 0
        y_max = max(all_y) + padding if all_y else 60
        
        self.ax1.set_xlim(x_min, x_max)
        self.ax1.set_ylim(y_min, y_max)

        # 更新基站状态
        for bs_id, artists in self.bs_artists.items():
            bs = self.nodes['base_stations'][bs_id]
            # 更新颜色（根据利用率）
            # util = bs['compute'] / bs['max_compute'] if bs['type']=='main' else 0
            # artists.set_color(plt.cm.RdYlGn(util))
            # artists[0].set_color(plt.cm.RdYlGn(util))
            # 更新文本
            # artists[1].set_text(
            #     f"{bs_id}\n{bs['compute']:.0f}/{bs['max_compute']:.0f}"
            # )
        
        # 更新机房颜色（根据带宽使用）
        for room_id, artists in self.room_artists.items():
            room = self.nodes['rooms'][room_id]
            used_bw = sum(
                bs['bandwidth'] for bs in self.nodes['base_stations'].values() 
                if bs['room_id'] == room_id
            )
            util = used_bw / room['bandwidth']
            artists.set_color(plt.cm.RdYlBu(util))
            # artists[0].set_color(plt.cm.RdYlBu(util))
        
        # === 请求分布可视化改进 ===
        # 清除旧请求
        for artist in self.request_artists:
            artist.remove()
        self.request_artists = []
        
        # 绘制新请求
        if self.request_history:
            # 按时间颜色渐变
            colors = plt.cm.Reds(np.linspace(0.5, 1, len(self.request_history)))
            
            # 分组绘制以提高性能
            scatter = self.ax1.scatter(
                [r['position'][0] for r in self.request_history],
                [r['position'][1] for r in self.request_history],
                c=colors,
                s=np.linspace(40, 20, len(self.request_history)),  # 新请求更大
                alpha=0.7,
                edgecolors='k',
                marker='o',
                zorder=10  # 确保在最上层
            )
            self.request_artists.append(scatter)
            
            # # 动态调整视图
            # all_x = [r['position'][0] for r in self.request_history] + \
            #        [n['position'][0] for n in self.nodes['rooms'].values()] + \
            #        [n['position'][0] for n in self.nodes['base_stations'].values()]
            # all_y = [r['position'][1] for r in self.request_history] + \
            #        [n['position'][1] for n in self.nodes['rooms'].values()] + \
            #        [n['position'][1] for n in self.nodes['base_stations'].values()]
            
            # self.ax1.set_xlim(min(all_x)-2, max(all_x)+2)
            # self.ax1.set_ylim(min(all_y)-2, max(all_y)+2)
        
        # ===== 更新资源视图 =====
        # 更新 Cloud 柱状图（y=0）
        cloud_util = self.metrics['cloud_requests'] / (self.metrics['total_requests'] + 1e-5)
        self.util_bars[0][0].set_width(cloud_util)
        self.util_texts[0].set_text(f"{cloud_util:.1%}")
        self.util_texts[0].set_position((cloud_util + 0.02, 0))
        
        # 更新基站柱状图（y=1, 2, ...）
        main_bs = [bs for bs in self.nodes['base_stations'].values() if bs['type']=='main']
        for i, bs in enumerate(main_bs, 1):  # i 从 1 开始
            util = (bs['max_compute'] - bs['compute']) / bs['max_compute']
            self.util_bars[i][0].set_width(util)
            self.util_texts[i].set_text(f"{util:.1%}")
            self.util_texts[i].set_position((util + 0.02, i))
        
        # 更新统计文本
        success_rate = self.metrics['succeed_requests'] / (self.metrics['total_requests'] + 1e-5)
        avg_latency = self.metrics['total_latency'] / (self.metrics['succeed_requests'] + 1e-5)
        self.stats_text.set_text(
            f"Total Requests: {self.metrics['total_requests']}\n"
            f"Success Rate: {success_rate:.1%}\n"
            f"Avg Latency: {avg_latency:.1f}ms"
        )
        
        # 立即重绘
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()



    
    