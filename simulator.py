import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D 
from collections import deque
from torch_geometric.data import Data
import random
import heapq
import math

class ComputingNetworkSimulator:
    def __init__(self, bs_csv_path, room_csv_path):
        # 加载原始数据
        self.bs_df = pd.read_csv(bs_csv_path)
        self.bs_df['room_id'] = self.bs_df['home_compute_node_id'].astype(str)
        self.room_df = pd.read_csv(room_csv_path)
        self.room_df['room_id'] = self.room_df['id'].astype(str)
        
        self.request_artists = []
        
        # 添加云端节点
        self.cloud_node = {
            'node_id': 'cloud',
            'type': 'cloud',
            'compute': float('inf'),
            'position': (110, 110),
            'latency': 5  # 基础延迟，统一单位为毫秒
        }
        
        # 构建节点字典
        self.nodes = {
            'rooms': {},
            'base_stations': {},
            'cloud': {'cloud': self.cloud_node} 
        }
        
        # 创建机房节点
        for _, row in self.room_df.iterrows():
            room_id = str(row['id'])
            has_extra = row['is_active']
            # 单个算力板的算力为50个单位
            compute_power = 50 * row['allocated_boards'] if has_extra == 1 else 0
            
            self.nodes['rooms'][room_id] = {
                'node_id': room_id,
                'type': 'room',
                'position': (row['x_coord'], row['y_coord']),
                'compute': compute_power,
                'max_compute': compute_power,
                'latency': 0
            }
        
        # 创建基站节点
        for _, row in self.bs_df.iterrows():
            bs_id = str(row['id'])
            room_id = str(row['home_compute_node_id'])
            
            self.nodes['base_stations'][bs_id] = {
                'node_id': bs_id,
                'type': 'bs',
                'position': (row['x_coord'], row['y_coord']),
                'latency': 0,
                'room_id': room_id
            }
            
        # 为每个机房添加连接的基站列表
        for room_id, room in self.nodes['rooms'].items():
            room['connected_bs'] = [
                bs_id for bs_id, bs in self.nodes['base_stations'].items() 
                if bs['room_id'] == room_id
            ]
        
        # 初始化动态状态
        self._reset_dynamic_state()
        
        # 请求生成参数
        self.request_rate = 2  # 每秒请求数
        self.current_time = 0
        self.request_counter = 0
        self.pending_events = []  # 事件队列 (时间, 事件类型, 数据)
        self.current_request = None  # 当前正在处理的请求
        
        # 添加动作空间大小
        self.action_space_size = len(self.nodes['rooms']) + 1  # 云端 + 所有机房

    def _reset_dynamic_state(self):
        """重置动态状态"""
        self.request_history = deque(maxlen=1000)
        self.metrics = {
            'total_requests': 0,
            'succeed_requests': 0,
            'cloud_requests': 0,
            'total_latency': 0,
            'total_processing': 0,
            'total_reward': 0
        }
        
        # 初始化算力
        for room in self.nodes['rooms'].values():
            room['compute'] = room['max_compute']
        
        # 清空事件队列
        self.pending_events = []
        self.current_request = None

    def _generate_request(self):
        """生成新的计算请求 - 增强空间依赖性和机房热点"""
        self.request_counter += 1
        inter_arrival = np.random.exponential(1/self.request_rate)
        self.current_time += inter_arrival
        
        # 根据机房位置创建热点区域
        room_positions = [room['position'] for room in self.nodes['rooms'].values()]
        
        # 70%的请求在机房附近生成（热点区域）
        if random.random() < 0.7 and room_positions:
            # 随机选择一个机房作为热点中心
            center_room = random.choice(room_positions)
            
            # 在机房周围生成请求（正态分布）
            position = (
                np.clip(np.random.normal(center_room[0], 10), 0, 100),
                np.clip(np.random.normal(center_room[1], 10), 0, 100)
            )
        else:
            # 30%的请求在随机区域生成
            position = (
                np.random.uniform(0, 100),
                np.random.uniform(0, 100)
            )
        
        # 添加位置依赖性：新请求位置靠近前一个请求的概率更高
        if self.request_history and random.random() < 0.4:
            last_position = self.request_history[-1]['position']
            position = (
                np.clip(last_position[0] + np.random.normal(0, 5), 0, 100),
                np.clip(last_position[1] + np.random.normal(0, 5), 0, 100)
            )
        
        # 根据请求类型分配不同的特性
        request_type = np.random.choice(
            ['safety-critical', 'infotainment', 'adas'], 
            p=[0.2, 0.5, 0.3]  # 出现概率分布
        )
        
        # 不同请求类型的特性参数
        if request_type == 'safety-critical':
            # 安全攸关请求：高实时性要求，中等算力需求，中等处理时间
            compute_demand = np.clip(np.random.normal(20, 3), 5, 40)  # 平均20，标准差3
            max_latency = np.random.choice([15, 20, 25])  # 严格的延迟要求
            base_process = 2  # 基础处理时间
            process_time = np.clip(np.random.normal(base_process, 0.05), 0.5, 4)  # 2±0.05秒
        elif request_type == 'infotainment':
            # 信息娱乐请求：较低的实时性要求，中等算力需求，较长处理时间
            compute_demand = np.clip(np.random.normal(15, 3), 5, 30)  # 平均15，标准差3
            max_latency = np.random.choice([30, 35, 40])  # 较宽松的延迟要求
            base_process = 20  # 基础处理时间
            process_time = np.clip(np.random.normal(base_process, 3), 5, 50)  # 20±3秒
        else:  # adas (Advanced Driver Assistance Systems)
            # 高级驾驶辅助：中等实时性要求，高算力需求，中等处理时间
            compute_demand = np.clip(np.random.normal(30, 2), 20, 40)  # 平均30，标准差2
            max_latency = np.random.choice([20, 25, 30])  # 中等延迟要求
            base_process = 10  # 基础处理时间
            process_time = np.clip(np.random.normal(base_process, 1), 2, 30)  # 10±1秒
        
        # 机房附近的请求更可能是高计算需求类型
        if room_positions:
            min_dist = min([self._calculate_distance(position, room) for room in room_positions])
            if min_dist < 15:  # 在机房附近
                # 增加高计算需求请求的概率
                if random.random() < 0.7:
                    compute_demand = np.clip(compute_demand * 1.5, 5, 20)
        
        req = {
            'req_id': f"REQ_{self.request_counter}_{request_type[0]}",  # 添加类型标识
            'timestamp': self.current_time,
            'position': position,
            'compute_demand': compute_demand,
            'max_latency': max_latency,
            'process_time': process_time,
            'allocations': {},  # 记录算力分配情况
            'home_room': None,  # 记录所属本地机房
            'target_room': None,  # 记录最终处理机房
            'type': request_type,  # 记录请求类型
            'completed': False  # 是否已完成处理
        }
        self.current_request = req  # 保存当前请求
        return req

    def _calculate_distance(self, pos1, pos2):
        """计算两点之间的欧氏距离"""
        return math.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)

    def _calculate_latency(self, src, dst, type):
        """计算两个位置之间的延迟"""
        distance = self._calculate_distance(
            src if isinstance(src, tuple) else src['position'],
            dst if isinstance(dst, tuple) else dst['position']
        )
        if type == 'req2bs':
            return distance * 0.2  # 每单位距离对应0.2ms延迟
        if type == 'bs2room':
            return distance * 0.05  # 每单位距离对应0.05ms延迟

    def _find_nearest_bs(self, position):
        """找到距离请求位置最近的基站"""
        min_latency = float('inf')
        nearest_bs = None
        
        for bs in self.nodes['base_stations'].values():
            latency = self._calculate_latency(position, bs['position'], 'req2bs')
            if latency < min_latency:
                min_latency = latency
                nearest_bs = bs
        
        return nearest_bs, min_latency

    def _allocate_resources(self, room_id, demand):
        """尝试从指定机房分配资源"""
        room = self.nodes['rooms'][room_id]
        if room['compute'] >= demand:
            room['compute'] -= demand
            return demand
        return 0

    def step(self, action):
        """执行调度动作"""
        # 处理到期事件（释放算力）
        self._process_pending_events()
        
        # 生成新请求
        req = self._generate_request()
        self.request_history.append(req)
        self.metrics['total_requests'] += 1
        
        # 找到最近的基站及其所属的本地机房
        nearest_bs, bs_latency = self._find_nearest_bs(req['position'])
        home_room_id = nearest_bs['room_id']
        home_room = self.nodes['rooms'][home_room_id]
        req['home_room'] = home_room_id
        
        # 计算基础延迟（请求位置到基站）
        base_latency = bs_latency
        
        # 解析动作
        if action == 'cloud':
            # 使用云端算力
            req['target_room'] = 'cloud'
            allocations = {'cloud': req['compute_demand']}
            is_cloud = True
            room_latency = 0
            self.metrics['cloud_requests'] += 1
        else:
            # 尝试从目标机房分配算力
            target_room_id = action
            allocated = self._allocate_resources(target_room_id, req['compute_demand'])
            
            if allocated > 0:
                # 成功在目标机房分配
                req['target_room'] = target_room_id
                allocations = {target_room_id: allocated}
                is_cloud = False
                
                # 计算机房延迟：基站到本地机房 + 本地机房到目标机房
                room_to_home = self._calculate_latency(
                    nearest_bs['position'], home_room['position'], 'bs2room')
                
                if target_room_id == home_room_id:
                    room_latency = room_to_home
                else:
                    target_room = self.nodes['rooms'][target_room_id]
                    home_to_target = self._calculate_latency(
                        home_room['position'], target_room['position'], 'bs2room')
                    room_latency = (room_to_home + home_to_target)
            else:
                # 分配失败，转用云端
                req['target_room'] = 'cloud'
                allocations = {'cloud': req['compute_demand']}
                is_cloud = True
                room_latency = 0
                self.metrics['cloud_requests'] += 1
        
        # 计算总延迟（请求到基站 + 机房处理延迟）
        total_latency = base_latency + room_latency + req['compute_demand'] * 0.1
        
        # 检查请求是否成功（是否满足最大延迟要求）
        is_success = total_latency <= req['max_latency']
        
        # 记录请求状态
        req['allocations'] = allocations
        req['total_latency'] = total_latency
        req['is_success'] = is_success
        
        # 记录处理完成事件
        completion_time = self.current_time + req['process_time']
        heapq.heappush(self.pending_events, 
                    (completion_time, 'release', req))
        
        # 计算奖励
        reward = self._calculate_reward(req, total_latency, is_cloud)
        
        # 更新指标
        self.metrics['succeed_requests'] += 1
        self.metrics['total_latency'] += total_latency
        self.metrics['total_processing'] += req['compute_demand']
        self.metrics['total_reward'] += reward
        
        # 添加当前请求的详细信息到指标中
        current_metrics = {
            'last_request': req.copy(),
            'last_latency': total_latency,
            'last_success': is_success,
            'used_cloud': is_cloud,
            'last_reward': reward
        }
        
        # 合并到总指标
        metrics = {**self.metrics, **current_metrics}
        
        next_state = self._get_state()
        done = self.current_time > 3600  # 模拟1小时
        
        return next_state, reward, done, metrics

    def _process_pending_events(self):
        """处理到期事件（释放算力）"""
        while self.pending_events and self.pending_events[0][0] <= self.current_time:
            event_time, event_type, event_data = heapq.heappop(self.pending_events)
            
            if event_type == 'release':
                self._release_resources(event_data)

    def _release_resources(self, req):
        """释放请求占用的算力资源"""
        for room_id, amount in req['allocations'].items():
            if room_id != 'cloud' and room_id in self.nodes['rooms']:
                # 恢复算力到原始机房
                self.nodes['rooms'][room_id]['compute'] += amount
                # 确保不超过最大算力
                if self.nodes['rooms'][room_id]['compute'] > self.nodes['rooms'][room_id]['max_compute']:
                    self.nodes['rooms'][room_id]['compute'] = self.nodes['rooms'][room_id]['max_compute']

    def _calculate_reward(self, req, latency, is_cloud):
        """计算奖励值"""
        # 基础奖励
        if is_cloud:
            base = 10  # 云端基础奖励
        elif req['target_room'] == req['home_room']:
            base = 20  # 本地机房奖励
        else:
            base = 20  # 其他机房奖励
        
        # 延迟惩罚：超时线性惩罚
        latency_penalty = max(0, latency - req['max_latency']) + latency * 0.2
        
        # 资源效率奖励：鼓励高利用率
        # efficiency_bonus = min(2.0, req['compute_demand'] / 5)
        
        # 云端使用惩罚
        cloud_cost = -3 if is_cloud else 0
        
        # 增加长期资源利用率奖励
        utilization_bonus = 0
        for room in self.nodes['rooms'].values():
            # 计算长期利用率（最近10个请求的平均）
            if hasattr(room, 'recent_utilization'):
                avg_util = sum(room.recent_utilization) / len(room.recent_utilization)
                utilization_bonus += min(2.0, avg_util)
    
        # 增加动作一致性奖励（鼓励相似请求使用相同机房）
        consistency_bonus = 0
        # if self.request_history:
        #     last_action = self.request_history[-1].get('action', '')
        #     if action == last_action:
        #         consistency_bonus = 1.0
        
        return base - latency_penalty + cloud_cost + utilization_bonus + consistency_bonus

    def _generate_random_position(self):
        """生成随机位置"""
        return (np.random.uniform(0, 80), np.random.uniform(0, 80))
    
    def get_valid_actions_mask(self):
        """获取合法动作的布尔掩码"""
        # 如果没有当前请求，返回全False
        if not hasattr(self, 'current_request') or not self.current_request:
            return torch.zeros(self.action_space_size, dtype=torch.bool)
        
        valid_actions = []
        demand = self.current_request['compute_demand']
        
        # 云端总是可用
        valid_actions.append(True)
        
        # 检查所有机房是否有足够算力
        for room in self.nodes['rooms'].values():
            if room['compute'] >= demand:
                valid_actions.append(True)
            else:
                valid_actions.append(False)
        
        return torch.tensor(valid_actions, dtype=torch.bool)
    
    def get_valid_actions(self):
        """获取当前可用的合法动作列表"""
        # 如果没有当前请求，返回空列表
        if not hasattr(self, 'current_request') or not self.current_request:
            return []
        
        valid_actions = ['cloud']
        demand = self.current_request['compute_demand']
        
        # 检查所有机房是否有足够算力
        for room_id, room in self.nodes['rooms'].items():
            if room['compute'] >= demand:
                valid_actions.append(room_id)
        
        return valid_actions

    def _get_state(self):
        """获取当前状态图"""
        node_features = []
        edge_index = []
        
        # 添加云端节点
        node_features.append([
            self.cloud_node['position'][0],
            self.cloud_node['position'][1],
            2,  # 节点类型: 2=云端
            self.cloud_node['latency'],
            0   # 第5个特征（占位）
        ])
        
        # 添加机房节点
        for room_id, room in sorted(self.nodes['rooms'].items()):
            compute_ratio = room['compute'] / room['max_compute'] if room['max_compute'] > 0 else 0
            node_features.append([
                room['position'][0],
                room['position'][1],
                1,  # 节点类型: 1=机房
                compute_ratio,
                0   # 第5个特征（占位）
            ])
        
        # 添加基站节点
        for bs_id, bs in sorted(self.nodes['base_stations'].items()):
            node_features.append([
                bs['position'][0],
                bs['position'][1],
                0,  # 节点类型: 0=基站
                bs['latency'],
                0   # 第5个特征（占位）
            ])
        
        # 构建节点索引映射
        node_index_map = {}
        idx = 0
        
        # 云端节点索引
        node_index_map['cloud'] = idx
        idx += 1
        
        # 机房节点索引
        for room_id in sorted(self.nodes['rooms'].keys()):
            node_index_map[room_id] = idx
            idx += 1
            
        # 基站节点索引
        for bs_id in sorted(self.nodes['base_stations'].keys()):
            node_index_map[bs_id] = idx
            idx += 1
        
        # 构建连接边
        # 1. 基站连接到所属机房
        for bs_id, bs in self.nodes['base_stations'].items():
            room_id = bs['room_id']
            edge_index.append([node_index_map[bs_id], node_index_map[room_id]])
            edge_index.append([node_index_map[room_id], node_index_map[bs_id]])
        
        # 2. 机房之间全连接
        room_ids = sorted(self.nodes['rooms'].keys())
        for i in range(len(room_ids)):
            for j in range(i+1, len(room_ids)):
                src = room_ids[i]
                dst = room_ids[j]
                edge_index.append([node_index_map[src], node_index_map[dst]])
                edge_index.append([node_index_map[dst], node_index_map[src]])
        
        # 3. 所有机房连接到云端
        for room_id in room_ids:
            edge_index.append([node_index_map[room_id], node_index_map['cloud']])
            edge_index.append([node_index_map['cloud'], node_index_map[room_id]])
        
        # 获取合法动作掩码
        valid_actions_mask = self.get_valid_actions_mask()
        
        return {
            'x': torch.tensor(node_features, dtype=torch.float32),
            'edge_index': torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
            'valid_actions': valid_actions_mask
        }

    def reset(self):
        """重置环境"""
        self._reset_dynamic_state()
        self.current_time = 0
        # 生成第一个请求
        self._generate_request()
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
        self.ax2.set_ylim(0, len(self.nodes['rooms'])+2)
        self.ax2.axis('off')
        
        # 初始化绘图元素
        self._init_visual_elements()
        
    def _init_visual_elements(self):
        """创建所有可视化元素"""
        # ===== 主视图元素 =====
        self.bs_artists = {}
        self.room_artists = {}
        
        # 图例元素
        legend_elements = [
            Line2D([0], [0], marker='*', color='w', label='Cloud Node',
                markerfacecolor='gold', markersize=15),
            Line2D([0], [0], marker='*', color='w', label='Room',
                markerfacecolor='red', markersize=10),
            Line2D([0], [0], marker='s', color='w', label='Base Station',
                markerfacecolor='blue', markersize=8)
        ]
        
        # 添加图例
        self.ax1.legend(handles=legend_elements, 
                    loc='upper right',
                    bbox_to_anchor=(1.25, 1),
                    fontsize=8)
        
        # 绘制云端
        self.cloud_artist = self.ax1.scatter(
            [self.cloud_node['position'][0]], [self.cloud_node['position'][1]],
            c='gold', s=250, marker='*'
        )

        # 绘制基站
        for bs in self.nodes['base_stations'].values():
            sc = self.ax1.scatter(
                bs['position'][0], bs['position'][1],
                c='blue', s=20, marker='s'
            )
            self.bs_artists[bs['node_id']] = sc
        
        # 绘制机房
        for room in self.nodes['rooms'].values():
            sc = self.ax1.scatter(
                room['position'][0], room['position'][1],
                c='red', s=150, marker='*'
            )
            self.room_artists[room['node_id']] = sc
            # 添加机房ID标签
            self.ax1.text(
                room['position'][0] + 1, room['position'][1] + 1,
                room['node_id'], fontsize=8)
        
        
        # 绘制连接线
        self.line_artists = []
        
        # 绘制基站-机房连接线
        for bs in self.nodes['base_stations'].values():
            room_id = bs['room_id']
            if room_id in self.nodes['rooms']:
                room = self.nodes['rooms'][room_id]
                line, = self.ax1.plot(
                    [bs['position'][0], room['position'][0]],
                    [bs['position'][1], room['position'][1]],
                    'g-', alpha=0.2
                )
                self.line_artists.append(line)
        
        # ===== 资源视图 =====
        room_ids = sorted(self.nodes['rooms'].keys())
        n_bars = len(room_ids)
        
        # 设置 y 轴范围
        self.ax2.set_ylim(-0.5, n_bars - 0.5)
        self.ax2.set_xlim(-0.5, 1.2)
        
        # 初始化柱状图和文本
        self.util_bars = []
        self.util_texts = []
        for i, room_id in enumerate(room_ids):
            bar = self.ax2.barh(i, 0, height=0.6)
            self.ax2.text(-0.1, i, room_id, ha='right', va='center', fontsize=10)
            util_text = self.ax2.text(0, i, "", ha='left', va='center', fontsize=9)
            self.util_bars.append(bar)
            self.util_texts.append(util_text)
        
        # 统计文本
        self.stats_text = self.ax2.text(
            0.5, n_bars + 0.5, 
            "Total Requests: 0\nSuccess Rate: 0%", 
            ha='center'
        )

    def update_visualization(self):
        """动态更新可视化"""
        # ===== 更新主视图 =====
        # 计算动态范围
        all_x = [pos[0] for pos in self._all_positions()]
        all_y = [pos[1] for pos in self._all_positions()]
        
        padding = 5
        x_min = min(all_x) - padding if all_x else 0
        x_max = max(all_x) + padding if all_x else 60
        y_min = min(all_y) - padding if all_y else 0
        y_max = max(all_y) + padding if all_y else 60
        
        self.ax1.set_xlim(x_min, x_max)
        self.ax1.set_ylim(y_min, y_max)

        # 更新机房颜色（根据利用率）
        for room_id, artist in self.room_artists.items():
            room = self.nodes['rooms'][room_id]
            util = (room['max_compute'] - room['compute']) / room['max_compute'] if room['max_compute'] > 0 else 0
            color = plt.cm.RdYlBu(util)
            artist.set_color(color)
        
        # 更新请求位置
        for artist in self.request_artists:
            artist.remove()
        self.request_artists = []
        
        if self.request_history:
            # 按时间颜色渐变
            colors = plt.cm.Reds(np.linspace(0.5, 1, len(self.request_history)))
            
            # 分组绘制请求
            scatter = self.ax1.scatter(
                [r['position'][0] for r in self.request_history],
                [r['position'][1] for r in self.request_history],
                c=colors,
                s=np.linspace(40, 20, len(self.request_history)),
                alpha=0.7,
                edgecolors='k',
                marker='o',
                zorder=10
            )
            self.request_artists.append(scatter)
        
        # ===== 更新资源视图 =====
        room_ids = sorted(self.nodes['rooms'].keys())
        
        # 更新机房利用率
        for i, room_id in enumerate(room_ids):
            room = self.nodes['rooms'][room_id]
            util = (room['max_compute'] - room['compute']) / room['max_compute'] if room['max_compute'] > 0 else 0
            self.util_bars[i][0].set_width(util)
            self.util_texts[i].set_text(f"{util:.1%}")
            self.util_texts[i].set_position((util + 0.02, i))
        
        # 更新统计文本
        success_rate = self.metrics['succeed_requests'] / (self.metrics['total_requests'] + 1e-5)
        avg_latency = self.metrics['total_latency'] / (self.metrics['succeed_requests'] + 1e-5)
        self.stats_text.set_text(
            f"Total Requests: {self.metrics['total_requests']}\n"
            f"Success Rate: {success_rate:.1%}\n"
            f"Avg Latency: {avg_latency:.1f}ms\n"
            f"Cloud Requests: {self.metrics['cloud_requests']}\n"
            f"Total Processing: {self.metrics['total_processing']}"
        )
        
        # 立即重绘
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _all_positions(self):
        """获取所有实体的位置"""
        positions = [r['position'] for r in self.request_history]
        positions.extend([n['position'] for n in self.nodes['rooms'].values()])
        positions.extend([n['position'] for n in self.nodes['base_stations'].values()])
        positions.append(self.cloud_node['position'])
        return positions