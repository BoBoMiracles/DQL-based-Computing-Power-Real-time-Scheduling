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
from geopy.distance import geodesic
import os

class ComputingNetworkSimulator:
    def __init__(self, bs_csv_path, room_csv_path, rate, simulation_time, reward_weights=(0.7, 1.0, 0.2, 0.1)):
        # 加载原始数据
        self.bs_df = pd.read_csv(bs_csv_path)
        self.room_df = pd.read_csv(room_csv_path)
        
        # 过滤经度范围 (113.5 < x_coord < 114.5)
        self.bs_df = self.bs_df[(self.bs_df['x_coord'] > 113.5) & (self.bs_df['x_coord'] < 114.5)]
        self.room_df = self.room_df[(self.room_df['x_coord'] > 113.5) & (self.room_df['x_coord'] < 114.5)]
        
        # 计算经纬度范围 - 使用x_coord和y_coord
        self.min_lat = min(self.bs_df['y_coord'].min(), self.room_df['y_coord'].min())
        self.max_lat = max(self.bs_df['y_coord'].max(), self.room_df['y_coord'].max())
        self.min_lon = min(self.bs_df['x_coord'].min(), self.room_df['x_coord'].min())
        self.max_lon = max(self.bs_df['x_coord'].max(), self.room_df['x_coord'].max())
        
        # 添加云端节点
        self.cloud_node = {
            'node_id': 'cloud',
            'type': 'cloud',
            'compute': float('inf'),
            'position': (self.max_lat + 1, self.max_lon + 1),  # 放在区域外
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
            room_id = f"{row['id']}_{row['y_coord']}_{row['x_coord']}"
            compute_power = 50 * row['allocated_boards']
            
            self.nodes['rooms'][room_id] = {
                'node_id': room_id,
                'type': 'room',
                'position': (row['y_coord'], row['x_coord']),
                'compute': compute_power,
                'max_compute': compute_power,
                'latency': 0
            }
        
        # 创建基站节点
        for _, row in self.bs_df.iterrows():
            bs_id = str(row['id'])
            
            # 查找归属机房
            home_room_id = None
            for room_id, room in self.nodes['rooms'].items():
                # 提取机房ID和位置
                room_id_num = room_id.split('_')[0]  # 提取ID部分
                room_x = room['position'][0]
                room_y = room['position'][1]
                
                # 检查是否匹配
                if (room_id_num == str(row['home_compute_node_id']) and 
                    abs(room_x - row['home_y_coord']) < 1e-5 and 
                    abs(room_y - row['home_x_coord']) < 1e-5):
                    home_room_id = room_id
                    break
            
            # 查找分配机房
            assigned_room_id = None
            for room_id, room in self.nodes['rooms'].items():
                # 提取机房ID和位置
                room_id_num = room_id.split('_')[0]  # 提取ID部分
                room_x = room['position'][0]
                room_y = room['position'][1]
                
                # 检查是否匹配
                if (room_id_num == str(row['assigned_compute_node_id']) and 
                    abs(room_x - row['assigned_y_coord']) < 1e-5 and 
                    abs(room_y - row['assigned_x_coord']) < 1e-5):
                    assigned_room_id = room_id
                    break
            
            self.nodes['base_stations'][bs_id] = {
                'node_id': bs_id,
                'type': 'bs',
                'position': (row['y_coord'], row['x_coord']),
                'latency': 0,
                'home_room': home_room_id,
                'assigned_room': assigned_room_id
            }
        
        # 为每个机房添加连接的基站列表
        for room_id, room in self.nodes['rooms'].items():
            room['connected_bs'] = [
                bs_id for bs_id, bs in self.nodes['base_stations'].items() 
                if bs['assigned_room'] == room_id or bs['home_room'] == room_id
            ]

        # 添加服务配置文件
        self.SERVICE_PROFILES = {
            "urgent_braking": (25, 8, 5, 3, 20, 0.1),  # 紧急制动
            "traffic_control": (20, 5, 15, 3, 20, 0.2),  # 交通信号控制
            "collision_avoidance": (40, 10, 20, 5, 40, 0.3),  # 避免碰撞
            "sensor_sharing": (60, 15, 50, 8, 30, 0.2),  # 传感器共享
            "hd_map_update": (50, 25, 100, 10, 50, 0.1),  # 高清地图更新
            "infotainment": (30, 20, 200, 15, 60, 0.1),  # 信息娱乐
        }
        
        # 服务类型到类别的映射
        self.service_categories = {
            "urgent_braking": "ultra_low_latency",
            "traffic_control": "ultra_low_latency",
            "collision_avoidance": "low_latency",
            "sensor_sharing": "low_latency",
            "hd_map_update": "high_latency_tolerance",
            "infotainment": "high_latency_tolerance"
        }
        
        # 处理时间参数 (基础值, 标准差)
        self.process_time_params = {
            "ultra_low_latency": (2, 0.05),  # 极低延迟类：处理时间短
            "low_latency": (10, 1),  # 低延迟类：处理时间中等
            "high_latency_tolerance": (20, 3)  # 高延迟容忍类：处理时间长
        }
        
        # 初始化动态状态
        self.total_time = simulation_time 
        self.request_rate = rate
        self.current_time = 0
        self.request_counter = 0
        self.pending_events = []  # 事件队列 (时间, 事件类型, 数据)
        self.current_request = None  # 当前正在处理的请求
        
        # 添加奖励权重参数
        self.reward_weights = reward_weights  # (基础权重, 延迟权重, 利用率权重, 一致性权重)
        
        # 添加动作空间大小
        self.action_space_size = len(self.nodes['rooms']) + 1  # 云端 + 所有机房

        # 添加利用率历史队列的最大长度
        self.utilization_history_length = 100
        self._reset_dynamic_state()

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

        # 初始化每个机房的利用率历史记录
        for room in self.nodes['rooms'].values():
            room['utilization_history'] = deque(maxlen=self.utilization_history_length)

    def _calculate_distance(self, pos1, pos2):
        """使用Haversine公式计算两点之间的真实距离（单位：公里）"""
        return geodesic(pos1, pos2).kilometers

    def _calculate_latency(self, distance_km, connection_type):
        """
        根据距离和连接类型计算延迟
        :param distance_km: 距离（公里）
        :param connection_type: 连接类型 ('req2bs', 'bs2room', 'room2room')
        :return: 延迟（毫秒）
        """
        # 请求到基站：每公里0.5毫秒
        if connection_type == 'req2bs':
            return distance_km * 0.5
        
        # 基站到机房/机房之间：每10公里100微秒（0.1毫秒）
        return distance_km * 0.1

    def _generate_request(self):
        """生成新的计算请求 - 使用SERVICE_PROFILES参数"""
        self.request_counter += 1
        inter_arrival = np.random.exponential(1/self.request_rate)
        self.current_time += inter_arrival
        
        # 根据SERVICE_PROFILES的概率选择服务类型
        service_types = list(self.SERVICE_PROFILES.keys())
        probs = [self.SERVICE_PROFILES[st][5] for st in service_types]  # 概率是元组的第6个元素
        service_type = random.choices(service_types, weights=probs, k=1)[0]
        
        # 获取服务类型的参数
        params = self.SERVICE_PROFILES[service_type]
        cpu_mean, cpu_std, mem_mean, mem_std, latency, prob = params
        
        # 生成算力需求（基于CPU均值）
        compute_demand = np.clip(np.random.normal(cpu_mean, cpu_std), 5, 100)
        
        # 设置最大延迟
        max_latency = latency
        
        # 根据服务类别生成处理时间
        category = self.service_categories[service_type]
        base_process, process_std = self.process_time_params[category]
        process_time = np.clip(np.random.normal(base_process, process_std), 0.5, 50)
        
        # 70%的请求在机房附近生成（热点区域）
        if random.random() < 0.7 and self.nodes['rooms']:
            # 选择机房作为热点中心 - 优先选择算力强的机房
            room_weights = [room['max_compute'] for room in self.nodes['rooms'].values()]
            total_weight = sum(room_weights)
            probs = [w/total_weight for w in room_weights]
            
            # 随机选择一个机房作为热点中心
            center_room = random.choices(
                list(self.nodes['rooms'].values()), 
                weights=probs
            )[0]
            
            # 在机房周围生成请求（正态分布）
            # 标准差为0.1度（约11公里）
            position = (
                np.clip(np.random.normal(center_room['position'][0], 0.1), self.min_lat, self.max_lat),
                np.clip(np.random.normal(center_room['position'][1], 0.1), self.min_lon, self.max_lon)
            )
            
            # 标记为热点请求
            is_hotspot = True
        else:
            # 30%的请求在随机区域生成
            position = (
                np.random.uniform(self.min_lat, self.max_lat),
                np.random.uniform(self.min_lon, self.max_lon)
            )
            is_hotspot = False
        
        # 添加位置依赖性：新请求位置靠近前一个请求的概率更高
        if self.request_history and random.random() < 0.4:
            last_position = self.request_history[-1]['position']
            position = (
                np.clip(np.random.normal(last_position[0], 0.05), self.min_lat, self.max_lat),
                np.clip(np.random.normal(last_position[1], 0.05), self.min_lon, self.max_lon)
            )
        
        # 热点区域的请求更可能是高计算需求类型
        if is_hotspot and random.random() < 0.7:
            compute_demand = np.clip(compute_demand * 1.5, 5, 100)
        
        # 添加时间依赖性：连续请求类型相同的概率更高
        if self.request_history and random.random() < 0.3:
            last_type = self.request_history[-1]['type']
            if last_type != service_type:
                # 50%的概率保持相同类型
                if random.random() < 0.5:
                    service_type = last_type
                    # 重新获取参数
                    params = self.SERVICE_PROFILES[service_type]
                    cpu_mean, cpu_std, mem_mean, mem_std, latency, prob = params
                    compute_demand = np.clip(np.random.normal(cpu_mean, cpu_std), 5, 100)
                    max_latency = latency
                    category = self.service_categories[service_type]
                    base_process, process_std = self.process_time_params[category]
                    process_time = np.clip(np.random.normal(base_process, process_std), 0.5, 50)
        
        req = {
            'req_id': f"REQ_{self.request_counter}_{service_type}",
            'timestamp': self.current_time,
            'position': position,
            'compute_demand': compute_demand,
            'max_latency': max_latency,
            'process_time': process_time,
            'allocations': {},
            'home_room': None,
            'target_room': None,
            'type': service_type,
            'completed': False,
            'is_hotspot': is_hotspot
        }
        self.current_request = req
        return req

    def _find_nearest_bs(self, position):
        """找到距离请求位置最近的基站"""
        min_latency = float('inf')
        nearest_bs = None
        
        for bs in self.nodes['base_stations'].values():
            distance = self._calculate_distance(position, bs['position'])
            latency = self._calculate_latency(distance, 'req2bs')
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
        
        # 找到最近的基站
        nearest_bs, bs_latency = self._find_nearest_bs(req['position'])
        base_latency = bs_latency
        
        # 确定归属机房（优先使用分配机房）
        home_room_id = None
        if nearest_bs['assigned_room']:
            home_room_id = nearest_bs['assigned_room']
        elif nearest_bs['home_room']:
            home_room_id = nearest_bs['home_room']
        
        home_room = self.nodes['rooms'].get(home_room_id, None)
        req['home_room'] = home_room_id
        
        # 解析动作
        if action == 'cloud':
            # 使用云端算力
            req['target_room'] = 'cloud'
            allocations = {'cloud': req['compute_demand']}
            is_cloud = True
            allocated = 0 
            
            # 计算完整云端路径延迟
            if home_room:
                # 计算基站到机房的延迟
                bs_to_room_dist = self._calculate_distance(
                    nearest_bs['position'], home_room['position'])
                bs_to_room_latency = self._calculate_latency(bs_to_room_dist, 'bs2room')
                
                # 机房到云端的固定延迟
                room_to_cloud = self.cloud_node['latency']
                room_latency = bs_to_room_latency + room_to_cloud
            else:
                room_latency = self.cloud_node['latency']  # 直接到云端
                
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
                
                # 计算机房延迟
                if home_room:
                    # 计算基站到归属机房的延迟
                    bs_to_home_dist = self._calculate_distance(
                        nearest_bs['position'], home_room['position'])
                    bs_to_home_latency = self._calculate_latency(bs_to_home_dist, 'bs2room')
                    
                    if target_room_id == home_room_id:
                        room_latency = bs_to_home_latency
                    else:
                        target_room = self.nodes['rooms'][target_room_id]
                        # 计算归属机房到目标机房的延迟
                        home_to_target_dist = self._calculate_distance(
                            home_room['position'], target_room['position'])
                        home_to_target_latency = self._calculate_latency(home_to_target_dist, 'room2room')
                        room_latency = bs_to_home_latency + home_to_target_latency
                else:
                    # 没有归属机房，直接到目标机房
                    target_room = self.nodes['rooms'][target_room_id]
                    bs_to_target_dist = self._calculate_distance(
                        nearest_bs['position'], target_room['position'])
                    room_latency = self._calculate_latency(bs_to_target_dist, 'bs2room')
            else:
                # 分配失败，转用云端
                req['target_room'] = 'cloud'
                allocations = {'cloud': req['compute_demand']}
                is_cloud = True
                
                if home_room:
                    bs_to_room_dist = self._calculate_distance(
                        nearest_bs['position'], home_room['position'])
                    bs_to_room_latency = self._calculate_latency(bs_to_room_dist, 'bs2room')
                    room_to_cloud = self.cloud_node['latency']
                    room_latency = bs_to_room_latency + room_to_cloud
                else:
                    room_latency = self.cloud_node['latency']
                    
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
        reward = self._calculate_reward(req, total_latency, is_cloud, action) 
        
        # 更新指标
        if is_success:
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
        done = self.current_time > self.total_time # 模拟1小时
        
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

    def _calculate_reward(self, req, latency, is_cloud, action):
        """计算奖励值（使用可配置权重）"""
        base_weight, latency_weight, util_weight, consistency_weight = self.reward_weights
        
        # 基础奖励
        if is_cloud:
            base = 2
        elif req['target_room'] == req['home_room']:
            base = 5
        else:
            base = 5
        
        # 延迟惩罚
        latency_penalty = max(0, latency - req['max_latency']) * 1 + latency * 0.5
        
        # 计算长期利用率奖励
        utilization_bonus = self._get_utilization_bonus()
        # utilization_bonus = 0
        
        # 机房选择的稳定性奖励
        # consistency_bonus = self._get_consistency_bonus(req, action)
        consistency_bonus = 0
        
        # 应用权重
        base_reward = base * base_weight
        latency_penalty = latency_penalty * latency_weight
        utilization_bonus = utilization_bonus * util_weight
        consistency_bonus = consistency_bonus * consistency_weight
        
        return base_reward - latency_penalty + utilization_bonus + consistency_bonus

    def _get_utilization_bonus(self):
        """计算所有机房的平均利用率奖励"""
        total_utilization = 0
        valid_rooms = 0
        
        for room_id, room in self.nodes['rooms'].items():
            if room['utilization_history']:
                # 计算该机房的平均利用率（指数加权平均）
                history = list(room['utilization_history'])
                weights = np.exp(np.linspace(0, 1, len(history)))  # 最近的值权重更高
                weights /= weights.sum()
                
                avg_util = np.dot(history, weights)
                total_utilization += min(1.0, avg_util)  # 上限100%利用率
                valid_rooms += 1
        
        # 计算全局平均利用率奖励
        # 0.5到2之间的线性映射：50%利用率→1.0，80%利用率→1.6
        if valid_rooms > 0:
            global_avg = total_utilization / valid_rooms
            # 基本奖励 + 高阶优化奖励（鼓励维持70-80%的理想利用率）
            base_bonus = min(global_avg, 0.8) * 2.0  # 线性部分
            optimal_bonus = max(0, min(global_avg - 0.7, 0.1)) * 3.0  # 70-80%额外奖励
            return base_bonus + optimal_bonus
        
        return 0

    def _get_consistency_bonus(self, req, action):
        """机房选择一致性奖励"""
        if not self.request_history:
            return 0
            
        # 检查连续使用的机房
        last_req = self.request_history[-1]
        
        # 1. 当请求类型相同时，使用相同机房
        if req['type'] == last_req.get('type', ''):
            if 'target_room' in last_req and action == last_req['target_room']:
                return 1.5
        
        # 2. 当请求位置相近时，使用相同机房
        distance = self._calculate_distance(req['position'], last_req['position'])
        if distance < 10:
            if 'target_room' in last_req and action == last_req['target_room']:
                return 1.0
        
        return 0

    def _generate_random_position(self):
        """生成随机位置"""
        return (np.random.uniform(0, 80), np.random.uniform(0, 80))
    
    def get_valid_actions_mask(self):
        """获取合法动作的布尔掩码"""
        if not hasattr(self, 'current_request') or not self.current_request:
            return torch.zeros(self.action_space_size, dtype=torch.bool)
        
        valid_actions = []
        demand = self.current_request['compute_demand']
        
        # 云端总是可用
        valid_actions.append(True)
        
        # 按排序后的机房ID检查算力
        room_ids = sorted(self.nodes['rooms'].keys())  # 关键修改：排序后遍历
        for room_id in room_ids:
            room = self.nodes['rooms'][room_id]
            if room['compute'] >= demand:
                valid_actions.append(True)
            else:
                valid_actions.append(False)
        
        return torch.tensor(valid_actions, dtype=torch.bool)

    def get_valid_actions(self):
        """获取当前可用的合法动作列表"""
        if not hasattr(self, 'current_request') or not self.current_request:
            return []
        
        valid_actions = ['cloud']
        demand = self.current_request['compute_demand']
        
        # 按排序后的机房ID检查算力
        room_ids = sorted(self.nodes['rooms'].keys())  # 关键修改：排序后遍历
        for room_id in room_ids:
            room = self.nodes['rooms'][room_id]
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
            room_id = bs['assigned_room'] or bs['home_room']
            if room_id and room_id in node_index_map:
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
            
            # 添加机房ID标签（只显示ID部分）
            room_id_parts = room['node_id'].split('_')
            display_id = room_id_parts[0] if len(room_id_parts) > 0 else room['node_id']
            self.ax1.text(
                room['position'][0] + 1, room['position'][1] + 1,
                display_id, fontsize=8)
        
        # 绘制连接线
        self.line_artists = []
        
        # 绘制基站-机房连接线
        for bs in self.nodes['base_stations'].values():
            room_id = bs['assigned_room'] or bs['home_room']
            if room_id and room_id in self.nodes['rooms']:
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
            # 只显示ID部分
            room_id_parts = room_id.split('_')
            display_id = room_id_parts[0] if len(room_id_parts) > 0 else room_id
            
            bar = self.ax2.barh(i, 0, height=0.6)
            self.ax2.text(-0.1, i, display_id, ha='right', va='center', fontsize=10)
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


    def visualize_topology(self, output_path="network_topology.png"):
        """可视化网络拓扑结构并保存为图片"""
        plt.figure(figsize=(15, 12))
        ax = plt.gca()
        
        # 设置边界
        padding = 0.1 * (self.max_lon - self.min_lon)
        ax.set_xlim(self.min_lon - padding, self.max_lon + padding)
        ax.set_ylim(self.min_lat - padding, self.max_lat + padding)
        
        # 添加标题和标签
        plt.title(f"Computing Network Topology (Rate={self.request_rate})", fontsize=16)
        plt.xlabel("Longitude (x_coord)", fontsize=12)
        plt.ylabel("Latitude (y_coord)", fontsize=12)
        
        # 绘制基站
        for bs_id, bs in self.nodes['base_stations'].items():
            plt.scatter(
                bs['position'][1], bs['position'][0],  # x_coord, y_coord
                s=50, color='blue', marker='^', alpha=0.7,
                edgecolor='black', linewidth=0.5
            )
            # 添加基站ID标签
            # plt.text(
            #     bs['position'][1] + 0.001, bs['position'][0] + 0.001,
            #     f"BS{bs_id}", fontsize=8, ha='left', va='bottom'
            # )
        
        # 绘制机房 - 根据算力板数量调整大小和颜色
        max_compute = max(room['max_compute'] for room in self.nodes['rooms'].values())
        min_compute = min(room['max_compute'] for room in self.nodes['rooms'].values())
        
        # 创建颜色映射
        norm = plt.Normalize(vmin=min_compute, vmax=max_compute)
        cmap = plt.cm.viridis
        
        for room_id, room in self.nodes['rooms'].items():
            # 计算大小和颜色
            size = 100 + 500 * (room['max_compute'] - min_compute) / (max_compute - min_compute + 1e-5)
            color = cmap(norm(room['max_compute']))
            
            plt.scatter(
                room['position'][1], room['position'][0],  # x_coord, y_coord
                s=size, color=color, marker='s', alpha=0.8,
                edgecolor='black', linewidth=1.5
            )
            # 添加机房ID和算力板数量标签
            # room_id_parts = room_id.split('_')
            # display_id = room_id_parts[0] if len(room_id_parts) > 0 else room_id
            # plt.text(
            #     room['position'][1] + 0.001, room['position'][0] + 0.001,
            #     f"Room{display_id}\n({room['max_compute']/50:.0f} boards)",
            #     fontsize=9, ha='left', va='bottom'
            # )
        
        # 添加连接线 - 基站到归属机房
        for bs_id, bs in self.nodes['base_stations'].items():
            if bs['home_room'] and bs['home_room'] in self.nodes['rooms']:
                home_room = self.nodes['rooms'][bs['home_room']]
                plt.plot(
                    [bs['position'][1], home_room['position'][1]],
                    [bs['position'][0], home_room['position'][0]],
                    'g-', linewidth=0.8, alpha=0.5
                )
        
        # 添加连接线 - 基站到分配机房
        for bs_id, bs in self.nodes['base_stations'].items():
            if bs['assigned_room'] and bs['assigned_room'] in self.nodes['rooms']:
                assigned_room = self.nodes['rooms'][bs['assigned_room']]
                plt.plot(
                    [bs['position'][1], assigned_room['position'][1]],
                    [bs['position'][0], assigned_room['position'][0]],
                    'r--', linewidth=0.8, alpha=0.7
                )
        
        # 添加图例
        legend_elements = [
            plt.Line2D([0], [0], marker='^', color='w', label='Base Station',
                       markerfacecolor='blue', markersize=10),
            plt.Line2D([0], [0], marker='s', color='w', label='Compute Room',
                       markerfacecolor='gray', markersize=10),
            plt.Line2D([0], [0], color='green', lw=2, label='Home Room Connection'),
            plt.Line2D([0], [0], color='red', lw=2, linestyle='--', label='Assigned Room Connection')
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        # 添加颜色条表示算力
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.7)
        cbar.set_label('Compute Capacity', fontsize=12)
        
        # 添加比例尺
        scale_lon = self.min_lon + 0.1 * (self.max_lon - self.min_lon)
        scale_lat = self.min_lat + 0.05 * (self.max_lat - self.min_lat)
        scale_km = 10  # 10公里比例尺
        
        # 计算10公里在经度上的大致距离
        point1 = (scale_lat, scale_lon)
        point2 = (scale_lat, scale_lon + 0.1)  # 初始猜测
        actual_dist = geodesic(point1, point2).km
        
        # 调整经度差以达到10公里
        lon_delta = 10 * 0.1 / actual_dist
        
        # plt.plot(
        #     [scale_lon, scale_lon + lon_delta],
        #     [scale_lat, scale_lat],
        #     'k-', linewidth=2
        # )
        # plt.text(
        #     scale_lon + lon_delta/2, scale_lat - 0.005,
        #     '10 km', fontsize=10, ha='center'
        # )
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Network topology visualization saved to {output_path}")