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
import os

class ComputingNetworkSimulator:
    def __init__(self, bs_csv_path, room_csv_path, rate, simulation_time, reward_weights=(0.7, 0.8, 0.3, 0.2), 
                 tidal_flow=False, tidal_period_count=20, tidal_amplitude=0.6):
        # 加载原始数据
        self.bs_df = pd.read_csv(bs_csv_path)
        self.bs_df['room_id'] = self.bs_df['home_compute_node_id'].astype(str)
        self.room_df = pd.read_csv(room_csv_path)
        self.room_df['room_id'] = self.room_df['id'].astype(str)
        
        # 1. 数据过滤 (针对真实数据)
        # 判断是否是经纬度数据
        is_real_data = self.bs_df['x_coord'].mean() > 100 
        
        if is_real_data:
            print(f"Detected Real-world Coordinates. Filtering and Normalizing to 0-100 based on BS area...")
            
            # 1. 过滤范围 (保留你的过滤逻辑)
            self.bs_df = self.bs_df[(self.bs_df['x_coord'] > 113.5) & (self.bs_df['x_coord'] < 114.5)]
            self.room_df = self.room_df[(self.room_df['x_coord'] > 113.5) & (self.room_df['x_coord'] < 114.5)]
            
            # 2. 【关键修改】获取基站(BS)的原始边界用于定义缩放范围
            bs_raw_min_x = self.bs_df['x_coord'].min()
            bs_raw_max_x = self.bs_df['x_coord'].max()
            bs_raw_min_y = self.bs_df['y_coord'].min()
            bs_raw_max_y = self.bs_df['y_coord'].max()
            
            # 3. 计算原始跨度 (以基站分布为准)
            bs_span_x = bs_raw_max_x - bs_raw_min_x
            bs_span_y = bs_raw_max_y - bs_raw_min_y
            
            # 4. 确定缩放目标，保持长宽比
            TARGET_SIZE = 95.0 # 目标映射的最大尺寸 (0-95, 留出2.5边距)
            
            # 修正地理畸变比例: 深圳附近(纬度22.5)，1度纬度 ≈ 111km，1度经度 ≈ 102km
            LAT_LON_RATIO = 111.0 / 102.0  
            
            # 修正后的 Y 跨度 (物理距离比例)
            corrected_bs_span_y = bs_span_y * LAT_LON_RATIO
            
            # 计算统一的缩放因子 (以最大的 corrected span 为基准)
            span_for_scaling = max(bs_span_x, corrected_bs_span_y)
            scale_factor = TARGET_SIZE / span_for_scaling
            
            # 5. 定义转换函数 (Min-Max Normalization with Aspect Ratio Preservation)
            OFFSET = 2.5 # 偏移量，让图形居中而不贴边
            
            # 转换函数基于 BS 的 min/max 和统一的 scale_factor
            def transform_x(lon): 
                # 基站被平均投影到 0-100 范围内
                return (lon - bs_raw_min_x) * scale_factor + OFFSET
            
            def transform_y(lat): 
                # 注意这里乘了 LAT_LON_RATIO 来修正地理畸变
                return (lat - bs_raw_min_y) * scale_factor * LAT_LON_RATIO + OFFSET
            
            # 6. 应用转换：同时应用于 BS 和 Room
            self.bs_df['x_coord'] = self.bs_df['x_coord'].apply(transform_x)
            self.bs_df['y_coord'] = self.bs_df['y_coord'].apply(transform_y)
            
            self.room_df['x_coord'] = self.room_df['x_coord'].apply(transform_x)
            self.room_df['y_coord'] = self.room_df['y_coord'].apply(transform_y)
            
            print("Base Stations normalized to 0-100 scale (Aspect Ratio Preserved).")
            print("Compute Room coordinates projected using the same scale (may exceed 0-100).")

        # 7. 无论是否是真实数据，都重新计算边界并强制修正为 0-100
        # 仿真的核心移动区域始终是 0-100
        self.min_x, self.max_x = 0, 100
        self.min_y, self.max_y = 0, 100

        self.request_artists = []
        
        # 添加云端节点
        self.cloud_node = {
            'node_id': 'cloud',
            'type': 'cloud',
            'compute': float('inf'),
            'position': (105, 105),
            'latency': 50  # 基础延迟，统一单位为毫秒
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
                'cumulative_compute_time': 0.0,  # 累计使用的 (算力 * 时间)
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
        
        # 修改：潮汐参数定义（移动到更合适的位置）
        self.base_rate = rate  # 基础请求率
        self.tidal_flow = tidal_flow
        self.total_time = simulation_time
        
        # 潮汐参数
        if tidal_flow:
            self.tidal_period = simulation_time / tidal_period_count  # 仿真时间内经历2个完整周期
            self.tidal_amplitude = tidal_amplitude  # 波动幅度 60%
            
            # 定义空间潮汐的两个重心
            if len(self.nodes['rooms']) > 0:
                room_positions = [room['position'] for room in self.nodes['rooms'].values()]
                # 自动计算两个重心
                self.tidal_center_1 = np.array([
                    np.quantile([p[0] for p in room_positions], 0.3), 
                    np.quantile([p[1] for p in room_positions], 0.3)
                ])
                self.tidal_center_2 = np.array([
                    np.quantile([p[0] for p in room_positions], 0.7), 
                    np.quantile([p[1] for p in room_positions], 0.7)
                ])
            else:
                # 备用重心
                self.tidal_center_1 = np.array([25.0, 25.0])
                self.tidal_center_2 = np.array([75.0, 75.0])
            
            print(f"潮汐模式已启用: 周期={self.tidal_period}s, 幅度={self.tidal_amplitude*100}%")
            print(f"潮汐重心: Center1={self.tidal_center_1}, Center2={self.tidal_center_2}")
        
        # 添加请求历史记录
        self.rate_history = []  # 记录请求率变化

        self.current_time = 0
        self.request_counter = 0
        self.pending_events = []  # 事件队列 (时间, 事件类型, 数据)
        self.current_request = None  # 当前正在处理的请求
        
        # 添加奖励权重参数
        self.reward_weights = reward_weights  # (基础权重, 延迟权重, 利用率权重, 一致性权重)
        # self.MAX_ABS_REWARD = 3.0
        self.MAX_ABS_REWARD = 5.0
        
        # 添加动作空间大小
        self.action_space_size = len(self.nodes['rooms']) + 1  # 云端 + 所有机房

        self._reset_dynamic_state()

        # 添加服务配置文件
        self.SERVICE_PROFILES = {
            "urgent_braking": (5, 1, 5, 3, 10, 0.1),  # 紧急制动
            "traffic_control": (10, 2, 15, 3, 15, 0.1),  # 交通信号控制
            "collision_avoidance": (50, 10, 20, 5, 40, 0.3),  # 避免碰撞
            "sensor_sharing": (60, 15, 50, 8, 30, 0.2),  # 传感器共享
            "hd_map_update": (80, 20, 100, 10, 100, 0.1),  # 高清地图更新
            "infotainment": (100, 25, 200, 15, 150, 0.2),  # 信息娱乐
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
            "ultra_low_latency": (1.5, 0.05),  # 极低延迟类：处理时间短
            "low_latency": (10, 1),  # 低延迟类：处理时间中等
            "high_latency_tolerance": (25, 3)  # 高延迟容忍类：处理时间长
        }

        # 添加边界属性
        self.min_x, self.max_x = 0, 100
        self.min_y, self.max_y = 0, 100

        self._precompute_static_data()

    def _precompute_static_data(self):
        """预计算静态数据以加速后续操作"""
        # 预排序节点ID列表
        self.sorted_room_ids = sorted(self.nodes['rooms'].keys())
        self.sorted_bs_ids = sorted(self.nodes['base_stations'].keys())
        
        # 预先构建一次状态以缓存边索引
        _ = self._get_state()

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
            room['cumulative_compute_time'] = 0.0
        
        # 清空事件队列
        self.pending_events = []
        self.current_request = None

    def _get_current_rate(self):
        """
        获取当前请求生成速率（支持潮汐模式）
        """
        if not self.tidal_flow:
            return self.base_rate
        
        # 使用正弦波模拟潮汐变化
        # 公式: Rate(t) = Base * (1 + A * sin(2 * pi * t / T))
        phase = 2 * math.pi * self.current_time / self.tidal_period
        factor = 1 + self.tidal_amplitude * math.sin(phase)
        current_rate = max(0.0, self.base_rate * factor)  # 保证rate至少为0.0
        
        return current_rate

    def _get_spatial_bias(self):
        """
        获取当前的空间偏置重心（用于潮汐模式）
        """
        if not self.tidal_flow:
            return None
        
        # 随时间在center_1和center_2之间平滑移动
        phase = 2 * math.pi * self.current_time / self.tidal_period
        weight = 0.5 * (1 + math.cos(phase))  # 使用cosine实现平滑过渡
        
        # 当前的目标重心
        target_center = (self.tidal_center_1 * weight + 
                        self.tidal_center_2 * (1 - weight))
        
        return target_center

    def _get_service_transition_prob(self, from_service, to_service):
        """
        定义服务类型之间的转换概率矩阵
        """
        transition_matrix = {
            "urgent_braking": {
                "urgent_braking": 0.7, "traffic_control": 0.2, "collision_avoidance": 0.1
            },
            "traffic_control": {
                "traffic_control": 0.6, "urgent_braking": 0.2, "sensor_sharing": 0.2
            },
            "collision_avoidance": {
                "collision_avoidance": 0.5, "sensor_sharing": 0.3, "hd_map_update": 0.2
            },
            "sensor_sharing": {
                "sensor_sharing": 0.4, "collision_avoidance": 0.3, "infotainment": 0.3
            },
            "hd_map_update": {
                "hd_map_update": 0.6, "infotainment": 0.4
            },
            "infotainment": {
                "infotainment": 0.8, "hd_map_update": 0.2
            }
        }
        
        return transition_matrix.get(from_service, {}).get(to_service, 0.1)

    def _reflect_boundary(self, value, min_val, max_val):
        """反射边界处理，避免边界聚集"""
        range_width = max_val - min_val
        
        if value < min_val:
            # 低于最小值，反射到范围内
            overshoot = min_val - value
            reflections = int(overshoot / range_width) + 1
            if reflections % 2 == 1:  # 奇数次反射
                value = min_val + (overshoot % range_width)
            else:  # 偶数次反射
                value = max_val - (overshoot % range_width)
        elif value > max_val:
            # 高于最大值，反射到范围内
            overshoot = value - max_val
            reflections = int(overshoot / range_width) + 1
            if reflections % 2 == 1:  # 奇数次反射
                value = max_val - (overshoot % range_width)
            else:  # 偶数次反射
                value = min_val + (overshoot % range_width)
        
        return np.clip(value, min_val, max_val)

    def _generate_request(self):
        """生成新的计算请求 - 集成潮汐模式和增强时空相关性"""
        self.request_counter += 1
        
        # 1. 使用动态速率（支持潮汐模式）
        current_rate = self._get_current_rate()
        inter_arrival = np.random.exponential(1/current_rate)
        self.current_time += inter_arrival
        
        # 记录当前请求率（用于调试和分析）
        if not hasattr(self, 'rate_history'):
            self.rate_history = []
        self.rate_history.append((self.current_time, current_rate))
        
        # 2. 根据SERVICE_PROFILES的概率选择服务类型
        service_types = list(self.SERVICE_PROFILES.keys())
        probs = [self.SERVICE_PROFILES[st][5] for st in service_types]
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
        
        # 3. 增强的时空相关性生成位置
        room_positions = [room['position'] for room in self.nodes['rooms'].values()]
        bs_positions = [bs['position'] for bs in self.nodes['base_stations'].values()]
        
        # 计算潮汐参数
        if self.tidal_flow:
            phase = 2 * math.pi * self.current_time / self.tidal_period
            tidal_strength = abs(math.sin(phase))
            # 获取当前潮汐重心
            current_center = self._get_spatial_bias()
        else:
            phase = 0
            tidal_strength = 0
            current_center = None
        
        # 超低延迟请求必须出现在基站附近
        if category == "ultra_low_latency" and bs_positions:
            # 随机选择一个基站作为中心    
            center_bs = random.choice(bs_positions)
            
            # 在基站周围近距离生成超低延迟请求
            position = (
                np.clip(np.random.normal(center_bs[0], 1.0), self.min_x, self.max_x),
                np.clip(np.random.normal(center_bs[1], 1.0), self.min_y, self.max_y)
            )
            
            is_hotspot = True
            current_direction = np.random.uniform(0, 2 * np.pi)
        
        else:
            # 增强的连续性概率计算（考虑潮汐强度）
            continuity_prob = 0.6 + tidal_strength * 0.3  # 潮汐强时连续性更强
            hotspot_prob = 0.3 + tidal_strength * 0.2     # 潮汐强时热点更集中
            
            # 增强连续性：基于前一个请求位置生成（模拟车辆移动）
            if self.request_history and random.random() < continuity_prob:
                last_request = self.request_history[-1]
                last_position = last_request['position']
                
                # 模拟车辆移动：方向持续性和速度变化
                if '_last_direction' in last_request:
                    # 保持方向持续性（60%概率保持方向，40%概率小幅变化）
                    if random.random() < 0.6:
                        direction = last_request['_last_direction']
                    else:
                        # 小幅变化方向
                        direction = last_request['_last_direction'] + np.random.normal(0, 0.5)
                else:
                    # 初始随机方向
                    direction = np.random.uniform(0, 2 * np.pi)
                
                # 潮汐影响：40%概率受潮汐方向影响
                if self.tidal_flow and random.random() < 0.4 and current_center is not None:
                    # 计算朝向潮汐重心的方向
                    dx_target = current_center[0] - last_position[0]
                    dy_target = current_center[1] - last_position[1]
                    target_direction = math.atan2(dy_target, dx_target)
                    
                    # 混合原有方向和潮汐方向
                    tidal_influence = 0.3 + 0.4 * tidal_strength
                    direction = (direction * (1 - tidal_influence) + 
                                target_direction * tidal_influence)
                
                # 速度变化：基于服务类型和潮汐调整
                speed_mean = 5  # 平均移动速度
                speed_std = 1   # 速度标准差
                
                # 根据服务类型调整速度
                if category == "high_latency_tolerance":
                    speed_mean = 7
                    speed_std = 1.5
                
                # 潮汐高峰期移动速度加快
                if self.tidal_flow:
                    speed_boost = 1.0 + 0.5 * tidal_strength
                    speed_mean *= speed_boost
                
                speed = np.clip(np.random.normal(speed_mean, speed_std), 1, 10)
                distance = speed * inter_arrival  # 基于时间间隔计算移动距离
                
                # 计算新位置
                dx = distance * np.cos(direction)
                dy = distance * np.sin(direction)
                
                new_x = last_position[0] + dx
                new_y = last_position[1] + dy
                
                # 使用反射边界处理
                new_x = self._reflect_boundary(new_x, self.min_x, self.max_x)
                new_y = self._reflect_boundary(new_y, self.min_y, self.max_y)
                
                position = (new_x, new_y)
                current_direction = direction
                is_hotspot = False
                
            # 在机房附近生成（模拟车辆靠近基础设施）
            elif random.random() < hotspot_prob and room_positions:
                # 选择热点中心（考虑潮汐重心）
                if self.tidal_flow and random.random() < 0.6 and current_center is not None:
                    center_x, center_y = current_center
                else:
                    # 随机选择一个机房作为中心
                    center_room = random.choice(room_positions)
                    center_x, center_y = center_room
                
                # 热点区域大小随潮汐变化
                if self.tidal_flow:
                    hotspot_radius = 8 + 6 * tidal_strength  # 半径在8-14之间变化
                else:
                    hotspot_radius = 10
                    
                x_offset = np.random.normal(0, hotspot_radius/3)
                y_offset = np.random.normal(0, hotspot_radius/3)
                
                position = (
                    self._reflect_boundary(center_x + x_offset, self.min_x, self.max_x),
                    self._reflect_boundary(center_y + y_offset, self.min_y, self.max_y)
                )
                
                current_direction = np.random.uniform(0, 2 * np.pi)
                is_hotspot = True
                
            else:
                # 随机生成，但受潮汐偏置影响
                if self.tidal_flow and random.random() < 0.3 and current_center is not None:
                    # 偏向潮汐重心区域
                    bias_strength = 0.3  # 偏置强度
                    base_x = np.random.beta(2, 2) * (self.max_x - self.min_x) + self.min_x
                    base_y = np.random.beta(2, 2) * (self.max_y - self.min_y) + self.min_y
                    
                    position = (
                        base_x * (1 - bias_strength) + current_center[0] * bias_strength,
                        base_y * (1 - bias_strength) + current_center[1] * bias_strength
                    )
                else:
                    position = (
                        np.random.beta(2, 2) * (self.max_x - self.min_x) + self.min_x,
                        np.random.beta(2, 2) * (self.max_y - self.min_y) + self.min_y
                    )
                current_direction = np.random.uniform(0, 2 * np.pi)
                is_hotspot = False
        
        # 4. 热点区域的请求计算需求调整
        if is_hotspot and random.random() < 0.7:
            compute_demand = np.clip(compute_demand * 1.5, 5, 100)
        
        # 5. 增强的服务类型连续性
        if self.request_history and random.random() < 0.4:  # 40%概率保持相同类型
            last_type = self.request_history[-1]['type']
            if last_type != service_type:
                # 考虑服务类型的自然转换概率
                transition_prob = self._get_service_transition_prob(last_type, service_type)
                if random.random() < transition_prob:
                    service_type = last_type
                    # 重新获取参数
                    params = self.SERVICE_PROFILES[service_type]
                    cpu_mean, cpu_std, mem_mean, mem_std, latency, prob = params
                    compute_demand = np.clip(np.random.normal(cpu_mean, cpu_std), 5, 100)
                    max_latency = latency
                    category = self.service_categories[service_type]
                    base_process, process_std = self.process_time_params[category]
                    process_time = np.clip(np.random.normal(base_process, process_std), 0.5, 50)
        
        # 6. 创建请求对象
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
            'is_hotspot': is_hotspot,
            '_last_direction': current_direction,
            'current_rate': current_rate,  # 记录生成时的请求率
            'tidal_strength': tidal_strength if self.tidal_flow else 0  # 记录潮汐强度
        }
        
        if not hasattr(self, 'request_history'):
            self.request_history = []
        if not self.request_history or self.request_history[-1] is not req:
            self.request_history.append(req)
        self.current_request = req
        
        # 调试信息（可选）
        # if self.request_counter % 10000 == 0 and self.tidal_flow:
        #     print(f"请求 #{self.request_counter}: 时间={self.current_time:.1f}s, "
        #         f"速率={current_rate:.2f}/s, 潮汐强度={tidal_strength:.2f}")
        
        return req

    def _greedy_policy(self, request):
        """
        greedy 贪婪策略: 总是选择总延迟最小的可用计算节点。
        
        Args:
            request (dict): 当前请求对象。
            
        Returns:
            str: 动作字符串（RoomID 或 'cloud'）
            float: 该动作下的理论总延迟
        """
        
        # 1. 预计算公共信息 (只计算一次)
        nearest_bs, bs_latency = self._find_nearest_bs(request['position'])
        home_room_id = nearest_bs['room_id']
        home_room = self.nodes['rooms'][home_room_id]
        
        bs_to_home = self._calculate_latency(
            nearest_bs['position'], home_room['position'], 'bs2room')
            
        # 打包公共信息
        bs_info = (nearest_bs, bs_latency, bs_to_home, home_room)

        candidates = {}

        # 2. 评估所有可用机房
        # 排序是为了保证确定性，避免字典序带来的随机误差
        for room_id, room in sorted(self.nodes['rooms'].items()):
            # 检查算力是否足够
            if room['compute'] >= request['compute_demand']:
                latency = self._compute_total_latency_for_candidate(
                    request, 'room', room, bs_info
                )
                candidates[room_id] = latency

        # 3. 评估云端
        cloud_latency = self._compute_total_latency_for_candidate(
            request, 'cloud', self.cloud_node, bs_info
        )
        candidates['cloud'] = cloud_latency

        # 4. 选择最优
        best_action = min(candidates, key=candidates.get)
        best_latency = candidates[best_action]

        return best_action, best_latency
    
    def _lowest_utilization_policy(self, request):
        """
        最低利用率优先策略 (LUF): 总是选择当前算力利用率最低的可用计算节点。
        
        Args:
            request (dict): 当前请求对象。
            
        Returns:
            str: 动作字符串（RoomID 或 'cloud'）
            float: 该动作下的理论总延迟（为了与原greedy函数签名一致，这里返回最小延迟）
        """
        
        best_utilization = float('inf')
        best_action = 'cloud' 
        
        # 1. 寻找最低利用率机房
        for room_id, room in sorted(self.nodes['rooms'].items()):
            if room['compute'] >= request['compute_demand'] and room['max_compute'] > 0:
                # 计算空闲率的反面：利用率
                current_utilization = (room['max_compute'] - room['compute']) / room['max_compute']
                
                if current_utilization < best_utilization:
                    best_utilization = current_utilization
                    best_action = room_id
        
        # 2. 计算该动作的理论延迟 (为了返回一致的格式)
        # 预计算公共信息
        nearest_bs, bs_latency = self._find_nearest_bs(request['position'])
        home_room_id = nearest_bs['room_id']
        home_room = self.nodes['rooms'][home_room_id]
        bs_to_home = self._calculate_latency(
            nearest_bs['position'], home_room['position'], 'bs2room')
        bs_info = (nearest_bs, bs_latency, bs_to_home, home_room)
        
        final_latency = 0
        if best_action == 'cloud':
            final_latency = self._compute_total_latency_for_candidate(
                request, 'cloud', self.cloud_node, bs_info
            )
        else:
            target_room = self.nodes['rooms'][best_action]
            final_latency = self._compute_total_latency_for_candidate(
                request, 'room', target_room, bs_info
            )
                                
        return best_action, final_latency

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
            return distance * 0.5  # 每单位距离对应0.5ms延迟
        if type == 'bs2room':
            return distance * 0.1  # 每单位距离对应0.1ms延迟
        if type == 'room2room':
            return distance * 0.1  # 每单位距离对应0.1ms延迟

    def _find_nearest_bs(self, position):
        """找到距离请求位置最近的基站"""
        min_latency = float('inf')
        nearest_bs = None
        
        x1, y1 = position
    
        # 先快速筛选候选基站
        candidates = []
        for bs in self.nodes['base_stations'].values():
            dx = abs(x1 - bs['position'][0])
            dy = abs(y1 - bs['position'][1])
            manhattan_dist = dx + dy
            
            # 如果曼哈顿距离已经很大，直接跳过
            if manhattan_dist < 100:  # 经验阈值
                candidates.append(bs)
        
        # 如果没有候选基站，则使用所有基站（确保不会漏掉）
        if not candidates:
            candidates = list(self.nodes['base_stations'].values())

        # 在候选基站中精确查找，使用您原有的延迟计算逻辑
        for bs in candidates:
            # 使用您原有的_calculate_latency函数
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

    def step(self, action, skip_generation=False):
        """
        执行调度动作。
        
        Args:
            action (str): 要执行的动作。
            skip_generation (bool): 如果为True，则跳过内部请求生成，
                                   并使用外部设置的 self.current_request。
                                   (用于微观对比测试)
        """
        # 处理到期事件（释放算力）
        self._process_pending_events()

        # 1. 立即获取当前请求。
        #    在训练模式下，这是上一步(或reset)生成的请求。
        #    在 skip_generation=True 模式下，这是 visualize.py 外部循环设置的请求。
        req = self.current_request
        
        # 2. 正常记录指标（针对当前请求）
        if not self.request_history or self.request_history[-1] is not req:
            self.request_history.append(req)
        self.metrics['total_requests'] += 1
        
        # 找到最近的基站及其所属的本地机房
        nearest_bs, bs_latency = self._find_nearest_bs(req['position'])
        home_room_id = nearest_bs['room_id']
        home_room = self.nodes['rooms'][home_room_id]
        bs_to_home = self._calculate_latency(nearest_bs['position'], home_room['position'], 'bs2room')
        bs_info = (nearest_bs, bs_latency, bs_to_home, home_room)
        req['home_room'] = home_room_id
        
        # 解析动作
        if action == 'cloud':
            # 使用云端算力
            req['target_room'] = 'cloud'
            allocations = {'cloud': req['compute_demand']}
            is_cloud = True
            allocated = 0 
            
            total_latency = self._compute_total_latency_for_candidate(
                req, 'cloud', self.cloud_node, bs_info
            )
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
                target_room = self.nodes['rooms'][target_room_id]
                
                total_latency = self._compute_total_latency_for_candidate(
                    req, 'room', target_room, bs_info
                )
                # 假设处理系数是 0.1，计算该请求占据的时间
                process_time = req['process_time']
                target_room['cumulative_compute_time'] += req['compute_demand'] * process_time

            else:
                # 分配失败，转用云端
                req['target_room'] = 'cloud'
                allocations = {'cloud': req['compute_demand']}
                is_cloud = True
                total_latency = self._compute_total_latency_for_candidate(
                    req, 'cloud', self.cloud_node, bs_info
                )
                self.metrics['cloud_requests'] += 1
        
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
        reward = self._calculate_reward(req, total_latency, is_cloud, is_success, action) 
        
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
        
        done = (self.current_time > self.total_time) and (not skip_generation) # 模拟1小时
        
        # 在函数的最后，*仅当* skip_generation=False 时，才生成 *下一个* 请求
        #    这个新请求将被存储在 self.current_request 中，
        #    并供 *下一次* step() 调用时使用。
        if not skip_generation and not done:
            self._generate_request()
        elif done:
            self.current_request = None

        next_state = self._get_state()

        return next_state, reward, done, metrics

    def get_global_average_utilization(self):
        """
        计算整个仿真期间的加权平均资源利用率
        公式: Sum(每个机房的累计算力时间) / (所有机房总容量 * 当前仿真时间)
        """
        if self.current_time <= 1.0: # 避免仿真初期时间过短导致除数接近0，造成数值爆炸
            return 0.0
            
        total_consumed_workload = 0.0
        total_capacity = 0.0
        
        for room in self.nodes['rooms'].values():
            total_consumed_workload += room['cumulative_compute_time']
            total_capacity += room['max_compute']
            
        if total_capacity == 0:
            return 0.0
            
        # 平均利用率 = 总消耗 / (总容量 * 仿真时长)
        avg_utilization = total_consumed_workload / (total_capacity * self.current_time)
        
        # 由于 cumulative_compute_time 是在请求到达时一次性加上的（包含未来的时间），
        # 所以在仿真早期，计算出的 utilization 可能会 > 1.0。
        # 为了防止 Reward 计算异常做一个软截断。
        return min(avg_utilization, 1.0)

    def _compute_total_latency_for_candidate(self, request, target_type, target_node, nearest_bs_info):
        """
        统一的延迟计算内核
        Args:
            request: 请求对象
            target_type: 'cloud' 或 'room'
            target_node: 目标节点对象 (cloud_node 或 room对象)
            nearest_bs_info: 包含 (bs_entity, bs_latency, bs_to_home_latency, home_room_entity) 的元组
        """
        nearest_bs, req2bs_latency, bs_to_home, home_room = nearest_bs_info
        
        # 1. 计算处理延迟
        # 假设处理系数是 0.1 (请根据您的实际设定调整)
        proc_latency = request['compute_demand'] * 0.1
        
        # 2. 计算网络路径延迟 (单程)
        if target_type == 'cloud':
            # 路径: BS -> Home -> Cloud
            network_latency = bs_to_home + target_node['latency']
        else:
            # 路径: BS -> Home -> Target Room
            if target_node['node_id'] == home_room['node_id']:
                network_latency = bs_to_home
            else:
                home_to_target = self._calculate_latency(
                    home_room['position'], target_node['position'], 'room2room')
                network_latency = bs_to_home + home_to_target
        
        # 3. 总延迟 = (接入RTT + 网络RTT) + 处理
        total_latency = (req2bs_latency * 2) + (network_latency * 2) + proc_latency
        
        return total_latency

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

    def _calculate_reward(self, req, latency, is_cloud, is_success, action):
        """计算奖励值（使用可配置权重）"""
        base_weight, latency_weight, long_term_util_weight, instant_util_weight = self.reward_weights
        
        # 基础奖励
        if is_success:
            if is_cloud:
                base = 1
            else:
                # 根据请求类型调整基础奖励
                category = self.service_categories[req['type']]
                if category == "ultra_low_latency":
                    base = 4.0 if req['target_room'] == req['home_room'] else 3.5
                elif category == "low_latency":
                    base = 3.5 if req['target_room'] == req['home_room'] else 3.0
                else:
                    base = 3.0 if req['target_room'] == req['home_room'] else 2.5
        else:
            # === 失败给予固定重罚 ===
            base = -5.0
        
        # --- 2. 延迟优化奖励/惩罚 ---
        # 如果成功，越快越好（延迟越小奖励越多）
        # 如果失败，越慢罚越重
        if is_success:
            # 延迟余量奖励：(最大允许 - 实际) * 系数
            latency_score = (req['max_latency'] - latency) * 0.1
        else:
            # 延迟超标惩罚：(实际 - 最大允许) * 系数
            latency_score = -(latency - req['max_latency']) * 0.2 # 失败时对延迟更敏感
            
        # 限制范围
        latency_score = np.clip(latency_score, -3.0, 2.0)
        
        # === 混合利用率奖励 ===
        long_term_util_bonus = 0
        instant_util_penalty = 0
        
        if is_success and not is_cloud:
            long_term_util_bonus = self._get_long_term_utilization_bonus()
            instant_util_penalty = self._get_instant_utilization_penalty(req, action)

            # 动态权重调整
            current_rate_factor = min(self.base_rate / 1.0, 2.0)
            long_term_util_bonus *= (long_term_util_weight * current_rate_factor)
            instant_util_penalty *= (instant_util_weight * current_rate_factor)
        
        # 机房选择的稳定性奖励
        # consistency_bonus = self._get_consistency_bonus(req, action)
        # consistency_bonus = 0
        
        # 应用权重
        base_reward = base * base_weight
        latency_part = latency_score * latency_weight
        
        reward = base_reward + latency_part + long_term_util_bonus - instant_util_penalty
        
        # 归一化
        scaled_reward = reward / self.MAX_ABS_REWARD
        
        return scaled_reward

    def _get_long_term_utilization_bonus(self):
        """改进的长期利用率奖励"""
        global_avg = self.get_global_average_utilization()
        
        if global_avg <= 0:
            return 0.0
        
        # 动态阈值调整（基于请求率）
        optimal_threshold = 0.4 + min(self.base_rate * 0.05, 0.4)  # 高请求率时提高最优阈值
        warning_threshold = optimal_threshold + 0.1
        
        if global_avg <= optimal_threshold:
            # 线性奖励：0% -> 0, optimal_threshold -> 2.0
            return (global_avg / optimal_threshold) * 2.0
        elif global_avg <= warning_threshold:
            # 警告区域：奖励递减
            progress = (global_avg - optimal_threshold) / (warning_threshold - optimal_threshold)
            return 2.0 * (1.0 - progress * 0.25)  # 从2.0线性降到1.5
        else:
            # 危险区域：惩罚
            excess = (global_avg - warning_threshold) / (1.0 - warning_threshold)
            return 1.5 - excess * 1.5  # 从1.5线性降到可能为负

    def _get_instant_utilization_penalty(self, req, action):
        """
        瞬时利用率惩罚：防止单个机房过载
        检查目标机房的当前利用率，过高则惩罚
        """
        if action == 'cloud' or action not in self.nodes['rooms']:
            return 0.0  # 云端或无目标机房，不惩罚
        
        target_room = self.nodes['rooms'][action]
        
        # 计算当前瞬时利用率
        if target_room['max_compute'] > 0:
            current_util = 1.0 - (target_room['compute'] / target_room['max_compute'])
        else:
            return 0.0
        
        # 无惩罚区域：利用率低于80%
        if current_util <= 0.8:
            return 0.0
        
        # 警告区域：80%-90%，轻微惩罚
        elif current_util <= 0.9:
            # 线性惩罚：从70%到85%，惩罚从0到1
            penalty = (current_util - 0.8) / 0.1
            return penalty
        
        # 危险区域：超过90%，重度惩罚
        else:
            # 基础惩罚 + 超额惩罚
            base_penalty = 1  # 90%的基础惩罚
            excess_penalty = (current_util - 0.9) * 2.0  # 超额部分加倍惩罚
            return base_penalty + excess_penalty

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
    
    def _build_node_index_map(self):
        """构建节点索引映射"""
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
        
        return node_index_map

    def _get_state(self):
        """优化状态获取 - 避免重复计算"""
        # 缓存固定部分（拓扑结构不变）
        if not hasattr(self, '_cached_edge_index'):
            # 只计算一次边索引
            self._cached_edge_index = self._build_static_edge_index()
        
        # 只动态计算节点特征
        node_features = self._get_dynamic_node_features()

        # 添加当前请求的特征
        if self.current_request:
            req = self.current_request
            span_x = max(self.max_x - self.min_x, 1e-6)
            span_y = max(self.max_y - self.min_y, 1e-6)
            max_service_latency = max(profile[4] for profile in self.SERVICE_PROFILES.values())
            req_features = [
                (req['position'][0] - self.min_x) / span_x,
                (req['position'][1] - self.min_y) / span_y,
                req['compute_demand'] / 100.0,
                req['max_latency'] / max_service_latency,
                req['process_time'] / 50.0
            ]
        else:
            # 处理没有请求的罕见情况（例如模拟结束时）
            req_features = [0.0, 0.0, 0.0, 0.0, 0.0]

        return {
            'x': torch.tensor(node_features, dtype=torch.float32),
            'edge_index': self._cached_edge_index,  # 使用缓存的边索引
            'valid_actions': self.get_valid_actions_mask(),
            # 将请求特征添加为图的全局属性 [1, 4]
            'req_feat': torch.tensor(req_features, dtype=torch.float32).unsqueeze(0) 
        }

    def _build_static_edge_index(self):
        """预先构建静态边索引"""
        edge_index = []
        node_index_map = self._build_node_index_map()
        
        # 构建连接边（与原来相同，但只执行一次）
        for bs_id, bs in self.nodes['base_stations'].items():
            room_id = bs['room_id']
            edge_index.append([node_index_map[bs_id], node_index_map[room_id]])
            edge_index.append([node_index_map[room_id], node_index_map[bs_id]])
        
        room_ids = sorted(self.nodes['rooms'].keys())
        for i in range(len(room_ids)):
            for j in range(i+1, len(room_ids)):
                src, dst = room_ids[i], room_ids[j]
                edge_index.append([node_index_map[src], node_index_map[dst]])
                edge_index.append([node_index_map[dst], node_index_map[src]])
        
        for room_id in room_ids:
            edge_index.append([node_index_map[room_id], node_index_map['cloud']])
            edge_index.append([node_index_map['cloud'], node_index_map[room_id]])
        
        return torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    def _get_dynamic_node_features(self):
        """只计算动态变化的节点特征"""
        node_features = []
        req_position = self.current_request['position'] if self.current_request else None
        max_distance = math.sqrt((self.max_x - self.min_x) ** 2 + (self.max_y - self.min_y) ** 2)
        max_room_compute = max(
            (room['max_compute'] for room in self.nodes['rooms'].values()),
            default=1.0
        )
        max_room_compute = max(max_room_compute, 1.0)
        nearest_bs_info = None
        max_latency = 1.0
        if self.current_request:
            nearest_bs, bs_latency = self._find_nearest_bs(self.current_request['position'])
            home_room = self.nodes['rooms'][nearest_bs['room_id']]
            bs_to_home = self._calculate_latency(nearest_bs['position'], home_room['position'], 'bs2room')
            nearest_bs_info = (nearest_bs, bs_latency, bs_to_home, home_room)
            max_latency = max(float(self.current_request['max_latency']), 1e-6)

        def req_distance_feature(position):
            if req_position is None or max_distance == 0:
                return 0.0
            return min(self._calculate_distance(req_position, position) / max_distance, 1.0)

        def norm_position(position):
            span_x = max(self.max_x - self.min_x, 1e-6)
            span_y = max(self.max_y - self.min_y, 1e-6)
            return (
                (position[0] - self.min_x) / span_x,
                (position[1] - self.min_y) / span_y
            )

        def latency_pressure(target_type, target_node):
            if not self.current_request or nearest_bs_info is None:
                return 0.0
            latency = self._compute_total_latency_for_candidate(
                self.current_request, target_type, target_node, nearest_bs_info
            )
            return min(latency / max_latency, 2.0) / 2.0
        
        # 云端节点（固定）
        cloud_x, cloud_y = norm_position(self.cloud_node['position'])
        node_features.append([
            cloud_x, cloud_y,
            1.0, 1.0, 0.0, 1.0,
            req_distance_feature(self.cloud_node['position']),
            latency_pressure('cloud', self.cloud_node)
        ])
        
        # 机房节点（动态：算力比率变化）
        for room_id, room in sorted(self.nodes['rooms'].items()):
            compute_ratio = room['compute'] / room['max_compute'] if room['max_compute'] > 0 else 0
            utilization_ratio = 1.0 - compute_ratio if room['max_compute'] > 0 else 0.0
            capacity_ratio = room['max_compute'] / max_room_compute
            room_x, room_y = norm_position(room['position'])
            node_features.append([
                room_x, room_y,
                0.5, compute_ratio, utilization_ratio, capacity_ratio,
                req_distance_feature(room['position']),
                latency_pressure('room', room)
            ])
        
        # 基站节点（固定）
        for bs_id, bs in sorted(self.nodes['base_stations'].items()):
            bs_x, bs_y = norm_position(bs['position'])
            bs_latency_pressure = 0.0
            if self.current_request:
                bs_latency_pressure = min(
                    self._calculate_latency(self.current_request['position'], bs['position'], 'req2bs') / max_latency,
                    2.0
                ) / 2.0
            node_features.append([
                bs_x, bs_y,
                0.0, 0.0, 0.0, 0.0,
                req_distance_feature(bs['position']),
                bs_latency_pressure
            ])
        
        return node_features

    def reset(self):
        """重置环境"""
        self._reset_dynamic_state()
        self.current_time = 0
        # 生成第一个请求
        self._generate_request()
        return self._get_state()

    def _all_positions(self):
        """获取所有实体的位置"""
        positions = [r['position'] for r in self.request_history]
        positions.extend([n['position'] for n in self.nodes['rooms'].values()])
        positions.extend([n['position'] for n in self.nodes['base_stations'].values()])
        positions.append(self.cloud_node['position'])
        return positions
