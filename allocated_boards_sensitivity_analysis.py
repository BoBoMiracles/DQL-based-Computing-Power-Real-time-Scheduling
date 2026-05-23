import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from simulator import ComputingNetworkSimulator
from gnn_dqn_agent import GNNAgent
from gnn_lstm_dqn_agent import LSTMDQNAgent
import os
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class AllocationTester:
    def __init__(self, bs_csv_path, room_csv_path, base_allocation_strategy='original'):
        """
        Initialize allocation tester
        
        Args:
            bs_csv_path: Base station data path
            room_csv_path: Room data path  
            base_allocation_strategy: Base allocation strategy
        """
        self.bs_csv_path = bs_csv_path
        self.room_csv_path = room_csv_path
        self.base_strategy = base_allocation_strategy
        
        # Load original data
        self.original_room_df = pd.read_csv(room_csv_path)
        self.original_bs_df = pd.read_csv(bs_csv_path)
        
        # Calculate total compute power
        self.total_compute_power = self.original_room_df['allocated_boards'].sum()
        self.active_rooms = self.original_room_df[self.original_room_df['is_active'] == 1]
        self.num_active_rooms = len(self.active_rooms)
        
        print(f"Total compute power: {self.total_compute_power} computing boards")
        print(f"Number of active rooms: {self.num_active_rooms}")
        print(f"Original allocation: {dict(zip(self.active_rooms['id'], self.active_rooms['allocated_boards']))}")
    
    def generate_allocation_strategies(self):
        """Generate different allocation strategies"""
        strategies = {}
        
        # 1. Original allocation
        strategies['original'] = {
            'description': 'Original optimal allocation',
            'allocations': dict(zip(self.active_rooms['id'], self.active_rooms['allocated_boards']))
        }
        
        # 2. Uniform allocation
        uniform_alloc = self.total_compute_power // self.num_active_rooms
        remainder = self.total_compute_power % self.num_active_rooms
        
        uniform_allocations = {}
        for i, room_id in enumerate(self.active_rooms['id']):
            alloc = uniform_alloc + (1 if i < remainder else 0)
            uniform_allocations[room_id] = max(1, alloc)  # Ensure at least 1 board
        
        strategies['uniform'] = {
            'description': f'Uniform allocation (approx {uniform_alloc} boards per room)',
            'allocations': uniform_allocations
        }
        
        # 3. Hotspot concentration allocation
        strategies['hotspot'] = self._generate_hotspot_allocation()
        
        # 4. Request density weighted allocation
        strategies['weighted'] = self._generate_weighted_allocation()
        
        # 5. Random allocation (multiple random seeds)
        for seed in [42, 123, 456]:
            strategies[f'random_seed{seed}'] = self._generate_random_allocation(seed)
        
        return strategies
    
    def _generate_hotspot_allocation(self):
        """Generate hotspot concentration allocation"""
        # Assume center area (40,40) to (60,60) as hotspot
        hotspot_center = (50, 50)
        hotspot_radius = 20
        
        # Calculate distance from rooms to hotspot
        room_distances = {}
        for _, room in self.active_rooms.iterrows():
            distance = np.sqrt((room['x_coord'] - hotspot_center[0])**2 + 
                             (room['y_coord'] - hotspot_center[1])**2)
            room_distances[room['id']] = distance
        
        # Closer distance gets more allocation (inverse weight)
        min_dist = min(room_distances.values())
        max_dist = max(room_distances.values())
        
        # Calculate weights: closer distance has higher weight
        weights = {}
        for room_id, dist in room_distances.items():
            # Normalized weight, closer distance has higher weight
            weight = 1.0 - (dist - min_dist) / (max_dist - min_dist + 1e-8)
            weights[room_id] = weight
        
        return self._allocate_by_weights(weights, "Hotspot concentration allocation")
    
    def _generate_weighted_allocation(self):
        """Generate request density weighted allocation"""
        # Estimate request density based on base station distribution
        bs_density = {}
        for _, room in self.active_rooms.iterrows():
            # Count connected base stations as weight
            connected_bs = len(self.original_bs_df[
                self.original_bs_df['home_compute_node_id'] == room['id']
            ])
            bs_density[room['id']] = connected_bs
        
        # If no base station connected, give minimum weight
        min_weight = 0.1
        for room_id in bs_density:
            if bs_density[room_id] == 0:
                bs_density[room_id] = min_weight
        
        return self._allocate_by_weights(bs_density, "Request density weighted allocation")
    
    def _allocate_by_weights(self, weights, description):
        """Allocate compute power based on weights"""
        total_weight = sum(weights.values())
        allocations = {}
        
        remaining_power = self.total_compute_power
        
        # First give minimum guarantee to each room
        min_allocation = 1  # Each room gets at least 1 board
        for room_id in weights:
            allocations[room_id] = min_allocation
            remaining_power -= min_allocation
        
        # Allocate remaining power by weight
        for room_id, weight in weights.items():
            share = weight / total_weight
            allocation = int(share * remaining_power)
            allocations[room_id] += allocation
        
        # Handle possible rounding errors
        allocated_total = sum(allocations.values())
        if allocated_total < self.total_compute_power:
            # Allocate remaining power to room with highest weight
            max_room = max(weights.items(), key=lambda x: x[1])[0]
            allocations[max_room] += self.total_compute_power - allocated_total
        
        return {
            'description': description,
            'allocations': allocations
        }
    
    def _generate_random_allocation(self, seed=42):
        """Generate random allocation"""
        np.random.seed(seed)
        room_ids = list(self.active_rooms['id'])
        
        # Generate random weights
        weights = {room_id: np.random.uniform(0.5, 2.0) for room_id in room_ids}
        
        allocations = self._allocate_by_weights(weights, f"Random allocation (seed={seed})")
        allocations['description'] = f"Random allocation (seed={seed})"
        
        return allocations
    
    def create_room_csv(self, allocation_strategy, output_dir='allocation_test_data'):
        """Create room CSV file for specified allocation strategy"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Copy original data
        new_room_df = self.original_room_df.copy()
        
        # Update allocated_boards
        for room_id, allocation in allocation_strategy['allocations'].items():
            mask = new_room_df['id'] == room_id
            new_room_df.loc[mask, 'allocated_boards'] = allocation
        
        # Save new file
        filename = f"room_{allocation_strategy['description'].replace(' ', '_').replace('(', '').replace(')', '')}.csv"
        filepath = os.path.join(output_dir, filename)
        new_room_df.to_csv(filepath, index=False)
        
        return filepath
    
    
    def test_allocation_strategies(self, rates=[3], simulation_time=36000, 
                                    model_types=['gnn'], num_episodes=100, 
                                    lstm_lengths=[5, 10, 15]):
        """Test all allocation strategies, support different LSTM window lengths"""
        results = {}
        
        # Generate all allocation strategies
        strategies = self.generate_allocation_strategies()
        
        for strategy_name, strategy in strategies.items():
            print(f"\n{'='*60}")
            print(f"Testing allocation strategy: {strategy_name} - {strategy['description']}")
            print(f"Allocation details: {strategy['allocations']}")
            print(f"\n{'='*60}")
            
            # Create new room CSV file
            room_csv_path = self.create_room_csv(strategy)
            
            strategy_results = {}
            
            for rate in rates:
                print(f"\nRequest rate: {rate}/s")
                
                # Initialize environment
                env = ComputingNetworkSimulator(
                    self.bs_csv_path, 
                    room_csv_path, 
                    rate=rate, 
                    simulation_time=simulation_time
                )
                
                rate_results = {}
                
                for model_type in model_types:
                    # Handle different window lengths for LSTM models
                    if model_type == 'gnn_lstm':
                        for lstm_len in lstm_lengths:
                            model_key = f"{model_type}_len{lstm_len}"
                            print(f"Training model: {model_key}")
                            
                            # Train and test model
                            model_results = self._train_and_test_model(
                                env, model_type, rate, num_episodes, lstm_len
                            )
                            
                            rate_results[model_key] = model_results
                    else:
                        # Non-LSTM models
                        print(f"Training model: {model_type}")
                        
                        model_results = self._train_and_test_model(
                            env, model_type, rate, num_episodes, lstm_len=None
                        )
                        
                        rate_results[model_type] = model_results
                
                strategy_results[rate] = rate_results
            
            results[strategy_name] = {
                'strategy_info': strategy,
                'performance': strategy_results
            }
        
        return results

    def _train_and_test_model(self, env, model_type, rate, num_episodes, lstm_len=None):
        """Train and test a single model, support different LSTM window lengths"""
        # Create corresponding agent based on model type
        if model_type == 'gnn':
            agent = GNNAgent(env, device='cuda')
        elif model_type == 'gnn_lstm':
            if lstm_len is None:
                lstm_len = 5  # Default window length
            agent = LSTMDQNAgent(env, device='cuda', history_len=lstm_len)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Simplified training process
        training_metrics = self._simplified_training(agent, env, num_episodes)
        
        # Test performance
        test_metrics = self._test_model(agent, env, num_test_episodes=10)
        
        return {
            'training_metrics': training_metrics,
            'test_metrics': test_metrics,
            'allocation_summary': self._get_allocation_summary(env),
            'lstm_history_len': lstm_len if model_type == 'gnn_lstm' else None
        }
    
    def _simplified_training(self, agent, env, num_episodes):
        """Simplified training process"""
        rewards_history = []
        success_rates = []
        
        for episode in range(num_episodes):
            state = env.reset()
            if hasattr(agent, 'reset_hidden_state'):
                agent.reset_hidden_state()
            
            episode_reward = 0
            episode_successes = 0
            episode_requests = 0
            
            done = False
            while not done:
                action = agent.get_action(state, epsilon=max(0.1, 1.0 - episode/num_episodes))
                next_state, reward, done, metrics = env.step(action)
                
                episode_reward += reward
                if metrics.get('last_success', False):
                    episode_successes += 1
                episode_requests += 1
                
                state = next_state
            
            rewards_history.append(episode_reward)
            success_rate = episode_successes / episode_requests if episode_requests > 0 else 0
            success_rates.append(success_rate)
        
        return {
            'final_reward': np.mean(rewards_history[-10:]),  # Average of last 10 episodes
            'final_success_rate': np.mean(success_rates[-10:]),
            'convergence_episode': self._find_convergence_episode(success_rates),
            'rewards_history': rewards_history,
            'success_rates_history': success_rates
        }
    
    def _test_model(self, agent, env, num_test_episodes=10):
        """Test model performance"""
        test_metrics = {
            'total_requests': 0,
            'succeed_requests': 0,
            'cloud_requests': 0,
            'total_latency': 0,
            'total_reward': 0
        }
        
        for _ in range(num_test_episodes):
            state = env.reset()
            if hasattr(agent, 'reset_hidden_state'):
                agent.reset_hidden_state()
            
            done = False
            while not done:
                action = agent.get_action(state, epsilon=0.01)  # Small epsilon for testing
                next_state, reward, done, metrics = env.step(action)
                
                # Accumulate metrics
                test_metrics['total_requests'] += 1
                test_metrics['total_reward'] += reward
                test_metrics['total_latency'] += metrics.get('last_latency', 0)
                
                if metrics.get('last_success', False):
                    test_metrics['succeed_requests'] += 1
                if metrics.get('used_cloud', False):
                    test_metrics['cloud_requests'] += 1
                
                state = next_state
        
        # Calculate averages
        if test_metrics['total_requests'] > 0:
            test_metrics['success_rate'] = test_metrics['succeed_requests'] / test_metrics['total_requests']
            test_metrics['avg_latency'] = test_metrics['total_latency'] / test_metrics['succeed_requests'] if test_metrics['succeed_requests'] > 0 else 0
            test_metrics['cloud_usage_rate'] = test_metrics['cloud_requests'] / test_metrics['total_requests']
            test_metrics['avg_reward'] = test_metrics['total_reward'] / test_metrics['total_requests']
        
        return test_metrics
    
    def _find_convergence_episode(self, success_rates, window=10, threshold=0.95):
        """Find convergence episode"""
        if len(success_rates) < window:
            return len(success_rates)
        
        for i in range(len(success_rates) - window):
            window_avg = np.mean(success_rates[i:i+window])
            if window_avg >= threshold * success_rates[-1]:
                return i + window
        
        return len(success_rates)
    
    def _get_allocation_summary(self, env):
        """Get allocation summary"""
        allocations = {}
        for room_id, room in env.nodes['rooms'].items():
            allocations[room_id] = room['max_compute'] / 50  # Convert back to number of boards
        
        return {
            'allocations': allocations,
            'gini_coefficient': self._calculate_gini_coefficient(list(allocations.values())),
            'allocation_entropy': self._calculate_allocation_entropy(allocations)
        }
    
    def _calculate_gini_coefficient(self, values):
        """Calculate Gini coefficient (measure of inequality)"""
        values = np.array(values)
        values = values[values > 0]  # Only consider positive values
        n = len(values)
        if n == 0:
            return 0
        
        # Calculate Gini coefficient
        sorted_values = np.sort(values)
        index = np.arange(1, n + 1)
        gini = (np.sum((2 * index - n - 1) * sorted_values)) / (n * np.sum(sorted_values))
        return gini
    
    def _calculate_allocation_entropy(self, allocations):
        """Calculate allocation entropy (measure of distribution uniformity)"""
        values = np.array(list(allocations.values()))
        values = values[values > 0]
        probabilities = values / values.sum()
        entropy = -np.sum(probabilities * np.log(probabilities))
        return entropy

def visualize_allocation_results(results, output_dir='allocation_test_results2'):
    """Visualize allocation strategy test results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data
    strategy_data = []
    performance_data = []
    
    for strategy_name, strategy_results in results.items():
        strategy_info = strategy_results['strategy_info']
        
        for rate, rate_results in strategy_results['performance'].items():
            for model_type, model_results in rate_results.items():
                test_metrics = model_results['test_metrics']
                allocation_summary = model_results['allocation_summary']
                training_metrics = model_results['training_metrics']
                
                # Strategy information
                strategy_data.append({
                    'strategy': strategy_name,
                    'description': strategy_info['description'],
                    'rate': rate,
                    'model_type': model_type,
                    'gini_coefficient': allocation_summary['gini_coefficient'],
                    'allocation_entropy': allocation_summary['allocation_entropy'],
                    'allocations': str(allocation_summary['allocations'])
                })
                
                # Performance information
                performance_data.append({
                    'strategy': strategy_name,
                    'rate': rate,
                    'model_type': model_type,
                    'success_rate': test_metrics.get('success_rate', 0),
                    'avg_latency': test_metrics.get('avg_latency', 0),
                    'cloud_usage_rate': test_metrics.get('cloud_usage_rate', 0),
                    'avg_reward': test_metrics.get('avg_reward', 0),
                    'final_training_reward': training_metrics.get('final_reward', 0),
                    'convergence_episode': training_metrics.get('convergence_episode', 0)
                })
    
    # Convert to DataFrame
    strategy_df = pd.DataFrame(strategy_data)
    performance_df = pd.DataFrame(performance_data)
    
    # Save raw data
    strategy_df.to_csv(f'{output_dir}/allocation_strategies.csv', index=False)
    performance_df.to_csv(f'{output_dir}/performance_results.csv', index=False)
    
    # Visualization
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Performance Comparison of Different Compute Allocation Strategies', fontsize=16, fontweight='bold')
    
    # 1. Success rate comparison
    sns.barplot(data=performance_df, x='strategy', y='success_rate', hue='model_type', ax=axes[0,0])
    axes[0,0].set_title('Success Rate Comparison')
    axes[0,0].set_ylabel('Success Rate')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # 2. Average latency comparison
    sns.barplot(data=performance_df, x='strategy', y='avg_latency', hue='model_type', ax=axes[0,1])
    axes[0,1].set_title('Average Latency Comparison')
    axes[0,1].set_ylabel('Latency (ms)')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # 3. Cloud usage rate comparison
    sns.barplot(data=performance_df, x='strategy', y='cloud_usage_rate', hue='model_type', ax=axes[0,2])
    axes[0,2].set_title('Cloud Usage Rate Comparison')
    axes[0,2].set_ylabel('Cloud Usage Rate')
    axes[0,2].tick_params(axis='x', rotation=45)
    
    # 4. Training convergence speed comparison
    sns.barplot(data=performance_df, x='strategy', y='convergence_episode', hue='model_type', ax=axes[1,0])
    axes[1,0].set_title('Training Convergence Speed Comparison')
    axes[1,0].set_ylabel('Episodes to Convergence')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # 5. Gini coefficient vs success rate
    merged_df = performance_df.merge(strategy_df, on=['strategy', 'rate', 'model_type'])
    sns.scatterplot(data=merged_df, x='gini_coefficient', y='success_rate', 
                   hue='strategy', size='avg_reward', ax=axes[1,1])
    axes[1,1].set_title('Allocation Inequality vs Success Rate')
    axes[1,1].set_xlabel('Gini Coefficient (Inequality)')
    axes[1,1].set_ylabel('Success Rate')
    
    # 6. Allocation entropy vs cloud usage rate
    sns.scatterplot(data=merged_df, x='allocation_entropy', y='cloud_usage_rate',
                   hue='strategy', size='avg_latency', ax=axes[1,2])
    axes[1,2].set_title('Allocation Uniformity vs Cloud Usage Rate')
    axes[1,2].set_xlabel('Allocation Entropy (Uniformity)')
    axes[1,2].set_ylabel('Cloud Usage Rate')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/allocation_strategy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot training curve comparison
    plt.figure(figsize=(12, 8))
    for strategy_name, strategy_results in results.items():
        for rate, rate_results in strategy_results['performance'].items():
            for model_type, model_results in rate_results.items():
                rewards_history = model_results['training_metrics']['rewards_history']
                success_rates = model_results['training_metrics']['success_rates_history']
                
                # Smooth curves
                window = max(1, len(rewards_history) // 50)
                smooth_rewards = pd.Series(rewards_history).rolling(window=window, center=True).mean()
                smooth_success = pd.Series(success_rates).rolling(window=window, center=True).mean()
                
                plt.plot(smooth_rewards, label=f'{strategy_name}_{model_type}')
    
    plt.title('Training Reward Curves for Different Allocation Strategies')
    plt.xlabel('Training Episode')
    plt.ylabel('Average Reward (Smoothed)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_curves_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization results saved to: {output_dir}")

def generate_test_report(results, report_path):
    """Generate detailed test report with all metrics"""
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Compute Allocation Strategy Test Report\n\n")
        
        f.write("## Test Overview\n")
        f.write("This test evaluates the impact of different compute allocation strategies on reinforcement learning model performance.\n\n")
        
        f.write("## Test Results Summary\n")
        f.write("| Strategy | Success Rate | Avg Latency (ms) | Cloud Usage Rate | Convergence Episodes |\n")
        f.write("|----------|--------------|------------------|------------------|----------------------|\n")
        
        best_strategy = None
        best_score = 0
        
        for strategy_name, strategy_results in results.items():
            # Calculate comprehensive metrics
            avg_success = 0
            avg_latency = 0
            avg_cloud_usage = 0
            avg_convergence = 0
            count = 0
            
            for rate, rate_results in strategy_results['performance'].items():
                for model_type, model_results in rate_results.items():
                    test_metrics = model_results['test_metrics']
                    training_metrics = model_results['training_metrics']
                    
                    success_rate = test_metrics.get('success_rate', 0)
                    latency = test_metrics.get('avg_latency', 0)
                    cloud_usage = test_metrics.get('cloud_usage_rate', 0)
                    convergence = training_metrics.get('convergence_episode', 0)
                    
                    avg_success += success_rate
                    avg_latency += latency
                    avg_cloud_usage += cloud_usage
                    avg_convergence += convergence
                    count += 1
            
            if count > 0:
                avg_success /= count
                avg_latency /= count
                avg_cloud_usage /= count
                avg_convergence /= count
                
                if avg_success > best_score:
                    best_score = avg_success
                    best_strategy = strategy_name
                
                # Write all metrics to table
                f.write(f"| {strategy_name} | {avg_success:.3f} | {avg_latency:.2f} | {avg_cloud_usage:.3f} | {avg_convergence:.0f} |\n")
        
        f.write(f"\n## Best Strategy\n")
        f.write(f"Based on test results, **{best_strategy}** strategy performed best.\n\n")
        
        f.write("## Detailed Analysis\n")
        for strategy_name, strategy_results in results.items():
            f.write(f"### {strategy_name}\n")
            f.write(f"{strategy_results['strategy_info']['description']}\n\n")
            
            f.write("Allocation details:\n")
            f.write("```python\n")
            f.write(f"{strategy_results['strategy_info']['allocations']}\n")
            f.write("```\n\n")
        
        f.write("## Conclusions and Recommendations\n")
        f.write("1. Compute allocation has a significant impact on model performance\n")
        f.write("2. It is recommended to optimize compute allocation based on actual request distribution\n")
        f.write("3. Regularly re-evaluate and adjust allocation strategies\n")

def main():
    """Main test function"""
    # Initialize tester
    tester = AllocationTester(
        bs_csv_path='gurobi_solution_service_sources.csv',
        room_csv_path='gurobi_solution_compute_nodes.csv'
    )
    
    # Test parameters
    test_rates = [0.1]  # Request rate
    test_time = 12000  # Simulation time
    test_episodes = 100  # Training episodes
    model_types = ['gnn']  # Model types to test
    lstm_window = [5, 10, 15]  # LSTM window lengths
    
    print("Starting compute allocation strategy testing...")
    print(f"Test parameters: Request rate={test_rates}, Simulation time={test_time}s, Training episodes={test_episodes}")
    
    # Execute tests
    results = tester.test_allocation_strategies(
        rates=test_rates,
        simulation_time=test_time,
        model_types=model_types,
        num_episodes=test_episodes,
        lstm_lengths=lstm_window
    )
    
    # Visualize results
    visualize_allocation_results(results)
    
    # Generate test report
    generate_test_report(results, 'allocation_test_results2/test_report.md')
    
    print("Testing completed!")

if __name__ == "__main__":
    main()