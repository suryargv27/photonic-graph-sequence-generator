import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import copy
from graph_gflownet import GFlowNetPolicy

# Constants
T1 = 0.1
T2 = 0.1
T3 = 10.0
B = 0.5

class ImprovedGFlowNet:
    """Improved GFlowNet with better exploration and variance reduction"""
    
    def __init__(self, n, hidden_dim=128, lr=1e-3, replay_buffer_size=1000):
        self.n = n
        self.policy = GFlowNetPolicy(n, hidden_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        # Flow network components
        self.Z = nn.Parameter(torch.tensor(0.0))
        
        # Replay buffer for experience replay
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        
        # Statistics
        self.training_stats = {
            'losses': [],
            'best_costs': [],
            'avg_costs': [],
            'success_rates': []
        }
    
    def add_to_replay_buffer(self, trajectory_data):
        """Add trajectory to replay buffer"""
        self.replay_buffer.append(trajectory_data)
    
    def sample_from_replay_buffer(self, batch_size):
        """Sample batch from replay buffer"""
        if len(self.replay_buffer) < batch_size:
            return list(self.replay_buffer)
        indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        return [self.replay_buffer[i] for i in indices]
    
    def sample_trajectory(self, env, temperature=1.0, epsilon=0.0):
        """
        Sample trajectory with temperature and epsilon-greedy
        
        Args:
            env: Environment
            temperature: Temperature for softmax (higher = more exploration)
            epsilon: Probability of random action
        """
        env.reset()
        trajectory = []
        states = [env.get_state()]
        actions = []
        costs = []
        
        done = False
        steps = 0
        max_steps = 100  # Prevent infinite loops
        
        while not done and steps < max_steps:
            valid_actions = env.get_valid_actions()
            
            if len(valid_actions) == 0:
                break
            
            # Epsilon-greedy with temperature
            if np.random.rand() < epsilon:
                action_idx = np.random.randint(len(valid_actions))
            else:
                with torch.no_grad():
                    logits = self.policy(env.get_state(), valid_actions)
                    if len(logits) == 0:
                        break
                    # Apply temperature
                    probs = F.softmax(logits / temperature, dim=0)
                    action_idx = Categorical(probs).sample().item()
            
            action = valid_actions[action_idx]
            actions.append(action)
            
            next_state, cost, done = env.step(action)
            costs.append(cost)
            states.append(next_state)
            
            trajectory.append({
                'state': states[-2],
                'action': action,
                'valid_actions': valid_actions,
                'action_idx': action_idx,
                'cost': cost,
                'next_state': next_state,
                'done': done
            })
            
            steps += 1
        
        return trajectory, states, actions, costs, done
    
    def compute_reward(self, total_cost, success):
        """
        Compute reward based on cost and success
        
        Args:
            total_cost: Total cost of trajectory
            success: Whether goal was reached
        """
        if not success:
            # Penalty for not reaching goal
            return np.exp(-100.0)
        # Reward is exponential of negative cost
        return np.exp(-total_cost)
    
    def train_step(self, env, num_trajectories=32, use_replay=True, replay_ratio=0.5, 
                   temperature=1.0, epsilon=0.1):
        """
        Training step with experience replay
        
        Args:
            env: Environment
            num_trajectories: Number of new trajectories to sample
            use_replay: Whether to use experience replay
            replay_ratio: Ratio of replay samples to new samples
            temperature: Sampling temperature
            epsilon: Epsilon for epsilon-greedy
        """
        trajectories = []
        
        # Sample new trajectories
        num_new = num_trajectories
        if use_replay and len(self.replay_buffer) > 0:
            num_new = int(num_trajectories * (1 - replay_ratio))
        
        for _ in range(num_new):
            traj, states, actions, costs, success = self.sample_trajectory(
                env, temperature=temperature, epsilon=epsilon
            )
            if len(traj) > 0:
                total_cost = sum(costs)
                reward = self.compute_reward(total_cost, success)
                traj_data = {
                    'trajectory': traj,
                    'states': states,
                    'actions': actions,
                    'costs': costs,
                    'total_cost': total_cost,
                    'reward': reward,
                    'success': success
                }
                trajectories.append(traj_data)
                self.add_to_replay_buffer(traj_data)
        
        # Add replay samples
        if use_replay and len(self.replay_buffer) > 0:
            num_replay = num_trajectories - num_new
            replay_samples = self.sample_from_replay_buffer(num_replay)
            trajectories.extend(replay_samples)
        
        if len(trajectories) == 0:
            return 0.0, 0.0, 0.0
        
        # Compute statistics
        avg_cost = np.mean([t['total_cost'] for t in trajectories])
        best_cost = np.min([t['total_cost'] for t in trajectories])
        success_rate = np.mean([t['success'] for t in trajectories])
        
        # Trajectory Balance loss with baseline
        losses = []
        
        # Compute baseline (mean log reward)
        log_rewards = [np.log(t['reward'] + 1e-10) for t in trajectories]
        baseline = np.mean(log_rewards)
        
        for traj_data in trajectories:
            traj = traj_data['trajectory']
            reward = traj_data['reward']
            
            # Forward policy log probs
            log_pf = 0.0
            for step in traj:
                logits = self.policy(step['state'], step['valid_actions'])
                if len(logits) > 0:
                    log_probs = F.log_softmax(logits, dim=0)
                    action_idx = step['valid_actions'].index(step['action'])
                    log_pf += log_probs[action_idx]
            
            # TB loss with baseline for variance reduction
            log_reward = torch.log(torch.tensor(reward + 1e-10))
            loss = (self.Z + log_pf - log_reward) ** 2
            losses.append(loss)
        
        # Optimize
        if len(losses) > 0:
            total_loss = torch.stack(losses).mean()
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.policy.parameters()) + [self.Z], 
                max_norm=1.0
            )
            self.optimizer.step()
            
            return total_loss.item(), avg_cost, best_cost
        
        return 0.0, avg_cost, best_cost
    
    def generate_solution(self, env, greedy=True, temperature=1.0):
        """Generate solution using trained policy"""
        env.reset()
        trajectory = []
        
        done = False
        steps = 0
        max_steps = 100
        
        while not done and steps < max_steps:
            valid_actions = env.get_valid_actions()
            
            if len(valid_actions) == 0:
                break
            
            with torch.no_grad():
                logits = self.policy(env.get_state(), valid_actions)
                if len(logits) == 0:
                    break
                
                if greedy:
                    action_idx = torch.argmax(logits).item()
                else:
                    probs = F.softmax(logits / temperature, dim=0)
                    action_idx = Categorical(probs).sample().item()
            
            action = valid_actions[action_idx]
            next_state, cost, done = env.step(action)
            
            trajectory.append({
                'action': action,
                'cost': cost
            })
            
            steps += 1
        
        return trajectory, env.total_cost, env.is_terminal()


def train_improved_gflownet(n, A, T, num_iterations=1000, num_trajectories=32,
                           use_replay=True, plot_results=True):
    """Train improved GFlowNet with curriculum learning"""
    from graph_gflownet import GraphEnvironment, GFlowNetPolicy
    
    env = GraphEnvironment(n, A, T)
    gfn = ImprovedGFlowNet(n, hidden_dim=128, lr=1e-3)
    
    print("Training Improved GFlowNet...")
    print(f"Graph size: {n} nodes")
    print(f"Initial state: A has {np.sum(A)} edges")
    print(f"Using experience replay: {use_replay}")
    print()
    
    best_cost = float('inf')
    best_trajectory = None
    
    # Curriculum learning: gradually reduce exploration
    for iteration in range(num_iterations):
        # Decay temperature and epsilon
        temperature = max(0.5, 2.0 * (1 - iteration / num_iterations))
        epsilon = max(0.05, 0.3 * (1 - iteration / num_iterations))
        
        loss, avg_cost, batch_best_cost = gfn.train_step(
            env, 
            num_trajectories=num_trajectories,
            use_replay=use_replay,
            temperature=temperature,
            epsilon=epsilon
        )
        
        # Record statistics
        gfn.training_stats['losses'].append(loss)
        gfn.training_stats['avg_costs'].append(avg_cost)
        gfn.training_stats['best_costs'].append(batch_best_cost)
        
        # Evaluate
        if iteration % 50 == 0:
            # Test with greedy policy
            test_costs = []
            test_successes = 0
            
            for _ in range(10):
                trajectory, total_cost, success = gfn.generate_solution(env, greedy=True)
                test_costs.append(total_cost)
                if success:
                    test_successes += 1
                
                if success and total_cost < best_cost:
                    best_cost = total_cost
                    best_trajectory = trajectory
            
            avg_test_cost = np.mean(test_costs)
            success_rate = test_successes / 10.0
            gfn.training_stats['success_rates'].append(success_rate)
            
            print(f"Iter {iteration:4d}: Loss={loss:.4f}, "
                  f"Train Avg={avg_cost:.4f}, Test Avg={avg_test_cost:.4f}, "
                  f"Success={success_rate:.2f}, Best={best_cost:.4f}, "
                  f"T={temperature:.2f}, ε={epsilon:.2f}")
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best cost found: {best_cost:.4f}")
    
    if best_trajectory:
        print(f"\nBest trajectory ({len(best_trajectory)} steps):")
        for i, step in enumerate(best_trajectory):
            print(f"  Step {i+1}: {step['action'][0]:10s} - cost: {step['cost']:.4f}")
        print(f"  Total cost: {best_cost:.4f}")
    
    # Plot training curves
    if plot_results and len(gfn.training_stats['losses']) > 0:
        plot_training_stats(gfn.training_stats)
    
    return gfn, best_cost, best_trajectory


def plot_training_stats(stats):
    """Plot training statistics"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Loss
    axes[0, 0].plot(stats['losses'])
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].grid(True)
    
    # Average cost
    axes[0, 1].plot(stats['avg_costs'])
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Average Cost')
    axes[0, 1].set_title('Average Training Cost')
    axes[0, 1].grid(True)
    
    # Best cost
    axes[1, 0].plot(stats['best_costs'])
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Best Cost')
    axes[1, 0].set_title('Best Cost per Batch')
    axes[1, 0].grid(True)
    
    # Success rate
    if len(stats['success_rates']) > 0:
        x = np.arange(len(stats['success_rates'])) * 50
        axes[1, 1].plot(x, stats['success_rates'])
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Success Rate')
        axes[1, 1].set_title('Test Success Rate')
        axes[1, 1].set_ylim([0, 1.1])
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_stats.png', dpi=150, bbox_inches='tight')
    print("\nTraining statistics plot saved to training_stats.png")
    plt.close()


def compare_policies(n, A, T, num_tests=50):
    """Compare random policy vs trained GFlowNet"""
    from graph_gflownet import GraphEnvironment
    
    env = GraphEnvironment(n, A, T)
    
    print("Comparing Random Policy vs GFlowNet...")
    print()
    
    # Train GFlowNet
    gfn, best_cost, best_traj = train_improved_gflownet(
        n, A, T, 
        num_iterations=500, 
        num_trajectories=32,
        plot_results=False
    )
    
    # Test random policy
    print("\nTesting Random Policy:")
    random_costs = []
    random_successes = 0
    
    for i in range(num_tests):
        env.reset()
        total_cost = 0
        done = False
        steps = 0
        max_steps = 100
        
        while not done and steps < max_steps:
            valid_actions = env.get_valid_actions()
            if len(valid_actions) == 0:
                break
            
            action = valid_actions[np.random.randint(len(valid_actions))]
            _, cost, done = env.step(action)
            total_cost += cost
            steps += 1
        
        if done:
            random_costs.append(total_cost)
            random_successes += 1
    
    # Test GFlowNet
    print("\nTesting GFlowNet Policy:")
    gfn_costs = []
    gfn_successes = 0
    
    for i in range(num_tests):
        trajectory, total_cost, success = gfn.generate_solution(env, greedy=True)
        if success:
            gfn_costs.append(total_cost)
            gfn_successes += 1
    
    # Print comparison
    print("\n" + "="*60)
    print("COMPARISON RESULTS:")
    print("="*60)
    print(f"Random Policy:")
    print(f"  Success Rate: {random_successes}/{num_tests} ({random_successes/num_tests*100:.1f}%)")
    if len(random_costs) > 0:
        print(f"  Average Cost: {np.mean(random_costs):.4f} ± {np.std(random_costs):.4f}")
        print(f"  Best Cost:    {np.min(random_costs):.4f}")
    
    print(f"\nGFlowNet Policy:")
    print(f"  Success Rate: {gfn_successes}/{num_tests} ({gfn_successes/num_tests*100:.1f}%)")
    if len(gfn_costs) > 0:
        print(f"  Average Cost: {np.mean(gfn_costs):.4f} ± {np.std(gfn_costs):.4f}")
        print(f"  Best Cost:    {np.min(gfn_costs):.4f}")
    
    if len(gfn_costs) > 0 and len(random_costs) > 0:
        improvement = (np.mean(random_costs) - np.mean(gfn_costs)) / np.mean(random_costs) * 100
        print(f"\nCost Improvement: {improvement:.1f}%")


# Example usage
if __name__ == "__main__":
    # Create example graph
    n = 6
    
    # Example adjacency matrix
    A = np.array([
        [0, 1, 1, 0, 0, 0],
        [1, 0, 1, 1, 0, 0],
        [1, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 1],
        [0, 0, 1, 1, 0, 1],
        [0, 0, 0, 1, 1, 0]
    ])
    
    T = np.array([-1, -1, -1, -1, -1, -1])
    
    # Train and compare
    compare_policies(n, A, T, num_tests=50)