import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import numpy as np

# ============================================================================
# 4. DEEP Q-NETWORK (DQN) ARCHITECTURE
# ============================================================================

class QNetwork(nn.Module):
    """Deep Q-Network for interest rate policy"""
    
    def __init__(self, state_dim=15, action_dim=4, hidden_dims=[64, 32]):
        super().__init__()
        
        # Input: state vector
        # Output: Q-values for each action
        
        layers = []
        prev_dim = state_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        
        # Output layer: Q-values for each action dimension
        # We have 4 continuous action dimensions
        self.network = nn.Sequential(*layers)
        
        # Separate heads for each action dimension (allowing different scales)
        self.q_heads = nn.ModuleList([
            nn.Linear(prev_dim, 1) for _ in range(action_dim)
        ])
        
        # Value stream (for Dueling DQN architecture)
        self.value_stream = nn.Sequential(
            nn.Linear(prev_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        # Advantage stream (for Dueling DQN)
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_dim, 16),
            nn.ReLU(),
            nn.Linear(16, action_dim)
        )
        
        self.use_dueling = True  # Dueling architecture typically better
        
    def forward(self, state):
        """Forward pass: state → Q-values for each action dimension"""
        
        features = self.network(state)
        
        if self.use_dueling:
            # Dueling DQN: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
            value = self.value_stream(features)
            advantages = self.advantage_stream(features)
            
            # Combine: Q = V + A - mean(A)
            q_values = value + advantages - advantages.mean(dim=1, keepdim=True)
            
            return q_values
        else:
            # Standard DQN: direct Q-values
            q_values = torch.cat([
                head(features) for head in self.q_heads
            ], dim=1)
            
            return q_values

class ReplayBuffer:
    """Experience replay buffer for stable training"""
    
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, experience):
        """Add experience tuple (s, a, r, s', done)"""
        self.buffer.append(experience)
        
    def sample(self, batch_size):
        """Sample random batch for training"""
        if len(self.buffer) < batch_size:
            return None
        
        batch = random.sample(self.buffer, batch_size)
        
        # Unzip batch
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """Deep Q-Learning agent for interest rate optimization"""
    
    def __init__(self, 
                 state_dim=15,
                 action_dim=4,
                 learning_rate=0.001,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay=0.995,
                 target_update_freq=100):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma  # Discount factor
        
        # ε-greedy exploration
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Networks
        self.policy_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target net is not trained directly
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Experience replay
        self.replay_buffer = ReplayBuffer(capacity=100000)
        
        # Training parameters
        self.batch_size = 64
        self.target_update_freq = target_update_freq
        self.train_step = 0
        
        # Action bounds
        self.action_bounds = torch.FloatTensor([
            [-0.02, 0.02],   # Δr₀
            [-0.05, 0.05],   # Δr₁
            [-0.10, 0.10],   # Δr₂
            [-0.10, 0.10]    # ΔU*
        ])
        
    def select_action(self, state, explore=True):
        """Select action using ε-greedy policy"""
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        if explore and random.random() < self.epsilon:
            # Random exploration
            action = torch.rand(self.action_dim) * 2 - 1  # [-1, 1]
            action = self._scale_action(action)
        else:
            # Greedy action
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                
                # For continuous action space, we output action directly
                # (using deterministic policy gradient approach)
                action = torch.tanh(q_values)  # Bound to [-1, 1]
                action = self._scale_action(action.squeeze(0))
        
        # Decay epsilon
        if explore:
            self.epsilon = max(self.epsilon_end, 
                              self.epsilon * self.epsilon_decay)
        
        return action.numpy()
    
    def _scale_action(self, action):
        """Scale action from [-1, 1] to actual parameter change ranges"""
        
        # action is in [-1, 1]
        scaled = torch.zeros_like(action)
        
        for i in range(self.action_dim):
            low, high = self.action_bounds[i]
            # Scale from [-1, 1] to [low, high]
            scaled[i] = low + (action[i] + 1) * (high - low) / 2
        
        return scaled
    
    def update(self):
        """Update policy network using experience replay"""
        
        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        if batch is None:
            return 0.0
        
        states, actions, rewards, next_states, dones = batch
        
        # Compute Q(s, a)
        current_q = self.policy_net(states)
        # For continuous actions, we need to evaluate at taken actions
        # Simplified: we output actions directly, so Q(s) is the action
        
        # Compute max_a' Q(s', a') using target network
        with torch.no_grad():
            next_q = self.target_net(next_states)
            max_next_q, _ = next_q.max(dim=1, keepdim=True)
            
            # Target: r + γ * max_a' Q(s', a') * (1 - done)
            target_q = rewards + self.gamma * max_next_q * (1 - dones)
        
        # Compute loss (Huber loss is more robust than MSE)
        loss = nn.functional.smooth_l1_loss(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Update target network periodically
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def save_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.replay_buffer.push((state, action, reward, next_state, done))
    
    def save(self, path):
        """Save agent state"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_step': self.train_step
        }, path)
    
    def load(self, path):
        """Load agent state"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.train_step = checkpoint['train_step']
