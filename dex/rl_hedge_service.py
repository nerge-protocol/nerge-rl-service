# ============================================================================
# FILE: rl_hedge_service.py
# ============================================================================

import asyncio
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import websockets
from web3 import Web3
import aiohttp
import redis
from dataclasses import dataclass
from enum import Enum
import hashlib
from scipy import stats
import logging
import os
import uuid

# Sui SDK
from sui_python_sdk import SuiClient, SuiTransaction

# ===================== CONFIGURATION =====================

@dataclass
class Config:
    # Sui connection
    SUI_RPC_URL: str = "https://fullnode.mainnet.sui.io:443"
    SUI_WS_URL: str = "wss://fullnode.mainnet.sui.io:443"
    PRIVATE_KEY: str = ""  # Load from env
    
    # Protocol addresses
    DEX_PACKAGE_ID: str = ""
    PROTECTION_MODULE_ID: str = ""
    
    # Exchange APIs
    DERIBIT_API_KEY: str = ""
    DERIBIT_SECRET: str = ""
    DYDX_API_KEY: str = ""
    
    # RL Model
    MODEL_PATH: str = "models/hedge_rl_model.pt"
    TRAINING_DATA_PATH: str = "data/training/"
    
    # Redis for state management
    REDIS_URL: str = "redis://localhost:6379"
    
    # Trading parameters
    MIN_HEDGE_AMOUNT_USD: float = 1000.0
    MAX_POSITION_SIZE_USD: float = 1000000.0
    HEDGE_THRESHOLD_SIGMA: float = 0.5  # Hedge when exposure > 0.5σ
    REBALANCE_INTERVAL_SEC: int = 300  # 5 minutes
    
    # Risk limits
    MAX_DELTA_EXPOSURE_ETH: float = 1000.0
    MAX_VEGA_EXPOSURE_USD: float = 100000.0
    MAX_DRAWDOWN_PCT: float = 0.10  # 10% max drawdown
    
    # Fees & Slippage
    SLIPPAGE_TOLERANCE: float = 0.002  # 0.2%
    MAX_FEE_RATE: float = 0.001  # 0.1%
    
    # Shutdown
    CLOSE_ON_SHUTDOWN: bool = False

# ===================== DATA MODELS =====================

class Asset(Enum):
    ETH = "ETH"
    BTC = "BTC"
    SOL = "SOL"
    SUI = "SUI"

class HedgeInstrument(Enum):
    PERPETUAL = "perpetual"
    OPTION = "option"
    SPOT = "spot"
    FUTURE = "future"

@dataclass
class PositionExposure:
    """Exposure from a single protected LP position"""
    position_id: str
    asset: Asset
    delta: float  # ETH equivalent exposure
    gamma: float  # Convexity exposure
    vega: float   # Volatility exposure
    theta: float  # Time decay exposure
    notional_usd: float
    health_factor: float  # 0.0 to 1.0
    
@dataclass  
class AggregateExposure:
    """Aggregate protocol exposure"""
    total_delta: float
    total_gamma: float
    total_vega: float
    total_notional: float
    by_asset: Dict[Asset, float]
    concentration_risk: float
    timestamp: datetime
    
@dataclass
class HedgeDecision:
    """RL decision for hedging"""
    instrument: HedgeInstrument
    asset: Asset
    amount: float  # Notional USD
    direction: str  # "long" or "short"
    price_limit: Optional[float]
    urgency: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    expected_cost: float
    expected_benefit: float
    
@dataclass
class MarketData:
    """Current market conditions"""
    timestamp: datetime
    prices: Dict[Asset, float]
    volatilities: Dict[Asset, float]  # Annualized
    funding_rates: Dict[str, float]  # Perp funding rates
    option_skew: Dict[Asset, float]  # Volatility skew
    liquidity_depth: Dict[Asset, float]  # $ available at 1% slippage
    market_regime: str  # "normal", "volatile", "crash"
    
@dataclass
class HedgePosition:
    """Active hedge position"""
    position_id: str
    instrument: HedgeInstrument
    asset: Asset
    notional_usd: float
    entry_price: float
    current_price: float
    delta: float
    pnl_usd: float
    created_at: datetime
    expiry: Optional[datetime]

# ===================== RL AGENT ARCHITECTURE =====================

class HedgePolicyNetwork(nn.Module):
    """Deep RL policy network for hedge decisions"""
    
    def __init__(self, state_dim=25, action_dim=8, hidden_dims=[256, 128, 64]):
        super().__init__()
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.LayerNorm(hidden_dims[2]),
            nn.ReLU(),
        )
        
        # Policy head (action probabilities)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dims[2], action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Value head (state value estimate)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dims[2], 1)
        )
        
        # Action parameter head (continuous parameters)
        self.action_param_head = nn.Sequential(
            nn.Linear(hidden_dims[2], 4),  # [amount, urgency, limit_price, duration]
            nn.Sigmoid()  # Output in [0, 1]
        )
        
    def forward(self, state):
        features = self.state_encoder(state)
        
        action_probs = self.policy_head(features)
        state_value = self.value_head(features)
        action_params = self.action_param_head(features)
        
        return action_probs, state_value, action_params

class ReplayBuffer:
    """Experience replay buffer for RL"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        """Push experience to buffer"""
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int):
        """Sample batch from buffer"""
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)

class HedgeRLAgent:
    """RL agent for optimal hedging"""
    
    def __init__(self, config: Config, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.config = config
        self.device = device
        
        # Networks
        self.policy_net = HedgePolicyNetwork().to(device)
        self.target_net = HedgePolicyNetwork().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.policy_net.parameters(),
            lr=1e-4,
            weight_decay=1e-5
        )
        
        # Experience replay
        self.replay_buffer = ReplayBuffer(capacity=100000)
        
        # Training state
        self.total_steps = 0
        self.epsilon = 0.1  # Exploration rate
        self.gamma = 0.99   # Discount factor
        self.tau = 0.005    # Soft update factor
        
        # Load pre-trained model if exists
        if os.path.exists(config.MODEL_PATH):
            self.load_model(config.MODEL_PATH)
            
        # Risk manager
        self.risk_manager = RiskManager(config)
        
        # Market data cache
        self.market_data = None
        self.exposure = None
        
    def encode_state(self, exposure: AggregateExposure, market_data: MarketData) -> torch.Tensor:
        """Encode state for RL agent"""
        
        state_vector = []
        
        # 1. Exposure features (10 dim)
        state_vector.append(exposure.total_delta / 1000)  # Normalize by 1000 ETH
        state_vector.append(exposure.total_gamma / 100)
        state_vector.append(exposure.total_vega / 100000)  # Normalize by $100k
        state_vector.append(exposure.total_notional / 10000000)  # $10M scale
        
        # Asset concentration
        for asset in [Asset.ETH, Asset.BTC, Asset.SOL, Asset.SUI]:
            state_vector.append(exposure.by_asset.get(asset, 0) / exposure.total_notional if exposure.total_notional > 0 else 0)
        
        state_vector.append(exposure.concentration_risk)
        
        # 2. Market features (10 dim)
        state_vector.append(market_data.prices.get(Asset.ETH, 0) / 5000)  # Normalize
        state_vector.append(market_data.volatilities.get(Asset.ETH, 0) / 2.0)  # 200% scale
        
        # Volatility regime encoding
        regime_map = {"normal": 0.0, "volatile": 0.5, "crash": 1.0}
        state_vector.append(regime_map.get(market_data.market_regime, 0.0))
        
        # Funding rates
        state_vector.append(market_data.funding_rates.get("ETH-PERP", 0) * 365)  # Annualized
        
        # Liquidity
        state_vector.append(min(market_data.liquidity_depth.get(Asset.ETH, 0) / 10000000, 1.0))
        
        # Option skew
        state_vector.append(market_data.option_skew.get(Asset.ETH, 0) / 0.2)  # 20% skew scale
        
        # Time features (5 dim)
        hour = market_data.timestamp.hour
        state_vector.append(np.sin(2 * np.pi * hour / 24))
        state_vector.append(np.cos(2 * np.pi * hour / 24))
        
        day_of_week = market_data.timestamp.weekday()
        state_vector.append(day_of_week / 7)
        
        # Recent PnL
        recent_pnl = self.get_recent_pnl(window_hours=24)
        state_vector.append(np.tanh(recent_pnl / 100000))  # Scale by $100k
        
        # Reserve ratio
        reserve_ratio = self.get_reserve_ratio()
        state_vector.append(reserve_ratio)
        
        # Convert to tensor
        state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
        
        return state_tensor
    
    def select_action(self, state: torch.Tensor, explore: bool = True) -> Dict:
        """Select hedging action using RL policy"""
        
        with torch.no_grad():
            action_probs, state_value, action_params = self.policy_net(state)
            
        if explore and np.random.random() < self.epsilon:
            # Random exploration
            action_idx = np.random.randint(action_probs.shape[1])
        else:
            # Greedy action
            action_idx = torch.argmax(action_probs).item()
        
        # Decode action
        action = self.decode_action(action_idx, action_params[0].cpu().numpy())
        
        return {
            "action_idx": action_idx,
            "action": action,
            "probs": action_probs[0].cpu().numpy(),
            "value": state_value.item(),
            "params": action_params[0].cpu().numpy()
        }
    
    def decode_action(self, action_idx: int, params: np.ndarray) -> HedgeDecision:
        """Decode RL action index to hedge decision"""
        
        # Action mapping
        actions = [
            (HedgeInstrument.PERPETUAL, "short"),
            (HedgeInstrument.PERPETUAL, "long"),
            (HedgeInstrument.OPTION, "buy_put"),
            (HedgeInstrument.OPTION, "buy_call"),
            (HedgeInstrument.OPTION, "sell_put"),
            (HedgeInstrument.OPTION, "sell_call"),
            (HedgeInstrument.SPOT, "sell"),
            (HedgeInstrument.SPOT, "buy"),
        ]
        
        instrument, direction = actions[action_idx]
        
        # Decode parameters
        amount_pct = params[0]  # 0 to 1
        urgency = params[1]
        price_limit_pct = params[2] * 0.1  # 0 to 10%
        duration_days = 1 + params[3] * 30  # 1 to 31 days
        
        # Calculate actual amounts based on exposure
        if instrument == HedgeInstrument.PERPETUAL:
            # Hedge delta exposure
            target_amount = abs(self.exposure.total_delta) * amount_pct if self.exposure else 0
            asset = Asset.ETH  # Primary hedge asset
        elif instrument == HedgeInstrument.OPTION:
            # Hedge gamma/vega exposure
            target_amount = abs(self.exposure.total_vega) * amount_pct if self.exposure else 0
            asset = Asset.ETH
        else:
            # Spot hedge
            target_amount = self.exposure.total_notional * amount_pct * 0.1 if self.exposure else 0
            asset = Asset.ETH
        
        # Apply risk limits
        target_amount = self.risk_manager.apply_risk_limits(
            instrument, asset, target_amount, direction
        )
        
        # Calculate price limit
        current_price = self.market_data.prices.get(asset, 0) if self.market_data else 0
        if direction in ["short", "sell", "buy_put", "sell_call"]:
            price_limit = current_price * (1 + price_limit_pct)
        else:
            price_limit = current_price * (1 - price_limit_pct)
        
        # Estimate costs
        expected_cost = self.estimate_transaction_cost(
            instrument, asset, target_amount, urgency
        )
        
        # Estimate benefit (reduction in VaR)
        expected_benefit = self.estimate_risk_reduction(
            instrument, asset, target_amount, direction
        )
        
        return HedgeDecision(
            instrument=instrument,
            asset=asset,
            amount=target_amount,
            direction=direction,
            price_limit=price_limit if price_limit > 0 else None,
            urgency=urgency,
            confidence=self.calculate_confidence(state),
            expected_cost=expected_cost,
            expected_benefit=expected_benefit
        )
    
    def update_policy(self, batch_size: int = 64) -> float:
        """Update RL policy using experience replay"""
        
        if len(self.replay_buffer) < batch_size:
            return 0.0
        
        # Sample batch
        batch = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = batch
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        action_probs, state_values, _ = self.policy_net(states)
        q_values = state_values.squeeze()
        
        # Next Q values (target network)
        with torch.no_grad():
            _, next_state_values, _ = self.target_net(next_states)
            next_q_values = next_state_values.squeeze()
        
        # Target Q values
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Calculate loss
        value_loss = nn.functional.mse_loss(q_values, target_q_values)
        
        # Policy gradient loss
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze())
        policy_loss = -(log_probs * (target_q_values - q_values).detach()).mean()
        
        # Total loss
        total_loss = value_loss + policy_loss + 0.01 * self.entropy_regularization(action_probs)
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Soft update target network
        self.soft_update_target_network()
        
        self.total_steps += 1
        
        return total_loss.item()
    
    def soft_update_target_network(self):
        """Soft update target network parameters"""
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def entropy_regularization(self, probs: torch.Tensor) -> torch.Tensor:
        """Entropy regularization for exploration"""
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean()
        return entropy
    
    def calculate_confidence(self, state: torch.Tensor) -> float:
        """Calculate confidence in current state"""
        
        # Confidence based on state familiarity
        # In production: use uncertainty estimation or ensemble methods
        
        # Simplified: confidence based on recent performance
        recent_pnl = self.get_recent_pnl(window_hours=24)
        win_rate = self.get_recent_win_rate(window_hours=24)
        
        confidence = 0.5 + 0.3 * np.tanh(recent_pnl / 50000) + 0.2 * (win_rate - 0.5)
        
        return np.clip(confidence, 0.1, 0.95)
    
    def estimate_transaction_cost(self, instrument: HedgeInstrument, 
                                 asset: Asset, amount: float, urgency: float) -> float:
        """Estimate transaction costs for hedge"""
        
        base_fees = {
            HedgeInstrument.PERPETUAL: 0.0005,  # 0.05%
            HedgeInstrument.OPTION: 0.02,       # 2% for options
            HedgeInstrument.SPOT: 0.001,        # 0.1%
        }
        
        # Slippage increases with urgency and amount
        slippage_factor = urgency * 0.01  # Up to 1% slippage
        
        # Market impact
        liquidity = self.market_data.liquidity_depth.get(asset, 1000000)
        impact_factor = min(amount / liquidity, 0.05)  # Up to 5% impact
        
        total_cost = base_fees[instrument] + slippage_factor + impact_factor
        
        return total_cost * amount
    
    def estimate_risk_reduction(self, instrument: HedgeInstrument,
                               asset: Asset, amount: float, direction: str) -> float:
        """Estimate risk reduction from hedge"""
        
        if not self.exposure:
            return 0.0
        
        # Calculate VaR reduction
        current_var = self.calculate_var(self.exposure, confidence=0.99)
        
        # Simulate new exposure after hedge
        new_exposure = self.simulate_hedge_impact(
            self.exposure, instrument, asset, amount, direction
        )
        new_var = self.calculate_var(new_exposure, confidence=0.99)
        
        var_reduction = current_var - new_var
        
        # Convert to USD benefit (simplified)
        benefit = var_reduction * 0.3  # Assume 30% of VaR reduction is actual benefit
        
        return max(benefit, 0)
    
    def calculate_var(self, exposure: AggregateExposure, confidence: float = 0.99) -> float:
        """Calculate Value at Risk for exposure"""
        
        # Simplified VaR calculation
        # In production: use historical simulation or Monte Carlo
        
        total_notional = exposure.total_notional
        concentration = exposure.concentration_risk
        
        # Base volatility assumption
        base_vol = self.market_data.volatilities.get(Asset.ETH, 1.0) if self.market_data else 1.0
        
        # VaR = Z * σ * √t * notional * concentration
        z_score = stats.norm.ppf(confidence)
        var = z_score * base_vol * np.sqrt(1/365) * total_notional * (1 + concentration)
        
        return var
    
    def simulate_hedge_impact(self, exposure: AggregateExposure,
                             instrument: HedgeInstrument, asset: Asset,
                             amount: float, direction: str) -> AggregateExposure:
        """Simulate exposure after hedge"""
        
        # Simplified simulation
        # In production: use proper Greeks calculation
        
        new_exposure = AggregateExposure(
            total_delta=exposure.total_delta,
            total_gamma=exposure.total_gamma,
            total_vega=exposure.total_vega,
            total_notional=exposure.total_notional,
            by_asset=exposure.by_asset.copy(),
            concentration_risk=exposure.concentration_risk,
            timestamp=exposure.timestamp
        )
        
        # Adjust delta for perp/short hedge
        if instrument == HedgeInstrument.PERPETUAL:
            if direction == "short":
                new_exposure.total_delta -= amount / self.market_data.prices.get(asset, 1)
            else:
                new_exposure.total_delta += amount / self.market_data.prices.get(asset, 1)
        
        # Adjust vega for option hedge
        elif instrument == HedgeInstrument.OPTION:
            vega_per_option = amount * 0.01  # Simplified
            if "buy" in direction:
                new_exposure.total_vega += vega_per_option
            else:
                new_exposure.total_vega -= vega_per_option
        
        return new_exposure
    
    def get_recent_pnl(self, window_hours: int = 24) -> float:
        """Get recent PnL from database"""
        # In production: query from database
        return 0.0
    
    def get_recent_win_rate(self, window_hours: int = 24) -> float:
        """Get recent win rate"""
        # In production: query from database
        return 0.5
    
    def get_reserve_ratio(self) -> float:
        """Get protocol reserve ratio"""
        # In production: query from blockchain
        return 1.0
    
    def save_experience(self, state, action_idx, reward, next_state, done):
        """Save experience to replay buffer"""
        self.replay_buffer.push(state, action_idx, reward, next_state, done)
    
    def save_model(self, path: str):
        """Save RL model"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
            'epsilon': self.epsilon,
        }, path)
    
    def load_model(self, path: str):
        """Load RL model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_steps = checkpoint['total_steps']
        self.epsilon = checkpoint['epsilon']

# ===================== RISK MANAGER =====================

class RiskManager:
    """Manages risk limits and constraints"""
    
    def __init__(self, config: Config):
        self.config = config
        self.active_positions = []
        self.position_history = []
        
        # Risk limits
        self.limits = {
            'max_delta_eth': config.MAX_DELTA_EXPOSURE_ETH,
            'max_vega_usd': config.MAX_VEGA_EXPOSURE_USD,
            'max_position_usd': config.MAX_POSITION_SIZE_USD,
            'max_drawdown_pct': config.MAX_DRAWDOWN_PCT,
            'max_concentration': 0.5,  # 50% in one instrument
        }
        
        # Current exposures
        self.current_exposures = {
            'delta_eth': 0.0,
            'vega_usd': 0.0,
            'total_notional': 0.0,
            'concentration': {}
        }
        
    def apply_risk_limits(self, instrument: HedgeInstrument, 
                         asset: Asset, amount: float, 
                         direction: str) -> float:
        """Apply risk limits to hedge amount"""
        
        # Check position size limit
        max_position = self.limits['max_position_usd']
        amount = min(amount, max_position)
        
        # Check concentration limit
        instrument_key = f"{instrument.value}_{asset.value}"
        current_instrument_exposure = self.current_exposures['concentration'].get(instrument_key, 0)
        max_instrument_exposure = self.limits['max_concentration'] * self.current_exposures['total_notional']
        
        if current_instrument_exposure + amount > max_instrument_exposure:
            amount = max(0, max_instrument_exposure - current_instrument_exposure)
        
        # Check delta limit for perp positions
        if instrument == HedgeInstrument.PERPETUAL:
            delta_change = amount / (self.get_market_price(asset) or 1)
            if direction == "short":
                delta_change = -delta_change
            
            new_delta = self.current_exposures['delta_eth'] + delta_change
            
            if abs(new_delta) > self.limits['max_delta_eth']:
                # Scale back
                max_delta_change = self.limits['max_delta_eth'] - abs(self.current_exposures['delta_eth'])
                max_amount = max_delta_change * (self.get_market_price(asset) or 1)
                amount = min(amount, max_amount)
        
        # Check vega limit for option positions
        elif instrument == HedgeInstrument.OPTION:
            # Simplified: assume 1% vega per $100 of option
            vega_change = amount * 0.01
            new_vega = self.current_exposures['vega_usd'] + vega_change
            
            if abs(new_vega) > self.limits['max_vega_usd']:
                max_vega_change = self.limits['max_vega_usd'] - abs(self.current_exposures['vega_usd'])
                max_amount = max_vega_change * 100  # $100 per 1% vega
                amount = min(amount, max_amount)
        
        # Check drawdown limit
        if self.check_drawdown_limit(amount, direction):
            amount = 0
        
        return amount
    
    def check_drawdown_limit(self, amount: float, direction: str) -> bool:
        """Check if trade would violate drawdown limit"""
        
        # Simplified drawdown check
        # In production: use proper VaR/CVaR
        
        current_pnl = self.calculate_total_pnl()
        max_loss = -self.limits['max_drawdown_pct'] * self.current_exposures['total_notional']
        
        # Estimate worst-case loss for this trade
        worst_case_loss = self.estimate_worst_case_loss(amount, direction)
        
        if current_pnl + worst_case_loss < max_loss:
            return True  # Would violate drawdown limit
        
        return False
    
    def calculate_total_pnl(self) -> float:
        """Calculate total PnL of all positions"""
        total_pnl = 0.0
        for position in self.active_positions:
            total_pnl += position.pnl_usd
        return total_pnl
    
    def estimate_worst_case_loss(self, amount: float, direction: str) -> float:
        """Estimate worst-case loss for a trade"""
        
        # Simplified: 10% of notional for directional trades
        # 5% for hedging trades
        
        if direction in ["long", "short", "buy", "sell"]:
            return -amount * 0.10
        else:
            return -amount * 0.05
    
    def get_market_price(self, asset: Asset) -> Optional[float]:
        """Get current market price"""
        # In production: query from price feed
        return None
    
    def update_exposures(self, position: HedgePosition):
        """Update risk exposures after trade"""
        
        # Update delta
        if position.instrument == HedgeInstrument.PERPETUAL:
            delta = position.notional_usd / (position.entry_price or 1)
            if position.direction == "short":
                delta = -delta
            self.current_exposures['delta_eth'] += delta
        
        # Update vega
        elif position.instrument == HedgeInstrument.OPTION:
            vega = position.notional_usd * 0.01  # Simplified
            self.current_exposures['vega_usd'] += vega
        
        # Update total notional
        self.current_exposures['total_notional'] += position.notional_usd
        
        # Update concentration
        instrument_key = f"{position.instrument.value}_{position.asset.value}"
        self.current_exposures['concentration'][instrument_key] = \
            self.current_exposures['concentration'].get(instrument_key, 0) + position.notional_usd

# ===================== SUI BLOCKCHAIN INTEGRATION =====================

class SuiHedgeManager:
    """Manages Sui blockchain interactions for hedging"""
    
    def __init__(self, config: Config):
        self.config = config
        self.sui_client = SuiClient(config.SUI_RPC_URL)
        
        # Load key
        self.signer_address = self.sui_client.get_address(config.PRIVATE_KEY)
        
        # Cache
        self.protection_state = None
        self.last_state_update = None
        
    async def get_protection_state(self) -> Dict:
        """Get current protection module state from Sui"""
        
        if (self.protection_state and 
            datetime.now() - self.last_state_update < timedelta(seconds=30)):
            return self.protection_state
        
        try:
            # Call view function on protection module
            result = await self.sui_client.call_move_function_async(
                package_id=self.config.PROTECTION_MODULE_ID,
                module="protection",
                function="get_protection_state",
                type_arguments=["0x2::sui::SUI", "0x2::sui::SUI"],  # X, Y tokens
                arguments=[self.config.PROTECTION_MODULE_ID],
                gas_budget=10000000
            )
            
            # Parse result
            self.protection_state = {
                'total_protected_value': result.get('total_protected_value', 0),
                'total_reserves': result.get('total_reserves', 0),
                'reserve_ratio': result.get('reserve_ratio', 0),
                'total_payouts': result.get('total_payouts', 0),
                'total_fees_collected': result.get('total_fees_collected', 0),
            }
            
            self.last_state_update = datetime.now()
            
            return self.protection_state
            
        except Exception as e:
            logging.error(f"Error fetching protection state: {e}")
            return {}
    
    async def get_exposure_from_positions(self) -> List[PositionExposure]:
        """Get exposure from all protected positions"""
        
        exposures = []
        
        try:
            # Get all positions from protection module
            # This would require iterating through positions in Move
            # Simplified for example
            
            # In production: use dynamic fields or events to get positions
            
            return exposures
            
        except Exception as e:
            logging.error(f"Error fetching positions: {e}")
            return []
    
    async def submit_hedge_decision(self, decision: HedgeDecision) -> bool:
        """Submit hedge decision to blockchain (for recording)"""
        
        try:
            # Create Move transaction to record hedge
            tx = SuiTransaction()
            
            # Call protection module to record hedge
            tx.move_call(
                target=f"{self.config.PROTECTION_MODULE_ID}::protection::record_hedge",
                arguments=[
                    tx.object(self.config.PROTECTION_MODULE_ID),
                    tx.pure.u64(int(decision.amount * 1e8)),  # Scaled
                    tx.pure.u8(0 if decision.direction == "long" else 1),
                    tx.pure.u64(int(datetime.now().timestamp() * 1000)),
                ],
                type_arguments=["0x2::sui::SUI", "0x2::sui::SUI"]
            )
            
            # Execute transaction
            result = await self.sui_client.execute_transaction_async(
                tx, self.config.PRIVATE_KEY
            )
            
            return result['effects']['status']['status'] == 'success'
            
        except Exception as e:
            logging.error(f"Error submitting hedge decision: {e}")
            return False

# ===================== MAIN HEDGE SERVICE =====================

class HedgeService:
    """Main hedge service orchestrator"""
    
    def __init__(self, config: Config):
        self.config = config
        self.running = False
        
        # Initialize components
        self.rl_agent = HedgeRLAgent(config)
        self.risk_manager = RiskManager(config)
        self.sui_manager = SuiHedgeManager(config)
        
        # Exchange clients (Mocked for now)
        # self.deribit = DeribitClient(config.DERIBIT_API_KEY, config.DERIBIT_SECRET)
        # self.dydx = DyDxClient(config.DYDX_API_KEY, config.DYDX_SECRET)
        
        # Market data
        self.market_data = None
        self.exposure = None
        
        # Active positions
        self.active_positions = []
        self.position_history = []
        
        # Redis for state persistence
        self.redis = redis.Redis.from_url(config.REDIS_URL)
        
        # Performance tracking
        self.performance_metrics = {
            'total_pnl': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'total_fees': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
        }
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('hedge_service.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    async def start(self):
        """Start hedge service"""
        self.logger.info("Starting Hedge Service...")
        self.running = True
        
        # Authenticate with exchanges
        # await self.deribit.authenticate()
        
        # Load existing positions
        await self.load_positions()
        
        # Start main loop
        asyncio.create_task(self.main_loop())
        
        # Start websocket connections
        # asyncio.create_task(self.websocket_listener())
        
        self.logger.info("Hedge Service started")
    
    async def stop(self):
        """Stop hedge service"""
        self.logger.info("Stopping Hedge Service...")
        self.running = False
        
        # Close positions if needed
        if self.config.CLOSE_ON_SHUTDOWN:
            await self.close_all_positions()
        
        # Save state
        await self.save_state()
        
        self.logger.info("Hedge Service stopped")
    
    async def main_loop(self):
        """Main service loop"""
        
        while self.running:
            try:
                # 1. Update market data
                await self.update_market_data()
                
                # 2. Update exposure from blockchain
                await self.update_exposure()
                
                # 3. Check if hedging is needed
                if self.should_hedge():
                    # 4. Get RL decision
                    decision = await self.get_hedge_decision()
                    
                    # 5. Execute hedge if decision is confident
                    if decision.confidence > 0.6:  # 60% confidence threshold
                        await self.execute_hedge(decision)
                
                # 6. Update RL agent with results
                await self.update_rl_agent()
                
                # 7. Monitor existing positions
                await self.monitor_positions()
                
                # 8. Save state periodically
                if int(datetime.now().timestamp()) % 300 == 0:  # Every 5 minutes
                    await self.save_state()
                
                # Wait for next cycle
                await asyncio.sleep(self.config.REBALANCE_INTERVAL_SEC)
                
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}", exc_info=True)
                await asyncio.sleep(30)  # Wait before retry
    
    async def update_market_data(self):
        """Update market data from exchanges"""
        
        try:
            # Mock data for now
            eth_price = 2000.0
            eth_vol = 0.8
            eth_funding = 0.0001
            liquidity = 10000000.0
            regime = "normal"
            
            self.market_data = MarketData(
                timestamp=datetime.now(),
                prices={Asset.ETH: eth_price},
                volatilities={Asset.ETH: eth_vol},
                funding_rates={"ETH-PERP": eth_funding},
                option_skew={Asset.ETH: 0.05},
                liquidity_depth={Asset.ETH: liquidity},
                market_regime=regime
            )
            
            # Cache in Redis
            self.redis.setex(
                "market_data",
                60,  # 1 minute TTL
                json.dumps(self.market_data.__dict__, default=str)
            )
            
        except Exception as e:
            self.logger.error(f"Error updating market data: {e}")
    
    async def update_exposure(self):
        """Update exposure from blockchain"""
        
        try:
            # Get protection state
            protection_state = await self.sui_manager.get_protection_state()
            
            # Get individual positions
            positions = await self.sui_manager.get_exposure_from_positions()
            
            # Calculate aggregate exposure
            total_delta = sum(p.delta for p in positions)
            total_gamma = sum(p.gamma for p in positions)
            total_vega = sum(p.vega for p in positions)
            total_notional = sum(p.notional_usd for p in positions)
            
            # Calculate by asset
            by_asset = {}
            for asset in Asset:
                asset_positions = [p for p in positions if p.asset == asset]
                by_asset[asset] = sum(p.notional_usd for p in asset_positions)
            
            # Calculate concentration risk
            concentration = 0.0 # self.calculate_concentration_risk(by_asset, total_notional)
            
            self.exposure = AggregateExposure(
                total_delta=total_delta,
                total_gamma=total_gamma,
                total_vega=total_vega,
                total_notional=total_notional,
                by_asset=by_asset,
                concentration_risk=concentration,
                timestamp=datetime.now()
            )
            
            # Log exposure
            self.logger.info(f"Exposure updated: Delta={total_delta:.2f} ETH, "
                           f"Vega=${total_vega:,.0f}, Notional=${total_notional:,.0f}")
            
        except Exception as e:
            self.logger.error(f"Error updating exposure: {e}")
    
    def should_hedge(self) -> bool:
        """Determine if hedging is needed"""
        
        if not self.exposure or not self.market_data:
            return False
        
        # Check delta exposure threshold
        delta_threshold = self.config.HEDGE_THRESHOLD_SIGMA * 100  # 100 ETH per sigma
        if abs(self.exposure.total_delta) > delta_threshold:
            self.logger.info(f"Hedge needed: Delta exposure {self.exposure.total_delta:.2f} ETH "
                           f"> threshold {delta_threshold} ETH")
            return True
        
        # Check vega exposure threshold
        vega_threshold = 50000  # $50k vega
        if abs(self.exposure.total_vega) > vega_threshold:
            self.logger.info(f"Hedge needed: Vega exposure ${self.exposure.total_vega:,.0f} "
                           f"> threshold ${vega_threshold:,.0f}")
            return True
        
        return False
    
    async def get_hedge_decision(self) -> HedgeDecision:
        """Get hedge decision from RL agent"""
        
        if not self.exposure or not self.market_data:
            return None
        
        # Encode state for RL
        state_tensor = self.rl_agent.encode_state(self.exposure, self.market_data)
        
        # Get RL decision
        rl_output = self.rl_agent.select_action(state_tensor, explore=False)
        
        # Decode to hedge decision
        decision = rl_output['action']
        
        # Log decision
        self.logger.info(f"RL Decision: {decision.instrument.value} {decision.direction} "
                       f"${decision.amount:,.0f} {decision.asset.value} "
                       f"(Confidence: {decision.confidence:.1%})")
        
        return decision
    
    async def execute_hedge(self, decision: HedgeDecision):
        """Execute hedge on exchange"""
        
        try:
            self.logger.info(f"Executing hedge: {decision.instrument.value} "
                           f"{decision.direction} ${decision.amount:,.0f}")
            
            # Record start time for performance tracking
            start_time = datetime.now()
            
            # Execute based on instrument (Mocked)
            # if decision.instrument == HedgeInstrument.PERPETUAL:
            #     await self.execute_perp_hedge(decision)
            # elif decision.instrument == HedgeInstrument.OPTION:
            #     await self.execute_option_hedge(decision)
            # elif decision.instrument == HedgeInstrument.SPOT:
            #     await self.execute_spot_hedge(decision)
            
            # Record to blockchain
            await self.sui_manager.submit_hedge_decision(decision)
            
            # Update performance metrics
            # await self.update_performance_metrics(decision, start_time)
            
            # Save experience for RL learning
            # await self.record_experience(decision)
            
        except Exception as e:
            self.logger.error(f"Error executing hedge: {e}", exc_info=True)
            
    async def monitor_positions(self):
        pass

    async def update_rl_agent(self):
        pass

    async def load_positions(self):
        pass

    async def save_state(self):
        pass
        
    async def close_all_positions(self):
        pass

if __name__ == "__main__":
    config = Config()
    service = HedgeService(config)
    asyncio.run(service.start())
