# rl_agent_service.py
# Production RL Agent for Adaptive LP Vault Management

import asyncio
import torch
import torch.nn as nn
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional
import logging

# Sui SDK imports (pseudo-code - adjust for actual SDK)
from pysui import SuiClient, TransactionBlock
from pysui.sui_crypto import SuiKeypair

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MarketState:
    """Market observation for RL agent"""
    current_price: float
    current_tick: int
    liquidity: float
    volume_24h: float
    volatility_7d: float
    price_change_1h: float
    price_change_24h: float
    bid_ask_spread: float
    
    # Historical features
    price_history: np.ndarray  # Last 20 prices
    volume_history: np.ndarray  # Last 20 volumes
    
    # Vault state
    total_tvl: float
    idle_capital_pct: float
    active_positions: int
    last_rebalance_ago_minutes: int


@dataclass
class PositionConfig:
    """Configuration for a single LP position"""
    tick_lower: int
    tick_upper: int
    capital_allocation_bps: int  # Basis points


@dataclass
class RebalanceAction:
    """RL agent's decision"""
    should_rebalance: bool
    positions: List[PositionConfig]
    expected_profit: float
    confidence: float


class AdaptiveLPModel(nn.Module):
    """
    Deep RL Model for LP Position Management
    
    Architecture:
    - LSTM for temporal patterns
    - Attention for feature importance
    - Actor-Critic for policy optimization
    """
    
    def __init__(
        self,
        state_dim: int = 50,
        hidden_dim: int = 256,
        max_positions: int = 5,
    ):
        super().__init__()
        
        self.max_positions = max_positions
        
        # Feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2),
        )
        
        # LSTM for temporal dependencies
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )
        
        # Actor: decides position configurations
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_positions * 3),  # 3 params per position
            nn.Tanh(),  # Output in [-1, 1]
        )
        
        # Critic: estimates value
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Rebalance decision
        self.rebalance_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # Probability of rebalancing
        )
    
    def forward(
        self,
        state: torch.Tensor,
        hidden: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple]:
        """
        Forward pass
        
        Returns:
            action: Position configuration
            value: State value estimate
            rebalance_prob: Probability to rebalance
            hidden: LSTM hidden state
        """
        # Extract features
        features = self.feature_net(state)
        
        # Add time dimension if needed
        if len(features.shape) == 2:
            features = features.unsqueeze(1)
        
        # LSTM processing
        lstm_out, hidden = self.lstm(features, hidden)
        lstm_out = lstm_out[:, -1, :]  # Take last timestep
        
        # Policy (actor)
        action = self.actor(lstm_out)
        
        # Value (critic)
        value = self.critic(lstm_out)
        
        # Rebalance decision
        rebalance_prob = self.rebalance_head(lstm_out)
        
        return action, value, rebalance_prob, hidden


class AdaptiveLPAgent:
    """
    Production RL Agent for managing adaptive LP vault
    """
    
    def __init__(
        self,
        vault_id: str,
        pool_id: str,
        package_id: str,
        registry_id: str,
        model_path: str,
        sui_client: SuiClient,
        keypair: SuiKeypair,
        config: dict,
    ):
        self.vault_id = vault_id
        self.pool_id = pool_id
        self.package_id = package_id
        self.registry_id = registry_id
        self.client = sui_client
        self.keypair = keypair
        self.config = config
        
        # Load trained model
        self.model = AdaptiveLPModel()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        # State history for LSTM
        self.state_history = deque(maxlen=100)
        self.lstm_hidden = None
        
        # Performance tracking
        self.rebalance_count = 0
        self.total_profit = 0.0
        self.successful_rebalances = 0
        
        logger.info(f"✅ RL Agent initialized for vault {vault_id}")
    
    async def run(self):
        """Main agent loop"""
        logger.info("🚀 Starting RL Agent...")
        
        while True:
            try:
                # 1. Observe market state
                state = await self.get_market_state()
                
                # 2. RL model decides action
                action = self.decide_action(state)
                
                # 3. Execute if profitable
                if action.should_rebalance:
                    logger.info(f"💡 Rebalance signal: confidence={action.confidence:.2%}")
                    await self.execute_rebalance(action)
                else:
                    logger.debug("⏸️  No rebalance needed")
                
                # 4. Wait before next observation
                await asyncio.sleep(self.config.get('check_interval_seconds', 60))
                
            except Exception as e:
                logger.error(f"❌ Error in agent loop: {e}", exc_info=True)
                await asyncio.sleep(10)
    
    async def get_market_state(self) -> MarketState:
        """Gather market data for RL model"""
        
        # Fetch on-chain data
        pool_state = await self.client.get_object(self.pool_id)
        vault_state = await self.client.get_object(self.vault_id)
        
        # Parse pool state
        current_price = self._parse_sqrt_price(pool_state['sqrt_price_x96'])
        current_tick = pool_state['tick']
        liquidity = pool_state['liquidity']
        
        # Fetch historical data (from indexer or cache)
        price_history = await self._get_price_history(lookback=20)
        volume_history = await self._get_volume_history(lookback=20)
        
        # Calculate derived metrics
        volume_24h = sum(volume_history[-24:])
        volatility_7d = np.std(price_history[-168:]) / np.mean(price_history[-168:])
        price_change_1h = (price_history[-1] - price_history[-60]) / price_history[-60]
        price_change_24h = (price_history[-1] - price_history[-1440]) / price_history[-1440]
        
        # Vault metrics
        vault_snapshot = await self._call_view_function(
            'get_vault_snapshot',
            [self.vault_id, self.pool_id]
        )
        
        total_tvl = vault_snapshot['total_value_0'] + vault_snapshot['total_value_1']
        idle_capital = vault_snapshot['idle_capital_0'] + vault_snapshot['idle_capital_1']
        idle_capital_pct = idle_capital / total_tvl if total_tvl > 0 else 0
        
        state = MarketState(
            current_price=current_price,
            current_tick=current_tick,
            liquidity=liquidity,
            volume_24h=volume_24h,
            volatility_7d=volatility_7d,
            price_change_1h=price_change_1h,
            price_change_24h=price_change_24h,
            bid_ask_spread=0.0,  # Calculate from order book
            price_history=np.array(price_history[-20:]),
            volume_history=np.array(volume_history[-20:]),
            total_tvl=total_tvl,
            idle_capital_pct=idle_capital_pct,
            active_positions=vault_snapshot['active_position_count'],
            last_rebalance_ago_minutes=self._minutes_since_last_rebalance(vault_snapshot),
        )
        
        self.state_history.append(state)
        return state
    
    def decide_action(self, state: MarketState) -> RebalanceAction:
        """Use RL model to decide action"""
        
        # Convert state to tensor
        state_vector = self._state_to_vector(state)
        state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)
        
        with torch.no_grad():
            # Forward pass
            action_raw, value, rebalance_prob, self.lstm_hidden = self.model(
                state_tensor,
                self.lstm_hidden,
            )
            
            rebalance_prob = rebalance_prob.item()
            
            # Decode action into position configs
            positions = self._decode_action(action_raw.squeeze().numpy(), state)
        
        # Estimate profit vs gas cost
        estimated_profit = self._estimate_profit(positions, state)
        gas_cost = self._estimate_gas_cost(len(positions))
        
        # Decision: rebalance if profit > cost with safety margin
        should_rebalance = (
            rebalance_prob > 0.6 and
            estimated_profit > gas_cost * 1.5 and
            self._time_since_last_rebalance() >= self.config['min_rebalance_interval_minutes']
        )
        
        return RebalanceAction(
            should_rebalance=should_rebalance,
            positions=positions,
            expected_profit=estimated_profit - gas_cost,
            confidence=rebalance_prob,
        )
    
    def _decode_action(
        self,
        action_raw: np.ndarray,
        state: MarketState,
    ) -> List[PositionConfig]:
        """Convert model output to position configurations"""
        
        # Action format: [num_positions, pos1_lower, pos1_upper, pos1_alloc, pos2_lower, ...]
        # Each value in [-1, 1], needs to be scaled
        
        # Number of positions (1-5)
        num_positions = int((action_raw[0] + 1) / 2 * 4) + 1  # Maps to 1-5
        num_positions = min(num_positions, self.config['max_positions'])
        
        positions = []
        current_tick = state.current_tick
        
        for i in range(num_positions):
            base_idx = 1 + i * 3
            
            # Decode tick offsets (relative to current tick)
            # Scale from [-1, 1] to [-500, +500] ticks
            lower_offset = int(action_raw[base_idx] * 500)
            upper_offset = int(action_raw[base_idx + 1] * 500)
            
            tick_lower = current_tick + lower_offset
            tick_upper = current_tick + upper_offset
            
            # Ensure valid range
            if tick_lower >= tick_upper:
                tick_upper = tick_lower + 60  # Minimum width
            
            # Round to tick spacing (60 in this case)
            tick_lower = (tick_lower // 60) * 60
            tick_upper = (tick_upper // 60) * 60
            
            # Decode allocation (0-100%)
            allocation_pct = (action_raw[base_idx + 2] + 1) / 2  # Map to [0, 1]
            allocation_bps = int(allocation_pct * 10000)
            
            positions.append(PositionConfig(
                tick_lower=tick_lower,
                tick_upper=tick_upper,
                capital_allocation_bps=allocation_bps,
            ))
        
        # Normalize allocations to sum to 100%
        total_alloc = sum(p.capital_allocation_bps for p in positions)
        if total_alloc > 0:
            for pos in positions:
                pos.capital_allocation_bps = int(
                    pos.capital_allocation_bps * 10000 / total_alloc
                )
        
        return positions
    
    async def execute_rebalance(self, action: RebalanceAction):
        """Submit rebalance transaction to chain"""
        
        try:
            logger.info(f"📤 Executing rebalance with {len(action.positions)} positions")
            
            # Build transaction
            tx = TransactionBlock()
            
            # Convert positions to Move format
            position_configs = []
            for pos in action.positions:
                position_configs.append({
                    'tick_lower': pos.tick_lower,
                    'tick_upper': pos.tick_upper,
                    'capital_allocation_bps': pos.capital_allocation_bps,
                })
            
            # Call rebalance function
            tx.move_call(
                target=f"{self.package_id}::adaptive_vault::rebalance",
                type_arguments=[self.config['token0_type'], self.config['token1_type']],
                arguments=[
                    tx.object(self.vault_id),
                    tx.object(self.pool_id),
                    tx.object(self.registry_id),
                    tx.pure(position_configs),
                    tx.object('0x6'),  # Clock
                ],
            )
            
            # Execute
            result = await self.client.sign_and_execute_transaction(
                tx,
                self.keypair,
            )
            
            if result['effects']['status']['status'] == 'success':
                self.rebalance_count += 1
                self.successful_rebalances += 1
                
                logger.info(f"✅ Rebalance successful!")
                logger.info(f"   Digest: {result['digest']}")
                logger.info(f"   Positions: {len(action.positions)}")
                logger.info(f"   Expected profit: ${action.expected_profit:.2f}")
                
                # Log for training feedback
                await self._log_rebalance_result(action, result)
            else:
                logger.error(f"❌ Rebalance failed: {result['effects']['status']}")
        
        except Exception as e:
            logger.error(f"❌ Error executing rebalance: {e}", exc_info=True)
    
    def _estimate_profit(
        self,
        positions: List[PositionConfig],
        state: MarketState,
    ) -> float:
        """Estimate profit from proposed positions"""
        
        # Simplified estimation
        # In production, this would be more sophisticated
        
        total_expected_fees = 0.0
        
        for pos in positions:
            # Calculate if position is in range
            in_range = (pos.tick_lower <= state.current_tick < pos.tick_upper)
            
            if in_range:
                # Estimate fees based on volume and allocation
                range_width = pos.tick_upper - pos.tick_lower
                concentration_factor = 1000.0 / range_width  # More concentrated = more fees
                
                capital_allocated = state.total_tvl * (pos.capital_allocation_bps / 10000)
                
                # Fees = volume * fee_tier * (your_liquidity / total_liquidity)
                # Simplified: assume you capture 1% of volume proportional to concentration
                expected_fees = (
                    state.volume_24h *
                    0.003 *  # 0.3% fee tier
                    (capital_allocated / (state.liquidity * state.current_price)) *
                    concentration_factor
                )
                
                total_expected_fees += expected_fees
        
        return total_expected_fees
    
    def _estimate_gas_cost(self, num_positions: int) -> float:
        """Estimate gas cost in USD"""
        # Rough estimate: base cost + per-position cost
        base_gas = 0.01  # Base transaction
        per_position = 0.005  # Per position
        
        sui_price = 2.0  # Would fetch real price
        
        return (base_gas + per_position * num_positions) * sui_price
    
    def _state_to_vector(self, state: MarketState) -> np.ndarray:
        """Convert MarketState to feature vector"""
        
        features = [
            # Price features
            np.log(state.current_price),
            state.price_change_1h,
            state.price_change_24h,
            
            # Volume features
            np.log(state.volume_24h + 1),
            state.volatility_7d,
            
            # Liquidity features
            np.log(state.liquidity + 1),
            state.bid_ask_spread,
            
            # Vault features
            np.log(state.total_tvl + 1),
            state.idle_capital_pct,
            state.active_positions / 5.0,  # Normalize
            state.last_rebalance_ago_minutes / 1440.0,  # Normalize to days
            
            # Historical features (flatten)
            *state.price_history[-10:],  # Last 10 prices
            *state.volume_history[-10:],  # Last 10 volumes
        ]
        
        return np.array(features, dtype=np.float32)
    
    def _time_since_last_rebalance(self) -> int:
        """Minutes since last rebalance"""
        # Would track from vault state
        return 999  # Placeholder
    
    async def _call_view_function(self, function_name: str, args: list):
        """Call a view function on the vault"""
        # Pseudo-code
        return {}
    
    async def _get_price_history(self, lookback: int) -> List[float]:
        """Fetch price history from indexer"""
        # Pseudo-code
        return [1.0] * lookback
    
    async def _get_volume_history(self, lookback: int) -> List[float]:
        """Fetch volume history"""
        # Pseudo-code
        return [1000.0] * lookback
    
    def _parse_sqrt_price(self, sqrt_price_x96: int) -> float:
        """Convert sqrt price to regular price"""
        sqrt_price = sqrt_price_x96 / (2 ** 96)
        return sqrt_price ** 2
    
    def _minutes_since_last_rebalance(self, vault_snapshot: dict) -> int:
        """Calculate minutes since last rebalance"""
        import time
        current_ms = int(time.time() * 1000)
        return (current_ms - vault_snapshot['last_rebalance_time']) // 60000
    
    async def _log_rebalance_result(self, action: RebalanceAction, result: dict):
        """Log rebalance for training feedback"""
        # Store to database for later model retraining
        pass


# ===== Training Pipeline =====

def train_rl_model(historical_data_path: str, output_model_path: str):
    """Train the RL model on historical data"""
    
    import gym
    from stable_baselines3 import PPO
    
    # This would be a full training pipeline
    # Simplified here for brevity
    
    logger.info("🎓 Training RL model...")
    
    # Load historical data
    # Create environment
    # Train model
    # Save model
    
    logger.info(f"✅ Model saved to {output_model_path}")


# ===== Main Entry Point =====

async def main():
    # Configuration
    config = {
        'vault_id': '0x...',
        'pool_id': '0x...',
        'package_id': '0x...',
        'registry_id': '0x...',
        'token0_type': '0x2::sui::SUI',
        'token1_type': '0x...::usdc::USDC',
        'model_path': './models/adaptive_lp_v1.pth',
        'check_interval_seconds': 60,
        'min_rebalance_interval_minutes': 60,
        'max_positions': 5,
    }
    
    # Initialize
    client = SuiClient('https://fullnode.testnet.sui.io:443')
    keypair = SuiKeypair.from_private_key('...')
    
    agent = AdaptiveLPAgent(
        vault_id=config['vault_id'],
        pool_id=config['pool_id'],
        package_id=config['package_id'],
        registry_id=config['registry_id'],
        model_path=config['model_path'],
        sui_client=client,
        keypair=keypair,
        config=config,
    )
    
    # Run
    await agent.run()


if __name__ == '__main__':
    asyncio.run(main())