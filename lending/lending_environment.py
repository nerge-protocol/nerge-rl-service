import numpy as np
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

# ============================================================================
# 1. STATE REPRESENTATION ENGINE
# ============================================================================

class StateRepresentation:
    """Encode market conditions into RL state vector"""
    
    def __init__(self, asset):
        self.asset = asset
        self.state_dim = 15  # Dimension of state vector
        
        # Mock historical data storage for trends
        self.history = {
            'utilization': [],
            'volatility': [],
            'bad_debt': []
        }
        
    def encode_state(self, timestamp, current_state) -> np.ndarray:
        """
        Definition 3.1: State space
        s_t = (U_t, σ_t, r_t^{market}, R_t^{protocol}, D_t)
        Expanded with more features for better learning
        """
        
        # Update history
        self.history['utilization'].append(current_state['utilization'])
        self.history['volatility'].append(current_state['volatility'])
        self.history['bad_debt'].append(current_state['bad_debt'])
        
        # Keep window size reasonable
        if len(self.history['utilization']) > 30:
            self.history['utilization'].pop(0)
            self.history['volatility'].pop(0)
            self.history['bad_debt'].pop(0)

        # 1. Core features (5 dimensions)
        utilization = current_state['utilization']
        volatility = current_state['volatility']
        market_rate = current_state['market_rate']
        reserve_ratio = current_state['reserve_ratio']
        # Normalized bad debt (assuming TVL is available in state or passed in)
        tvl = current_state.get('tvl', 1000000) # Default to avoid div by zero
        recent_bad_debt = current_state['bad_debt']
        
        # 2. Derived features (5 dimensions)
        trend_utilization = self.get_trend(self.history['utilization'], window=7)
        trend_volatility = self.get_trend(self.history['volatility'], window=7)
        spread_vs_market = market_rate - current_state['current_rate']
        health_score = self.calculate_pool_health(utilization, reserve_ratio, recent_bad_debt/tvl)
        stress_indicator = self.calculate_stress_index(volatility, market_rate)
        
        # 3. Temporal features (5 dimensions)
        # Handle timestamp being potentially just a datetime object or similar
        if isinstance(timestamp, (int, float)):
             dt = datetime.fromtimestamp(timestamp)
        else:
             dt = timestamp
             
        hour_of_day = dt.hour / 24.0
        day_of_week = dt.weekday() / 7.0
        days_since_shock = self.days_since_last_shock(timestamp)
        volatility_regime = self.get_volatility_regime(volatility)
        market_regime = self.get_market_regime(market_rate)
        
        # Normalize all features to [0, 1]
        state_vector = np.array([
            utilization,                   # U ∈ [0, 1]
            min(volatility / 2.0, 1.0),   # σ ∈ [0, 200%] → [0, 1]
            min(market_rate / 0.5, 1.0),  # r_market ∈ [0, 50%] → [0, 1]
            reserve_ratio,                # R ∈ [0, 1]
            min(recent_bad_debt / (tvl * 0.01 + 1e-9), 1.0),  # D
            (trend_utilization + 1) / 2,  # Trend ∈ [-1, 1] → [0, 1]
            (trend_volatility + 1) / 2,
            (spread_vs_market + 0.2) / 0.4,  # Spread ∈ [-20%, +20%] → [0, 1]
            health_score,
            stress_indicator,
            hour_of_day,
            day_of_week,
            min(days_since_shock / 30.0, 1.0),
            volatility_regime,
            market_regime
        ])
        
        return state_vector
    
    def get_trend(self, historical, window):
        """Calculate trend (slope) over window"""
        if len(historical) < 2:
            return 0.0
        
        data = historical[-window:]
        if len(data) < 2:
            return 0.0
            
        x = np.arange(len(data))
        slope, _ = np.polyfit(x, data, 1)
        
        # Normalize slope to [-1, 1]
        normalized_slope = np.tanh(slope * 10)  # tanh for bounded output
        return normalized_slope

    def calculate_pool_health(self, utilization, reserve_ratio, bad_debt_ratio):
        # Simple heuristic
        score = 1.0
        if utilization > 0.95: score -= 0.2
        if reserve_ratio < 0.10: score -= 0.2
        if bad_debt_ratio > 0.01: score -= 0.3
        return max(0.0, score)

    def calculate_stress_index(self, volatility, market_rate):
        # Higher vol and high market rates = high stress
        stress = (volatility / 2.0) * 0.5 + (market_rate / 0.2) * 0.5
        return min(1.0, stress)

    def days_since_last_shock(self, timestamp):
        # Placeholder
        return 10.0

    def get_volatility_regime(self, volatility):
        if volatility < 0.3: return 0.0 # Low
        if volatility < 0.8: return 0.5 # Medium
        return 1.0 # High

    def get_market_regime(self, market_rate):
        if market_rate < 0.03: return 0.0 # Low rates
        if market_rate < 0.08: return 0.5 # Normal
        return 1.0 # High rates

# ============================================================================
# 2. ACTION SPACE & POLICY PARAMETERIZATION
# ============================================================================

class ActionSpace:
    """Define possible actions (interest rate curve adjustments)"""
    
    def __init__(self):
        # Action: Adjust parameters of rate curve
        # r(U) = r₀ + r₁·U + r₂·max(U - U*, 0)
        
        self.action_dim = 4  # [Δr₀, Δr₁, Δr₂, ΔU*]
        
        # Action bounds (relative changes)
        self.bounds = {
            'dr0': (-0.02, 0.02),   # ±2% change in base rate
            'dr1': (-0.05, 0.05),   # ±5% change in linear coefficient
            'dr2': (-0.10, 0.10),   # ±10% change in kink coefficient
            'dUstar': (-0.10, 0.10) # ±10% change in kink position
        }
        
    def decode_action(self, action_vector: np.ndarray, current_params: Dict) -> Dict:
        """Convert RL action vector to actual rate parameters"""
        
        # Clip actions to bounds
        # Note: action_vector is assumed to be scaled to [-1, 1] coming from agent, 
        # but here we expect it to be mapped to the bounds already or we map it.
        # The agent code usually outputs [-1, 1] and scales it. 
        # Let's assume input here is already scaled to the bounds by the agent.
        
        # Apply actions to current parameters
        new_params = {
            'r0': current_params['r0'] + action_vector[0],
            'r1': current_params['r1'] + action_vector[1],
            'r2': current_params['r2'] + action_vector[2],
            'Ustar': current_params['Ustar'] + action_vector[3]
        }
        
        # Apply constraints
        new_params['r0'] = max(0.0, new_params['r0'])          # Non-negative base rate
        new_params['Ustar'] = np.clip(new_params['Ustar'], 0.5, 0.95)  # Reasonable kink
        
        return new_params
    
    def calculate_rate(self, params: Dict, utilization: float) -> float:
        """Calculate interest rate given parameters and utilization"""
        
        r = (
            params['r0'] +
            params['r1'] * utilization +
            params['r2'] * max(utilization - params['Ustar'], 0)
        )
        
        # Apply bounds
        r = np.clip(r, 0.0, 1.0)  # Between 0% and 100%
        
        return r

# ============================================================================
# 3. REWARD FUNCTION (CRITICAL COMPONENT)
# ============================================================================

class RewardFunction:
    """
    Definition 3.3: Reward function
    R(s_t, a_t) = w₁·Revenue_t + w₂·U_t - w₃·BadDebt_t - w₄·|Δr_t|
    """
    
    def __init__(self, weights=None):
        # Default weights (tuned via hyperparameter optimization)
        self.weights = weights or {
            'revenue': 1.0,      # w₁: Maximize protocol revenue
            'utilization': 0.3,  # w₂: Encourage high but not max utilization
            'bad_debt': 2.0,     # w₃: Strong penalty for bad debt
            'rate_volatility': 0.5,  # w₄: Penalize large rate changes (UX)
            'stability': 0.2,    # Additional: Reward stability
            'competitiveness': 0.1  # Additional: Reward being competitive
        }
        
    def calculate_reward(self, 
                        state_before: Dict,
                        state_after: Dict,
                        action: np.ndarray,
                        params_before: Dict,
                        params_after: Dict) -> float:
        """
        Calculate immediate reward for state-action pair
        """
        
        # 1. Revenue component
        revenue_change = state_after['revenue_24h'] - state_before['revenue_24h']
        tvl = state_before.get('tvl', 1.0)
        revenue_term = self.weights['revenue'] * revenue_change / tvl
        
        # 2. Utilization component
        # Target utilization around 85% (high but safe)
        target_utilization = 0.85
        util_penalty = abs(state_after['utilization'] - target_utilization)
        utilization_term = self.weights['utilization'] * (1 - util_penalty)
        
        # 3. Bad debt component (strong penalty)
        bad_debt_change = state_after['bad_debt'] - state_before['bad_debt']
        bad_debt_term = -self.weights['bad_debt'] * bad_debt_change / tvl
        
        # 4. Rate volatility component (user experience)
        rate_change_magnitude = self.calculate_rate_change_magnitude(
            params_before, params_after, state_before['utilization']
        )
        volatility_term = -self.weights['rate_volatility'] * rate_change_magnitude
        
        # 5. Stability bonus (reward for avoiding extreme states)
        stability_bonus = self.calculate_stability_bonus(state_after)
        
        # 6. Competitiveness bonus (vs market rates)
        competitiveness = self.calculate_competitiveness(state_after)
        competitiveness_term = self.weights['competitiveness'] * competitiveness
        
        # Combine all terms
        total_reward = (
            revenue_term +
            utilization_term +
            bad_debt_term +
            volatility_term +
            stability_bonus +
            competitiveness_term
        )
        
        # Clip reward to reasonable range
        total_reward = np.clip(total_reward, -10.0, 10.0)
        
        return total_reward
    
    def calculate_rate_change_magnitude(self, params_before, params_after, utilization):
        """How much did rates change for typical user?"""
        
        action_space = ActionSpace()
        
        # Calculate rate at current utilization before and after
        rate_before = action_space.calculate_rate(params_before, utilization)
        rate_after = action_space.calculate_rate(params_after, utilization)
        
        # Also consider rate at other utilizations (weighted average)
        util_points = [0.3, 0.5, 0.7, 0.85, 0.95]
        changes = []
        
        for u in util_points:
            r_b = action_space.calculate_rate(params_before, u)
            r_a = action_space.calculate_rate(params_after, u)
            changes.append(abs(r_a - r_b))
        
        # Weight changes by probability of utilization (assume normal around current)
        weights = np.exp(-0.5 * ((np.array(util_points) - utilization) / 0.2) ** 2)
        weights = weights / weights.sum()
        
        weighted_change = np.dot(changes, weights)
        
        return weighted_change
    
    def calculate_stability_bonus(self, state):
        """Reward being in safe operating region"""
        
        bonus = 0.0
        
        # Utilization in good range [0.6, 0.9]
        if 0.6 <= state['utilization'] <= 0.9:
            bonus += 0.1
        
        # Reserve ratio above minimum
        if state['reserve_ratio'] > 0.1:
            bonus += 0.05
        
        # No bad debt recently (assuming bad_debt_7d is tracked or approximated)
        if state.get('bad_debt', 0) == 0:
            bonus += 0.05
        
        # Volatility not extreme
        if state['volatility'] < 1.0:  # <100% annualized
            bonus += 0.05
        
        return bonus
    
    def calculate_competitiveness(self, state):
        """How competitive are our rates vs market?"""
        
        our_rate = state['current_rate']
        market_rate = state['market_rate']
        
        # Being within ±1% of market is good
        spread = abs(our_rate - market_rate)
        if spread <= 0.01:
            return 1.0 - (spread / 0.01)  # 1.0 if equal, 0.0 if 1% diff
        else:
            return max(0.0, 1.0 - (spread - 0.01) / 0.04)  # Decay to 0 at 5% diff

# ============================================================================
# 4. TRAINING ENVIRONMENT
# ============================================================================

class LendingEnvironment:
    """Simulated environment for RL training"""
    
    def __init__(self, asset='ETH', initial_tvl=100000000):
        self.asset = asset
        self.tvl = initial_tvl
        
        # Initial state
        self.state = {
            'utilization': 0.75,
            'volatility': 0.8,  # 80% annualized
            'market_rate': 0.06,
            'reserve_ratio': 0.15,
            'bad_debt': 0.0,
            'revenue_24h': 0.0,
            'current_rate': 0.08,
            'tvl': initial_tvl,
            'params': {
                'r0': 0.02,
                'r1': 0.03,
                'r2': 0.20,
                'Ustar': 0.85
            }
        }
        
        # State encoder
        self.encoder = StateRepresentation(asset)
        
        # Episode tracking
        self.current_step = 0
        self.max_steps = 365  # One year of daily steps
        self.current_date = datetime.now()
        
    def step(self, action):
        """
        Execute action, return (next_state, reward, done, info)
        """
        
        # Decode action into new parameters
        action_space = ActionSpace()
        new_params = action_space.decode_action(
            action, self.state['params']
        )
        
        # Apply new parameters
        old_state = self.state.copy()
        self.state['params'] = new_params
        
        # Calculate new rate
        new_rate = action_space.calculate_rate(
            new_params, self.state['utilization']
        )
        self.state['current_rate'] = new_rate
        
        # Simulate market response to new rate
        self._simulate_market_response()
        
        # Update state variables
        self._update_state_variables()
        
        # Calculate reward
        reward_fn = RewardFunction()
        reward = reward_fn.calculate_reward(
            old_state, self.state, action,
            old_state['params'], new_params
        )
        
        # Check if episode done
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        # Encode state for RL agent
        self.current_date += timedelta(days=1)
        encoded_state = self.encoder.encode_state(
            self.current_date, self.state
        )
        
        info = {
            'utilization': self.state['utilization'],
            'rate': self.state['current_rate'],
            'revenue': self.state['revenue_24h'],
            'bad_debt': self.state['bad_debt']
        }
        
        return encoded_state, reward, done, info
    
    def _simulate_market_response(self):
        """Simulate how borrowers and lenders react to rate changes"""
        
        current_rate = self.state['current_rate']
        market_rate = self.state['market_rate']
        
        # Borrowers: sensitive to rate increases
        rate_diff = current_rate - market_rate
        
        if rate_diff > 0.02:  # Our rate > market by 2%
            # Borrowers leave
            outflow = min(self.tvl * 0.05, self.tvl * rate_diff * 10)
            self.tvl -= outflow
            self.state['utilization'] *= 0.95
        
        elif rate_diff < -0.01:  # Our rate < market by 1%
            # Borrowers come
            inflow = min(self.tvl * 0.10, self.tvl * abs(rate_diff) * 20)
            self.tvl += inflow
            self.state['utilization'] = min(0.95, self.state['utilization'] * 1.05)
        
        # Simulate random market events
        if random.random() < 0.05:  # 5% chance of market event
            self._simulate_market_event()
    
    def _simulate_market_event(self):
        """Simulate black swan or volatility event"""
        
        event_type = random.choice(['volatility_spike', 'price_crash', 'liquidity_drain'])
        
        if event_type == 'volatility_spike':
            self.state['volatility'] = min(2.0, self.state['volatility'] * 2.0)
            
        elif event_type == 'price_crash':
            # Increased liquidations
            liquidation_volume = self.tvl * 0.1 * random.random()
            bad_debt = liquidation_volume * 0.05 * random.random()
            self.state['bad_debt'] += bad_debt
            self.state['utilization'] *= 0.9
            
        elif event_type == 'liquidity_drain':
            # Liquidity leaves protocol
            drain = self.tvl * 0.2 * random.random()
            self.tvl -= drain
            self.state['utilization'] = min(0.99, self.state['utilization'] * 1.1)
    
    def _update_state_variables(self):
        """Update state based on new conditions"""
        
        # Revenue: interest earned
        borrowed = self.tvl * self.state['utilization']
        daily_revenue = borrowed * self.state['current_rate'] / 365
        self.state['revenue_24h'] = daily_revenue
        
        # Update market rate (random walk)
        market_change = random.normalvariate(0, 0.001)
        self.state['market_rate'] = max(0.0, 
            self.state['market_rate'] + market_change
        )
        
        # Update volatility (mean-reverting)
        vol_target = 0.8
        vol_reversion = 0.1 * (vol_target - self.state['volatility'])
        vol_shock = random.normalvariate(0, 0.05)
        self.state['volatility'] = max(0.1, 
            self.state['volatility'] + vol_reversion + vol_shock
        )
        
        # Update reserves (from revenue)
        reserve_growth = daily_revenue * 0.15  # 15% of revenue to reserves
        reserve_balance = self.state['reserve_ratio'] * self.tvl
        new_reserve_balance = reserve_balance + reserve_growth
        self.state['reserve_ratio'] = new_reserve_balance / self.tvl
        
        # Decay bad debt (write off over time)
        self.state['bad_debt'] *= 0.99
        self.state['tvl'] = self.tvl # Update TVL in state dict
    
    def reset(self):
        """Reset environment to initial state"""
        self.__init__(asset=self.asset, initial_tvl=100000000) # Reset to initial
        encoded_state = self.encoder.encode_state(self.current_date, self.state)
        return encoded_state
