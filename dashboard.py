# dashboard.py - Real-time monitoring dashboard

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import redis
import json

class HedgeDashboard:
    def __init__(self, redis_url="redis://localhost:6379"):
        self.redis = redis.Redis.from_url(redis_url)
        
    def run(self):
        st.set_page_config(layout="wide", page_title="Hedge Service Dashboard")
        
        st.title("🤖 RL Hedge Service Dashboard")
        st.markdown("---")
        
        # Sidebar controls
        with st.sidebar:
            st.header("Controls")
            refresh_rate = st.slider("Refresh Rate (seconds)", 1, 60, 5)
            time_window = st.selectbox("Time Window", ["1h", "6h", "24h", "7d"])
            
            if st.button("Force Rebalance"):
                self.trigger_rebalance()
        
        # Main dashboard
        col1, col2, col3 = st.columns(3)
        
        with col1:
            self.display_summary_stats()
        
        with col2:
            self.display_exposure_chart()
        
        with col3:
            self.display_performance_metrics()
        
        st.markdown("---")
        
        # Charts row
        col1, col2 = st.columns(2)
        
        with col1:
            self.display_pnl_chart()
        
        with col2:
            self.display_hedge_decisions()
        
        # Position table
        st.subheader("Active Positions")
        self.display_positions_table()
        
        # Auto-refresh
        st.experimental_rerun()
    
    def display_summary_stats(self):
        st.subheader("📊 Summary")
        
        try:
            metrics = json.loads(self.redis.get("performance_metrics") or "{}")
            
            st.metric("Total PnL", f"${metrics.get('total_pnl', 0):,.2f}")
            st.metric("Win Rate", f"{metrics.get('win_rate', 0)*100:.1f}%")
            st.metric("Active Positions", len(json.loads(self.redis.get("active_positions") or "[]")))
            
        except:
            st.error("Error loading summary stats")
    
    def display_exposure_chart(self):
        st.subheader("📈 Exposure")
        
        # Get exposure data from Redis
        exposure_data = self.redis.get("exposure_history")
        
        if exposure_data:
            df = pd.DataFrame(json.loads(exposure_data))
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['delta'],
                name="Delta (ETH)",
                line=dict(color='blue')
            ))
            
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['vega'] / 1000,
                name="Vega ($k)",
                line=dict(color='green')
            ))
            
            fig.update_layout(
                height=300,
                showlegend=True,
                xaxis_title="Time",
                yaxis_title="Exposure"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No exposure data available")
    
    def display_performance_metrics(self):
        st.subheader("🎯 Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Sharpe Ratio", "1.24")
            st.metric("Max Drawdown", "-4.2%")
        
        with col2:
            st.metric("Total Trades", "142")
            st.metric("Avg Trade PnL", "$245")
    
    def display_pnl_chart(self):
        st.subheader("💰 PnL History")
        
        # Mock data for example
        dates = pd.date_range(end=datetime.now(), periods=100, freq='H')
        pnl = np.cumsum(np.random.randn(100) * 1000)
        
        fig = px.area(
            x=dates,
            y=pnl,
            labels={'x': 'Time', 'y': 'PnL ($)'}
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    def display_hedge_decisions(self):
        st.subheader("🤖 Recent Hedge Decisions")
        
        decisions = [
            {"time": "10:30", "action": "Buy ETH Put", "amount": "$50k", "confidence": "85%"},
            {"time": "10:15", "action": "Short ETH Perp", "amount": "$100k", "confidence": "72%"},
            {"time": "09:45", "action": "Sell BTC Call", "amount": "$30k", "confidence": "68%"},
        ]
        
        for decision in decisions:
            with st.container():
                cols = st.columns([1, 3, 2, 2])
                cols[0].write(f"**{decision['time']}**")
                cols[1].write(decision['action'])
                cols[2].write(decision['amount'])
                cols[3].metric("Confidence", decision['confidence'])
    
    def display_positions_table(self):
        positions = json.loads(self.redis.get("active_positions") or "[]")
        
        if positions:
            df = pd.DataFrame(positions)
            st.dataframe(df[['instrument', 'asset', 'notional_usd', 'pnl_usd', 'created_at']])
        else:
            st.info("No active positions")
    
    def trigger_rebalance(self):
        # Send rebalance signal to service
        self.redis.publish("hedge_control", "rebalance")
        st.success("Rebalance triggered!")

if __name__ == "__main__":
    dashboard = HedgeDashboard()
    dashboard.run()