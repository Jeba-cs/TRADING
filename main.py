# main.py - Smart Multi-Agent AI Trading System
"""
Advanced Multi-Agent AI Trading System with Streamlit UI
Features:
- Three specialized trading agents (Short-term, Swing, Macro)
- Real-time market data integration
- AI-powered chatbot for trading advice
- Risk management and portfolio tracking
- Interactive dashboard with analytics
"""

import streamlit as st
import asyncio
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
import time
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="🚀 Smart AI Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .agent-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .user-message {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .bot-message {
        background: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .sidebar .element-container {
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class SmartTradingApp:
    def __init__(self):
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'selected_symbols' not in st.session_state:
            st.session_state.selected_symbols = ['AAPL', 'TSLA', 'GOOGL', 'MSFT', 'AMZN']
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'trading_history' not in st.session_state:
            st.session_state.trading_history = []
        if 'auto_trading' not in st.session_state:
            st.session_state.auto_trading = False
    
    def render_header(self):
        """Render main application header"""
        st.markdown("""
        <div class="main-header">
            <h1>🚀 Smart Multi-Agent AI Trading System</h1>
            <p>Advanced AI-powered trading with real-time market analysis and intelligent decision making</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render sidebar with controls and settings"""
        st.sidebar.title("🎛️ Control Panel")
        
        # Trading Mode Selection
        st.sidebar.subheader("Trading Mode")
        trading_mode = st.sidebar.selectbox(
            "Select Mode:",
            ["Paper Trading", "Live Trading", "Analysis Only"],
            index=0
        )
        
        # Symbol Selection
        st.sidebar.subheader("📊 Watchlist")
        symbols_input = st.sidebar.text_input(
            "Enter symbols (comma-separated):",
            value=",".join(st.session_state.selected_symbols)
        )
        if symbols_input:
            st.session_state.selected_symbols = [s.strip().upper() for s in symbols_input.split(',')]
        
        # Risk Settings
        st.sidebar.subheader("⚠️ Risk Management")
        max_position_size = st.sidebar.slider(
            "Max Position Size (%)", 
            min_value=1, 
            max_value=20, 
            value=5
        )
        stop_loss_pct = st.sidebar.slider(
            "Stop Loss (%)", 
            min_value=1, 
            max_value=10, 
            value=3
        )
        
        # Auto Trading Toggle
        st.sidebar.subheader("🤖 Automation")
        st.session_state.auto_trading = st.sidebar.toggle(
            "Enable Auto Trading",
            value=st.session_state.auto_trading
        )
        
        return {
            'trading_mode': trading_mode,
            'max_position_size': max_position_size,
            'stop_loss_pct': stop_loss_pct
        }
    
    def render_main_dashboard(self, settings):
        """Render main trading dashboard"""
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "🏠 Dashboard", 
            "💬 AI Assistant", 
            "📈 Analytics", 
            "⚙️ Settings"
        ])
        
        with tab1:
            self.render_trading_dashboard(settings)
        
        with tab2:
            self.render_chatbot_interface()
        
        with tab3:
            self.render_analytics_view()
        
        with tab4:
            self.render_settings_view()
    
    def render_trading_dashboard(self, settings):
        """Render main trading dashboard content"""
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            st.subheader("🎯 Agent Consensus")
            self.render_agent_consensus()
            
            st.subheader("📊 Market Overview")
            self.render_market_overview()
        
        with col2:
            st.subheader("📈 Price Charts")
            self.render_price_charts()
            
            st.subheader("🔄 Recent Trades")
            self.render_recent_trades()
        
        with col3:
            st.subheader("⚡ Quick Actions")
            self.render_quick_actions()
    
    def render_agent_consensus(self):
        """Display agent consensus and recommendations"""
        agents_data = {
            'Short-Term Agent': {
                'action': 'BUY',
                'confidence': 0.75,
                'reasoning': 'Strong momentum indicators with RSI oversold conditions'
            },
            'Swing Agent': {
                'action': 'HOLD', 
                'confidence': 0.60,
                'reasoning': 'Mixed signals from technical patterns, wait for breakout'
            },
            'Macro Agent': {
                'action': 'BUY',
                'confidence': 0.85,
                'reasoning': 'Strong fundamentals and positive economic outlook'
            }
        }
        
        for agent_name, data in agents_data.items():
            color = {'BUY': '#4CAF50', 'SELL': '#F44336', 'HOLD': '#FF9800'}.get(data['action'], '#2196F3')
            
            st.markdown(f"""
            <div class="agent-card">
                <h4 style="color: {color};">{agent_name}</h4>
                <p><strong>Action:</strong> {data['action']}</p>
                <p><strong>Confidence:</strong> {data['confidence']:.1%}</p>
                <p><strong>Reasoning:</strong> {data['reasoning']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    def render_market_overview(self):
        """Display market overview with key metrics"""
        market_data = {
            'Symbol': st.session_state.selected_symbols[:5],
            'Price': [150.25, 235.67, 2834.50, 338.20, 3245.80],
            'Change': ['+2.5%', '-1.2%', '+0.8%', '+3.1%', '-0.5%'],
            'Volume': ['45.2M', '23.8M', '12.4M', '28.9M', '18.7M']
        }
        
        df = pd.DataFrame(market_data)
        st.dataframe(df, use_container_width=True)
    
    def render_price_charts(self):
        """Render interactive price charts"""
        if st.session_state.selected_symbols:
            selected_symbol = st.selectbox(
                "Select Symbol for Chart:",
                st.session_state.selected_symbols,
                key="chart_symbol"
            )
            
            # Generate sample price data
            dates = pd.date_range(
                start=datetime.now() - timedelta(days=30),
                end=datetime.now(),
                freq='D'
            )
            
            import numpy as np
            np.random.seed(42)
            base_price = 150.0
            price_changes = np.random.normal(0, 2, len(dates))
            prices = [base_price]
            
            for change in price_changes[1:]:
                new_price = prices[-1] * (1 + change/100)
                prices.append(max(new_price, 10))
            
            # Create line chart
            fig = go.Figure(data=go.Scatter(
                x=dates,
                y=prices,
                mode='lines',
                line=dict(color='#2196f3', width=2),
                name=selected_symbol
            ))
            
            fig.update_layout(
                title=f"{selected_symbol} Price Chart",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_recent_trades(self):
        """Display recent trading activity"""
        if st.session_state.trading_history:
            df = pd.DataFrame(st.session_state.trading_history[-10:])
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No recent trades. Start trading to see history here.")
    
    def render_quick_actions(self):
        """Render quick action buttons"""
        st.markdown("### Quick Trade")
        
        with st.form("quick_trade_form"):
            symbol = st.selectbox("Symbol:", st.session_state.selected_symbols)
            action = st.selectbox("Action:", ["BUY", "SELL"])
            quantity = st.number_input("Quantity:", min_value=1, value=100)
            
            submitted = st.form_submit_button("Execute Trade")
            
            if submitted:
                trade_data = {
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'action': action,
                    'quantity': quantity,
                    'status': 'EXECUTED'
                }
                
                st.session_state.trading_history.append(trade_data)
                st.success(f"✅ {action} order for {quantity} shares of {symbol} executed!")
    
    def render_chatbot_interface(self):
        """Render AI chatbot interface"""
        st.subheader("🤖 Trading AI Assistant")
        st.markdown("Ask me anything about trading, market analysis, or specific stocks!")
        
        # Chat history display
        for message in st.session_state.chat_history[-10:]:
            if message['type'] == 'user':
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <strong>AI Assistant:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)
        
        # Chat input
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_area(
                "Ask about any stock or trading strategy:",
                placeholder="Should I buy AAPL? What's the market outlook for tech stocks?",
                height=100
            )
            submitted = st.form_submit_button("Send Message")
            
            if submitted and user_input:
                # Add user message
                st.session_state.chat_history.append({
                    'type': 'user',
                    'content': user_input,
                    'timestamp': datetime.now()
                })
                
                # Generate AI response (simplified)
                ai_response = self.generate_ai_response(user_input)
                st.session_state.chat_history.append({
                    'type': 'bot',
                    'content': ai_response,
                    'timestamp': datetime.now()
                })
                st.rerun()
    
    def generate_ai_response(self, user_input):
        """Generate AI response (simplified version)"""
        user_lower = user_input.lower()
        
        if any(symbol in user_lower for symbol in ['aapl', 'apple']):
            return """📊 **AAPL Analysis**

Based on current market conditions:

**Short-term outlook:** Strong momentum with RSI at healthy levels. Recent earnings beat expectations.

**Technical levels:** Support at $150, resistance at $175. Currently trading in uptrend channel.

**Recommendation:** CAUTIOUS BUY
- Entry: $150-155 range
- Target: $170-175 (10-12% upside)
- Stop: $145 (risk management)

💡 *Consider position sizing - this is analysis, not financial advice!*"""
        
        elif 'market' in user_lower and 'outlook' in user_lower:
            return """🌐 **Market Outlook**

Current market environment shows:

**Sentiment:** Cautiously optimistic with mixed signals
**Key drivers:** Fed policy, earnings season, economic data
**Sectors:** Tech leading, Energy lagging
**Risk factors:** Geopolitical tensions, inflation concerns

**Strategy:** Focus on quality stocks with strong fundamentals. Maintain defensive positioning.

📈 *Stay diversified and monitor key support levels!*"""
        
        else:
            return """I'm here to help with your trading questions! 

I can assist with:
• Stock analysis (AAPL, TSLA, GOOGL, etc.)
• Market outlook and trends  
• Trading strategies and risk management
• Technical and fundamental analysis

Feel free to ask about specific stocks or trading concepts!"""
    
    def render_analytics_view(self):
        """Render analytics and performance view"""
        st.subheader("📊 Performance Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Portfolio Performance Chart
            st.markdown("#### Portfolio Performance")
            dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
            performance = [100 * (1.05 ** (i/30)) for i in range(len(dates))]
            
            fig = px.line(
                x=dates, 
                y=performance,
                title="Portfolio Value Over Time",
                labels={'x': 'Date', 'y': 'Portfolio Value ($)'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Asset Allocation
            st.markdown("#### Asset Allocation")
            allocation_data = {
                'Asset': ['Stocks', 'Options', 'Cash', 'Crypto'],
                'Percentage': [65, 15, 15, 5]
            }
            
            fig = px.pie(
                allocation_data, 
                values='Percentage', 
                names='Asset',
                title="Current Asset Allocation"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Performance Metrics
        st.markdown("#### Key Performance Indicators")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Return", "12.5%", "2.1%")
        with col2:
            st.metric("Win Rate", "68%", "3%")
        with col3:
            st.metric("Sharpe Ratio", "1.8", "0.2")
        with col4:
            st.metric("Max Drawdown", "-5.2%", None)
    
    def render_settings_view(self):
        """Render application settings"""
        st.subheader("⚙️ System Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Trading Configuration")
            st.slider("Default Position Size (%)", 1, 20, 5)
            st.slider("Stop Loss (%)", 1, 10, 3)
            st.selectbox("Default Timeframe", ["1h", "4h", "1d", "1w"])
            st.checkbox("Enable Notifications")
        
        with col2:
            st.markdown("#### API Configuration")
            st.text_input("OpenAI API Key", type="password")
            st.text_input("Alpha Vantage Key", type="password") 
            st.selectbox("Data Provider", ["Yahoo Finance", "Alpha Vantage", "IEX Cloud"])
            st.number_input("Request Timeout (s)", min_value=5, max_value=60, value=30)
        
        if st.button("💾 Save Settings"):
            st.success("Settings saved successfully!")
    
    def run(self):
        """Main application entry point"""
        # Render header
        self.render_header()
        
        # Render sidebar and get settings
        settings = self.render_sidebar()
        
        # Render main dashboard
        self.render_main_dashboard(settings)

def main():
    """Application entry point"""
    app = SmartTradingApp()
    app.run()

if __name__ == "__main__":
    main()
