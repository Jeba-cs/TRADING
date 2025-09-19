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
warnings.filterwarnings('ignore')
# Add this import at the top of main.py (around line 15)
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our custom modules
from config.settings import *
from agents.consensus_controller import ConsensusController
from agents.chatbot_agent import TradingChatbot
from data.data_manager import DataManager
from ui.dashboard import TradingDashboard
from ui.chatbot_ui import ChatbotInterface
from utils.llm_handler import LLMHandler
from models.portfolio import Portfolio
from risk_management.risk_controller import RiskController

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
        self.setup_components()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'portfolio' not in st.session_state:
            st.session_state.portfolio = Portfolio()
        if 'consensus_controller' not in st.session_state:
            st.session_state.consensus_controller = ConsensusController()
        if 'chatbot' not in st.session_state:
            st.session_state.chatbot = TradingChatbot()
        if 'data_manager' not in st.session_state:
            st.session_state.data_manager = DataManager()
        if 'risk_controller' not in st.session_state:
            st.session_state.risk_controller = RiskController()
        if 'trading_history' not in st.session_state:
            st.session_state.trading_history = []
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'selected_symbols' not in st.session_state:
            st.session_state.selected_symbols = ['AAPL', 'TSLA', 'GOOGL', 'MSFT', 'AMZN']
        if 'auto_trading' not in st.session_state:
            st.session_state.auto_trading = False

    def setup_components(self):
        selected_symbols = ['AAPL', 'TSLA', 'GOOGL', 'MSFT']  # Replace with your dynamic symbol list if you have
        self.dashboard = TradingDashboard(selected_symbols)
        self.chatbot_ui = ChatbotInterface()
        # ...

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
        
        if st.session_state.auto_trading:
            st.sidebar.success("🟢 Auto Trading Active")
        else:
            st.sidebar.info("🔵 Manual Mode")
        
        # Portfolio Summary
        st.sidebar.subheader("💼 Portfolio Summary")
        portfolio_value = st.session_state.portfolio.get_total_value()
        daily_pnl_dict = st.session_state.portfolio.get_daily_pnl()
        latest_pnl = list(daily_pnl_dict.values())[-1] if daily_pnl_dict else 0.0
        pnl_percentage = (latest_pnl / portfolio_value * 100) if portfolio_value > 0 else 0.0

        st.sidebar.metric(
            "Portfolio Value",
            f"${portfolio_value:,.2f}",
            f"{pnl_percentage:+.2f}%"  # ✅ Now using actual percentage
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
            
            st.subheader("🎲 Risk Metrics")
            self.render_risk_metrics()
    
    def render_agent_consensus(self):
        """Display agent consensus and recommendations"""
        try:
            # Get consensus from all agents
            if not hasattr(st.session_state, "_consensus_cache"):
                st.session_state._consensus_cache = asyncio.run(st.session_state.consensus_controller.get_consensus(
                    st.session_state.selected_symbols[0] if st.session_state.selected_symbols else 'AAPL'
                ))

            consensus_data = st.session_state._consensus_cache

            for agent_name, recommendation in consensus_data.items():
                agent_color = {
                    'Short-Term Agent': '#ff4444',
                    'Swing Agent': '#ffaa44',
                    'Macro Agent': '#44ff44'
                }.get(agent_name, '#4444ff')

                # Defensive programming for recommendation type
                if isinstance(recommendation, dict):
                    action = recommendation.get('action', 'HOLD')
                    confidence = recommendation.get('confidence', 0.5)
                    reasoning = recommendation.get('reasoning', 'Analysis in progress...')
                else:
                    # When recommendation is a simple string
                    action = recommendation
                    confidence = 0.5
                    reasoning = "Analysis pending or not provided."

                # Use action, confidence, reasoning for rendering here
                st.markdown(f"""
                #### {agent_name}
                **Action:** {action}
                **Confidence:** {confidence:.1%}
                **Reasoning:** {reasoning}
                """, unsafe_allow_html=True)


        except Exception as e:
            st.error(f"Error loading agent consensus: {str(e)}")
    
    def render_market_overview(self):
        """Display market overview with key metrics"""
        try:
            # Create sample market data
            market_data = {
                'Symbol': st.session_state.selected_symbols[:5],
                'Price': [150.25, 235.67, 2834.50, 338.20, 3245.80],
                'Change': ['+2.5%', '-1.2%', '+0.8%', '+3.1%', '-0.5%'],
                'Volume': ['45.2M', '23.8M', '12.4M', '28.9M', '18.7M']
            }
            
            df = pd.DataFrame(market_data)
            st.dataframe(df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error loading market data: {str(e)}")
    
    def render_price_charts(self):
        """Render interactive price charts"""
        try:
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
                
                # Create realistic price movements
                import numpy as np
                np.random.seed(42)
                base_price = 150.0
                price_changes = np.random.normal(0, 2, len(dates))
                prices = [base_price]
                
                for change in price_changes[1:]:
                    new_price = prices[-1] * (1 + change/100)
                    prices.append(max(new_price, 10))  # Minimum price of $10
                
                # Create candlestick chart
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
                
        except Exception as e:
            st.error(f"Error rendering charts: {str(e)}")
    
    def render_recent_trades(self):
        """Display recent trading activity"""
        if st.session_state.trading_history:
            df = pd.DataFrame(st.session_state.trading_history[-10:])  # Show last 10 trades
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
            order_type = st.selectbox("Order Type:", ["MARKET", "LIMIT"])
            
            if order_type == "LIMIT":
                limit_price = st.number_input("Limit Price:", min_value=0.01, value=100.0)
            
            submitted = st.form_submit_button("Execute Trade")
            
            if submitted:
                trade_data = {
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'action': action,
                    'quantity': quantity,
                    'type': order_type,
                    'status': 'EXECUTED'
                }
                
                if order_type == "LIMIT":
                    trade_data['limit_price'] = limit_price
                
                st.session_state.trading_history.append(trade_data)
                st.success(f"✅ {action} order for {quantity} shares of {symbol} executed!")
        
        # Emergency Actions
        st.markdown("### Emergency Actions")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚨 Close All Positions", type="secondary"):
                st.warning("All positions closed!")
        
        with col2:
            if st.button("⏸️ Pause Trading", type="secondary"):
                st.session_state.auto_trading = False
                st.info("Auto trading paused!")
    
    def render_risk_metrics(self):
        """Display risk management metrics"""
        risk_data = st.session_state.risk_controller.get_risk_metrics()
        
        st.metric("Portfolio Beta", f"{risk_data.get('beta', 1.2):.2f}")
        st.metric("VaR (1-day)", f"${risk_data.get('var_1d', 2500):,.0f}")
        st.metric("Sharpe Ratio", f"{risk_data.get('sharpe', 1.8):.2f}")
        st.metric("Max Drawdown", f"{risk_data.get('max_drawdown', -5.2):.1f}%")
    
    def render_chatbot_interface(self):
        """Render AI chatbot interface"""
        st.subheader("🤖 Trading AI Assistant")
        st.markdown("Ask me anything about trading, market analysis, or specific stocks!")
        
        # Chat history display
        for message in st.session_state.chat_history[-10:]:  # Show last 10 messages
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
                # Add user message to history
                st.session_state.chat_history.append({
                    'type': 'user',
                    'content': user_input,
                    'timestamp': datetime.now()
                })
                
                # Get AI response
                try:
                    response = st.session_state.chatbot.get_response(user_input)
                    st.session_state.chat_history.append({
                        'type': 'bot',
                        'content': response,
                        'timestamp': datetime.now()
                    })
                    st.rerun()
                except Exception as e:
                    st.error(f"Error getting AI response: {str(e)}")
        
        # Quick question buttons
        st.markdown("### Quick Questions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Market Analysis"):
                self.handle_quick_question("Give me a quick market analysis for today")
        
        with col2:
            if st.button("💡 Stock Suggestions"):
                self.handle_quick_question("Suggest some stocks to watch today")
        
        with col3:
            if st.button("⚠️ Risk Assessment"):
                self.handle_quick_question("Assess the current market risk level")
    
    def handle_quick_question(self, question):
        """Handle quick question button clicks"""
        try:
            st.session_state.chat_history.append({
                'type': 'user',
                'content': question,
                'timestamp': datetime.now()
            })
            
            response = st.session_state.chatbot.get_response(question)
            st.session_state.chat_history.append({
                'type': 'bot',
                'content': response,
                'timestamp': datetime.now()
            })
            st.rerun()
        except Exception as e:
            st.error(f"Error processing quick question: {str(e)}")
    
    def render_analytics_view(self):
        """Render analytics and performance view"""
        st.subheader("📊 Performance Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Portfolio Performance Chart
            st.markdown("#### Portfolio Performance")
            dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
            performance = [100 * (1.05 ** (i/30)) for i in range(len(dates))]  # 5% monthly growth
            
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
            st.metric("Avg Hold Time", "5.2 days", "-0.8 days")
        with col4:
            st.metric("Best Trade", "+$2,450", None)
    
    def render_settings_view(self):
        """Render application settings"""
        st.subheader("⚙️ System Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### API Configuration")
            with st.expander("Market Data APIs"):
                st.text_input("Alpha Vantage API Key", type="password", placeholder="Enter API key")
                st.text_input("Yahoo Finance (Free)", disabled=True, value="Enabled")
                st.text_input("IEX Cloud API Key", type="password", placeholder="Enter API key")
            
            with st.expander("AI/LLM APIs"):
                ai_provider = st.selectbox("AI Provider", ["OpenAI", "Google Gemini", "Anthropic Claude", "Local LLM"])
                st.text_input(f"{ai_provider} API Key", type="password", placeholder="Enter API key")
        
        with col2:
            st.markdown("#### Trading Settings")
            with st.expander("Execution Settings"):
                st.number_input("Order Timeout (seconds)", value=30)
                st.selectbox("Default Order Type", ["MARKET", "LIMIT", "STOP"])
                st.checkbox("Enable Pre-market Trading")
                st.checkbox("Enable After-hours Trading")
            
            with st.expander("Notification Settings"):
                st.checkbox("Email Notifications", value=True)
                st.checkbox("SMS Alerts")
                st.checkbox("Desktop Notifications", value=True)
                st.text_input("Email Address", placeholder="your.email@example.com")
        
        # Save Settings Button
        if st.button("💾 Save Settings", type="primary"):
            st.success("Settings saved successfully!")
    
    def run(self):
        """Main application entry point"""
        try:
            # Render header
            self.render_header()
            
            # Render sidebar and get settings
            settings = self.render_sidebar()
            
            # Render main dashboard
            self.render_main_dashboard(settings)
            
            # Auto-refresh for live data (every 30 seconds)
            if st.session_state.auto_trading:
                time.sleep(30)
                st.rerun()
                
        except Exception as e:
            st.error(f"Application Error: {str(e)}")
            st.info("Please refresh the page or contact support if the error persists.")

def main():
    """Application entry point"""
    app = SmartTradingApp()
    app.run()

if __name__ == "__main__":
    main()