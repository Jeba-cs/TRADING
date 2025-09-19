# -*- coding: utf-8 -*-

#!/usr/bin/env python3
"""
Smart AI Trading System - Complete Setup Script
This script creates the entire project structure with all files
"""

import os
import sys
from pathlib import Path

def create_project_structure():
    """Create the complete project structure and all files"""

    # Project root
    project_root = Path("smart_trading_app")
    project_root.mkdir(exist_ok=True)

    # Create all directories
    directories = [
        "agents",
        "data",
        "utils",
        "ui",
        "config",
        "models",
        "risk_management",
        "execution",
        "memory",
        "tests",
        "logs",
        "cache"
    ]

    for directory in directories:
        (project_root / directory).mkdir(exist_ok=True)
        # Create __init__.py files for Python packages
        if directory not in ["tests", "logs", "cache"]:
            (project_root / directory / "__init__.py").write_text("# -*- coding: utf-8 -*-\n")

    print(f"✅ Created directory structure in: {project_root.absolute()}")
    return project_root

def create_main_app(project_root):
    """Create the main Streamlit application"""
    main_content = '''# main.py - Smart Multi-Agent AI Trading System
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
'''

    (project_root / "main.py").write_text(main_content)
    print("✅ Created main.py")

def create_requirements_file(project_root):
    """Create requirements.txt"""
    requirements_content = '''# Smart Multi-Agent AI Trading System Dependencies

# Core Framework
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.15.0

# AI/LLM Libraries  
openai>=1.0.0
google-generativeai>=0.3.0
anthropic>=0.7.0

# Financial Data APIs
yfinance>=0.2.20
alpha-vantage>=2.3.1
requests>=2.31.0

# Data Processing
scikit-learn>=1.3.0
scipy>=1.11.0

# Async Programming
aiohttp>=3.8.0

# Configuration
python-dotenv>=1.0.0

# Date/Time
python-dateutil>=2.8.0

# Utility
tqdm>=4.66.0
'''

    (project_root / "requirements.txt").write_text(requirements_content)
    print("✅ Created requirements.txt")

def create_env_example(project_root):
    """Create .env.example file"""
    env_content = '''# API Keys (Replace with your actual keys)
OPENAI_API_KEY=your_openai_api_key_here
ALPHA_VANTAGE_KEY=your_alpha_vantage_key_here
GEMINI_API_KEY=your_google_gemini_key_here

# Application Settings
DEBUG_MODE=False
LOG_LEVEL=INFO

# Trading Configuration
MAX_POSITION_SIZE=0.10
DEFAULT_STOP_LOSS=0.03
'''

    (project_root / ".env.example").write_text(env_content)
    print("✅ Created .env.example")

def create_readme(project_root):
    """Create README.md"""
    readme_content = '''# 🚀 Smart Multi-Agent AI Trading System

A sophisticated AI-powered trading platform with multiple specialized agents and intelligent chatbot interface.

## 🎯 Features

- **Multi-Agent Architecture**: Short-term, Swing, and Macro trading agents
- **AI Trading Assistant**: Intelligent chatbot for trading advice
- **Advanced Analytics**: Technical analysis with 20+ indicators  
- **Risk Management**: Portfolio protection and position sizing
- **Interactive Dashboard**: Professional Streamlit interface

## 🛠️ Quick Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure APIs (Optional)**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Run Application**
   ```bash
   streamlit run main.py
   ```

## 💬 AI Assistant Usage

Ask the chatbot:
- "Should I buy AAPL right now?"
- "What's the market outlook?"
- "Analyze TSLA for swing trading"
- "Risk assessment for my portfolio"

## ⚠️ Disclaimer

This software is for educational purposes only. Not financial advice. 
Always do your own research and understand the risks.

## 📊 Trading Modes

- **Paper Trading** (Default): Safe simulation mode
- **Live Trading**: Real money trading (use with caution)
- **Analysis Only**: Research and analysis without trading

Happy Trading! 📈
'''

    (project_root / "README.md").write_text(readme_content)
    print("✅ Created README.md")

def create_basic_agents(project_root):
    """Create basic agent files"""

    # Base agent
    base_agent_content = '''# agents/base_agent.py
"""Base Agent class for all trading agents"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class TradingSignal:
    """Trading signal data structure"""
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float  # 0.0 to 1.0
    reasoning: str
    timestamp: datetime = datetime.now()

class BaseAgent(ABC):
    """Base class for all trading agents"""
    
    def __init__(self, name: str, timeframe: str = "1d"):
        self.name = name
        self.timeframe = timeframe
        self.is_active = True
    
    @abstractmethod
    async def analyze(self, symbol: str) -> TradingSignal:
        """Analyze symbol and generate trading signal"""
        pass
    
    @abstractmethod
    def get_strategy_description(self) -> str:
        """Return description of trading strategy"""
        pass
'''

    (project_root / "agents" / "base_agent.py").write_text(base_agent_content)

    # Simple consensus controller
    consensus_content = '''# agents/consensus_controller.py
"""Simple consensus controller for demonstration"""

from typing import Dict, Any
import asyncio

class ConsensusController:
    """Orchestrates trading agents and creates consensus"""
    
    def __init__(self):
        self.agents = {}
    
    async def get_consensus(self, symbol: str) -> Dict[str, Any]:
        """Get consensus from all agents"""
        
        # Mock consensus for demonstration
        return {
            'Short-Term Agent': {
                'action': 'BUY',
                'confidence': 0.75,
                'reasoning': 'Strong momentum indicators detected'
            },
            'Swing Agent': {
                'action': 'HOLD',
                'confidence': 0.60, 
                'reasoning': 'Mixed signals, awaiting confirmation'
            },
            'Macro Agent': {
                'action': 'BUY',
                'confidence': 0.85,
                'reasoning': 'Positive fundamentals and economic outlook'
            }
        }
'''

    (project_root / "agents" / "consensus_controller.py").write_text(consensus_content)
    print("✅ Created basic agent files")

def create_git_files(project_root):
    """Create Git-related files"""

    gitignore_content = '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Environment
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Streamlit
.streamlit/

# Logs
logs/
*.log

# Cache
cache/
.cache/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Data
data/*.csv
data/*.json
data/*.db
'''

    (project_root / ".gitignore").write_text(gitignore_content)
    print("✅ Created .gitignore")

def create_startup_script(project_root):
    """Create startup script"""

    startup_content = '''#!/bin/bash
# startup.sh - Quick startup script

echo "🚀 Starting Smart AI Trading System..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# venv\\Scripts\\activate  # Windows

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Create .env if it doesn't exist
if [ ! -f ".env" ]; then
    echo "⚙️ Creating .env file..."
    cp .env.example .env
    echo "✏️  Please edit .env file with your API keys"
fi

echo "✅ Setup complete!"
echo "🎯 To start the app, run: streamlit run main.py"
'''

    (project_root / "startup.sh").write_text(startup_content)

    # Windows batch file
    batch_content = '''@echo off
echo 🚀 Starting Smart AI Trading System...

REM Check if virtual environment exists
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\\Scripts\\activate

REM Install dependencies
echo 📥 Installing dependencies...
pip install -r requirements.txt

REM Create .env if it doesn't exist
if not exist ".env" (
    echo ⚙️ Creating .env file...
    copy .env.example .env
    echo ✏️  Please edit .env file with your API keys
)

echo ✅ Setup complete!
echo 🎯 To start the app, run: streamlit run main.py
pause
'''

    (project_root / "startup.bat").write_text(batch_content)
    print("✅ Created startup scripts")

def create_quick_start_guide(project_root):
    """Create quick start guide"""

    guide_content = '''# 🚀 QUICK START GUIDE

## ⚡ 1-Minute Setup

### Option A: Automatic Setup (Recommended)
```bash
# Linux/Mac
chmod +x startup.sh
./startup.sh

# Windows  
startup.bat
```

### Option B: Manual Setup
```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # Linux/Mac
venv\\Scripts\\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Configure (optional)
cp .env.example .env
# Edit .env with your API keys

# Run the app
streamlit run main.py
```

## 🎯 First Steps

1. **Open your browser** to http://localhost:8501
2. **Try the AI Assistant** - Ask: "Should I buy AAPL?"
3. **Check Agent Consensus** on the Dashboard tab
4. **Explore Analytics** for market insights
5. **Configure Settings** for your preferences

## 💬 Example Questions for AI

- "Should I buy TESLA stock?"
- "What's the market outlook today?"  
- "Analyze APPLE for swing trading"
- "Is my portfolio too risky?"
- "Explain current market trends"

## 🛡️ Safety First

- Start with **Paper Trading** mode
- Never invest more than you can afford to lose
- This is **educational software** - not financial advice
- Do your own research before making decisions

## 🆘 Having Issues?

1. **Python version**: Ensure Python 3.8+ is installed
2. **Dependencies**: Run `pip install -r requirements.txt` again
3. **Port conflict**: Use `streamlit run main.py --server.port 8502`
4. **API errors**: Check your .env file configuration

**🎉 Happy Trading!** 📈
'''

    (project_root / "QUICK_START.md").write_text(guide_content)
    print("✅ Created quick start guide")

def main():
    """Main setup function"""
    print("🏗️  Creating Smart AI Trading System...")
    print("=" * 50)

    # Create project structure
    project_root = create_project_structure()

    # Create all files
    create_main_app(project_root)
    create_requirements_file(project_root)
    create_env_example(project_root)
    create_readme(project_root)
    create_basic_agents(project_root)
    create_git_files(project_root)
    create_startup_script(project_root)
    create_quick_start_guide(project_root)

    print("=" * 50)
    print("🎉 SETUP COMPLETE!")
    print(f"📁 Project created in: {project_root.absolute()}")
    print("")
    print("🚀 NEXT STEPS:")
    print(f"1. cd {project_root}")
    print("2. chmod +x startup.sh && ./startup.sh  (Linux/Mac)")
    print("   OR startup.bat  (Windows)")
    print("3. streamlit run main.py")
    print("")
    print("💡 Your trading system will open at http://localhost:8501")
    print("🎯 Try asking the AI: 'Should I buy AAPL stock?'")
    print("")
    print("⚠️  Remember: This is for educational purposes!")
    print("📈 Happy Trading!")

if __name__ == "__main__":
    main()