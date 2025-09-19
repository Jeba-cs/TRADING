# config/settings.py
"""
Configuration settings for the Smart Trading System
"""

import os
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class TradingConfig:
    """Trading configuration parameters"""
    MAX_POSITION_SIZE: float = 0.10  # 10% of portfolio
    STOP_LOSS_PCT: float = 0.03      # 3% stop loss
    TAKE_PROFIT_PCT: float = 0.06    # 6% take profit
    MAX_OPEN_POSITIONS: int = 10
    MIN_TRADE_AMOUNT: float = 100.0
    COMMISSION_RATE: float = 0.001   # 0.1% commission
    
@dataclass 
class APIConfig:
    """API configuration settings"""
    ALPHA_VANTAGE_KEY: str = os.getenv('ALPHA_VANTAGE_KEY', '')
    OPENAI_API_KEY: str = os.getenv('OPENAI_API_KEY', '')
    GEMINI_API_KEY: str = os.getenv('GEMINI_API_KEY', '')
    IEX_CLOUD_TOKEN: str = os.getenv('IEX_CLOUD_TOKEN', '')
    POLYGON_API_KEY: str = os.getenv('POLYGON_API_KEY', '')
    
@dataclass
class AgentConfig:
    """Agent configuration parameters"""
    SHORT_TERM_TIMEFRAME: str = "1h"
    SWING_TIMEFRAME: str = "1d" 
    MACRO_TIMEFRAME: str = "1w"
    CONSENSUS_THRESHOLD: float = 0.6
    CONFIDENCE_THRESHOLD: float = 0.7
    
@dataclass
class RiskConfig:
    """Risk management configuration"""
    MAX_PORTFOLIO_RISK: float = 0.02  # 2% portfolio risk per trade
    MAX_CORRELATION: float = 0.7      # Max correlation between positions
    VAR_CONFIDENCE: float = 0.95      # 95% confidence VaR
    MAX_DRAWDOWN: float = 0.15        # 15% max drawdown
    
# Application settings
APP_TITLE = "Smart Multi-Agent AI Trading System"
APP_VERSION = "1.0.0"
DEBUG_MODE = os.getenv('DEBUG_MODE', 'False').lower() == 'true'

# Market data settings
MARKET_DATA_REFRESH_INTERVAL = 30  # seconds
SUPPORTED_EXCHANGES = ['NYSE', 'NASDAQ', 'AMEX']
DEFAULT_SYMBOLS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']

# LLM settings
DEFAULT_LLM_MODEL = "gpt-4"
MAX_TOKENS = 2000
TEMPERATURE = 0.3

# Database settings
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///trading_system.db')

# Logging settings
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Export configurations
trading_config = TradingConfig()
api_config = APIConfig()
agent_config = AgentConfig()
risk_config = RiskConfig()