# agents/base_agent.py
"""
Base Agent class for all trading agents
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

@dataclass
class TradingSignal:
    """Trading signal data structure"""
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float  # 0.0 to 1.0
    reasoning: str
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    position_size: Optional[float] = None
    timeframe: str = "1d"
    timestamp: datetime = datetime.now()

@dataclass
class MarketData:
    """Market data structure"""
    symbol: str
    price: float
    volume: int
    high: float
    low: float
    open: float
    close: float
    timestamp: datetime
    
class BaseAgent(ABC):
    """
    Base class for all trading agents
    Provides common functionality and interface
    """
    
    def __init__(self, name: str, timeframe: str = "1d"):
        self.name = name
        self.timeframe = timeframe
        self.logger = logging.getLogger(f"Agent.{name}")
        self.is_active = True
        self.confidence_threshold = 0.6
        self.max_positions = 5
        self.risk_tolerance = 0.02  # 2% risk per trade
        
    @abstractmethod
    async def analyze(self, symbol: str, market_data: MarketData) -> TradingSignal:
        """
        Analyze market data and generate trading signal
        
        Args:
            symbol: Stock symbol to analyze
            market_data: Current market data
            
        Returns:
            TradingSignal: Generated trading signal
        """
        pass
    
    @abstractmethod
    def get_strategy_description(self) -> str:
        """Return description of trading strategy"""
        pass
    
    def validate_signal(self, signal: TradingSignal) -> bool:
        """
        Validate trading signal before execution
        
        Args:
            signal: Trading signal to validate
            
        Returns:
            bool: True if signal is valid
        """
        if not signal:
            return False
            
        # Check confidence threshold
        if signal.confidence < self.confidence_threshold:
            self.logger.warning(f"Signal confidence {signal.confidence} below threshold {self.confidence_threshold}")
            return False
            
        # Check valid action
        if signal.action not in ['BUY', 'SELL', 'HOLD']:
            self.logger.error(f"Invalid action: {signal.action}")
            return False
            
        # Check valid symbol
        if not signal.symbol or len(signal.symbol) < 1:
            self.logger.error("Invalid symbol")
            return False
            
        return True
    
    def calculate_position_size(self, symbol: str, price: float, portfolio_value: float) -> float:
        """
        Calculate appropriate position size based on risk management
        
        Args:
            symbol: Stock symbol
            price: Stock price
            portfolio_value: Total portfolio value
            
        Returns:
            float: Position size in shares
        """
        try:
            # Risk-based position sizing
            risk_amount = portfolio_value * self.risk_tolerance
            
            # Assume 3% stop loss for position sizing
            stop_loss_pct = 0.03
            shares = risk_amount / (price * stop_loss_pct)
            
            # Don't exceed 10% of portfolio
            max_position_value = portfolio_value * 0.10
            max_shares = max_position_value / price
            
            return min(shares, max_shares)
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get agent performance metrics
        
        Returns:
            Dict: Performance metrics
        """
        return {
            'name': self.name,
            'timeframe': self.timeframe,
            'is_active': self.is_active,
            'confidence_threshold': self.confidence_threshold,
            'risk_tolerance': self.risk_tolerance,
            'total_signals': getattr(self, 'total_signals', 0),
            'successful_signals': getattr(self, 'successful_signals', 0),
            'win_rate': getattr(self, 'win_rate', 0.0)
        }
    
    def update_performance(self, signal_success: bool):
        """
        Update performance metrics
        
        Args:
            signal_success: Whether the signal was successful
        """
        if not hasattr(self, 'total_signals'):
            self.total_signals = 0
            self.successful_signals = 0
            
        self.total_signals += 1
        if signal_success:
            self.successful_signals += 1
            
        self.win_rate = self.successful_signals / self.total_signals if self.total_signals > 0 else 0.0
    
    def log_signal(self, signal: TradingSignal):
        """
        Log trading signal for debugging and analysis
        
        Args:
            signal: Trading signal to log
        """
        self.logger.info(
            f"Generated signal: {signal.symbol} {signal.action} "
            f"(Confidence: {signal.confidence:.2%}, Reasoning: {signal.reasoning[:100]}...)"
        )