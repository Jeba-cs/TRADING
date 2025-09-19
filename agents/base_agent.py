# agents/base_agent.py
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
