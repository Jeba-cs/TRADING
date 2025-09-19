# models/trading_signals.py
from dataclasses import dataclass
from datetime import datetime

@dataclass
class TradingSignal:
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float
    reasoning: str
    timestamp: datetime = datetime.now()
