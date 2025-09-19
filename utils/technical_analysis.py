# utils/technical_analysis.py
"""
Technical Analysis utilities for trading signals
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
import logging

class TechnicalAnalyzer:
    """
    Comprehensive technical analysis toolkit
    Provides various technical indicators and pattern recognition
    """
    
    def __init__(self):
        self.logger = logging.getLogger("TechnicalAnalyzer")
    
    def calculate_sma(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return prices.rolling(window=period).mean()
    
    def calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return prices.ewm(span=period, adjust=False).mean()
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        ema_fast = self.calculate_ema(prices, fast)
        ema_slow = self.calculate_ema(prices, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = self.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = self.calculate_sma(prices, period)
        std = prices.rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return upper_band, sma, lower_band
    
    def calculate_stochastic(self, data: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator"""
        high_roll = data['high'].rolling(window=period).max()
        low_roll = data['low'].rolling(window=period).min()
        
        stoch_k = 100 * ((data['close'] - low_roll) / (high_roll - low_roll))
        stoch_d = stoch_k.rolling(window=3).mean()
        
        return stoch_k, stoch_d
    
    def calculate_williams_r(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        high_roll = data['high'].rolling(window=period).max()
        low_roll = data['low'].rolling(window=period).min()
        
        williams_r = -100 * ((high_roll - data['close']) / (high_roll - low_roll))
        return williams_r
    
    def calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index (ADX)"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate Directional Movement
        dm_plus = high.diff()
        dm_minus = -low.diff()
        
        dm_plus[dm_plus < 0] = 0
        dm_minus[dm_minus < 0] = 0
        dm_plus[(dm_plus - dm_minus) <= 0] = 0
        dm_minus[(dm_minus - dm_plus) <= 0] = 0
        
        # Calculate smoothed averages
        atr = true_range.rolling(window=period).mean()
        di_plus = 100 * (dm_plus.rolling(window=period).mean() / atr)
        di_minus = 100 * (dm_minus.rolling(window=period).mean() / atr)
        
        # Calculate ADX
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def detect_support_resistance(self, data: pd.DataFrame, window: int = 20) -> Dict[str, List[float]]:
        """Detect support and resistance levels"""
        highs = data['high'].values
        lows = data['low'].values
        
        resistance_levels = []
        support_levels = []
        
        # Find local maxima and minima
        for i in range(window, len(highs) - window):
            # Check for resistance (local maxima)
            if highs[i] == max(highs[i-window:i+window+1]):
                resistance_levels.append(highs[i])
            
            # Check for support (local minima)
            if lows[i] == min(lows[i-window:i+window+1]):
                support_levels.append(lows[i])
        
        return {
            'resistance': resistance_levels,
            'support': support_levels
        }
    
    def identify_chart_patterns(self, data: pd.DataFrame) -> Dict[str, bool]:
        """Identify common chart patterns"""
        patterns = {
            'double_top': False,
            'double_bottom': False,
            'head_shoulders': False,
            'ascending_triangle': False,
            'descending_triangle': False,
            'bull_flag': False,
            'bear_flag': False
        }
        
        if len(data) < 50:
            return patterns
        
        highs = data['high'].values
        lows = data['low'].values
        closes = data['close'].values
        
        # Simple pattern detection (can be enhanced)
        try:
            # Double Top Detection
            recent_highs = highs[-20:]
            if len(recent_highs) >= 10:
                max_high = max(recent_highs)
                high_indices = [i for i, h in enumerate(recent_highs) if h > max_high * 0.98]
                if len(high_indices) >= 2 and (high_indices[-1] - high_indices[0]) > 5:
                    patterns['double_top'] = True
            
            # Double Bottom Detection
            recent_lows = lows[-20:]
            if len(recent_lows) >= 10:
                min_low = min(recent_lows)
                low_indices = [i for i, l in enumerate(recent_lows) if l < min_low * 1.02]
                if len(low_indices) >= 2 and (low_indices[-1] - low_indices[0]) > 5:
                    patterns['double_bottom'] = True
            
            # Bull Flag (simplified)
            if len(closes) >= 30:
                # Strong upward move followed by consolidation
                early_move = (closes[-30] - closes[-40]) / closes[-40] if len(closes) >= 40 else 0
                recent_volatility = np.std(closes[-10:]) / np.mean(closes[-10:])
                
                if early_move > 0.05 and recent_volatility < 0.02:
                    patterns['bull_flag'] = True
            
        except Exception as e:
            self.logger.warning(f"Pattern detection error: {e}")
        
        return patterns
    
    def calculate_fibonacci_levels(self, high_price: float, low_price: float) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels"""
        diff = high_price - low_price
        
        levels = {
            '0%': high_price,
            '23.6%': high_price - (0.236 * diff),
            '38.2%': high_price - (0.382 * diff),
            '50%': high_price - (0.5 * diff),
            '61.8%': high_price - (0.618 * diff),
            '78.6%': high_price - (0.786 * diff),
            '100%': low_price
        }
        
        return levels
    
    def calculate_pivot_points(self, high: float, low: float, close: float) -> Dict[str, float]:
        """Calculate pivot points and support/resistance levels"""
        pivot = (high + low + close) / 3
        
        levels = {
            'pivot': pivot,
            'r1': (2 * pivot) - low,
            'r2': pivot + (high - low),
            'r3': high + (2 * (pivot - low)),
            's1': (2 * pivot) - high,
            's2': pivot - (high - low),
            's3': low - (2 * (high - pivot))
        }
        
        return levels
    
    def calculate_volume_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate volume-based indicators"""
        volume = data['volume']
        close = data['close']
        
        # On-Balance Volume (OBV)
        obv = (volume * ((close > close.shift(1)).astype(int) * 2 - 1)).cumsum()
        
        # Volume Weighted Average Price (VWAP)
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        
        # Volume Rate of Change
        volume_roc = volume.pct_change(periods=10) * 100
        
        # Volume Moving Average
        volume_ma = volume.rolling(window=20).mean()
        
        return {
            'obv': obv,
            'vwap': vwap,
            'volume_roc': volume_roc,
            'volume_ma': volume_ma,
            'volume_ratio': volume / volume_ma
        }
    
    def calculate_momentum_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate momentum indicators"""
        close = data['close']
        
        # Rate of Change
        roc = close.pct_change(periods=10) * 100
        
        # Money Flow Index
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        money_flow = typical_price * data['volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        positive_flow_sum = positive_flow.rolling(window=14).sum()
        negative_flow_sum = negative_flow.rolling(window=14).sum()
        
        money_ratio = positive_flow_sum / negative_flow_sum
        mfi = 100 - (100 / (1 + money_ratio))
        
        # Commodity Channel Index
        cci_period = 20
        typical_price_ma = typical_price.rolling(window=cci_period).mean()
        mean_deviation = abs(typical_price - typical_price_ma).rolling(window=cci_period).mean()
        cci = (typical_price - typical_price_ma) / (0.015 * mean_deviation)
        
        return {
            'roc': roc,
            'mfi': mfi,
            'cci': cci
        }
    
    def get_trend_strength(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate overall trend strength"""
        close = data['close']
        
        # ADX for trend strength
        adx = self.calculate_adx(data)
        current_adx = adx.iloc[-1] if not adx.empty else 0
        
        # Price trend (linear regression slope)
        if len(close) >= 20:
            x = np.arange(len(close[-20:]))
            y = close[-20:].values
            slope = np.polyfit(x, y, 1)[0]
            trend_strength = abs(slope) / close.iloc[-1] * 100
        else:
            trend_strength = 0
        
        # Moving average alignment
        if len(close) >= 50:
            sma_10 = self.calculate_sma(close, 10).iloc[-1]
            sma_20 = self.calculate_sma(close, 20).iloc[-1]
            sma_50 = self.calculate_sma(close, 50).iloc[-1]
            
            ma_alignment = 1.0 if sma_10 > sma_20 > sma_50 else -1.0 if sma_10 < sma_20 < sma_50 else 0.0
        else:
            ma_alignment = 0.0
        
        return {
            'adx_strength': current_adx,
            'price_trend_strength': trend_strength,
            'ma_alignment': ma_alignment,
            'overall_strength': (current_adx + trend_strength + abs(ma_alignment) * 25) / 3
        }
    
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, str]:
        """Generate comprehensive trading signals"""
        signals = {}
        
        try:
            close = data['close']
            
            # RSI Signal
            rsi = self.calculate_rsi(close)
            current_rsi = rsi.iloc[-1] if not rsi.empty else 50
            
            if current_rsi < 30:
                signals['rsi'] = 'BUY'
            elif current_rsi > 70:
                signals['rsi'] = 'SELL'
            else:
                signals['rsi'] = 'HOLD'
            
            # MACD Signal
            macd_line, signal_line, histogram = self.calculate_macd(close)
            if not macd_line.empty and not signal_line.empty:
                if macd_line.iloc[-1] > signal_line.iloc[-1] and histogram.iloc[-1] > 0:
                    signals['macd'] = 'BUY'
                elif macd_line.iloc[-1] < signal_line.iloc[-1] and histogram.iloc[-1] < 0:
                    signals['macd'] = 'SELL'
                else:
                    signals['macd'] = 'HOLD'
            
            # Moving Average Signal
            if len(close) >= 50:
                sma_20 = self.calculate_sma(close, 20)
                sma_50 = self.calculate_sma(close, 50)
                
                if not sma_20.empty and not sma_50.empty:
                    if sma_20.iloc[-1] > sma_50.iloc[-1] and close.iloc[-1] > sma_20.iloc[-1]:
                        signals['ma'] = 'BUY'
                    elif sma_20.iloc[-1] < sma_50.iloc[-1] and close.iloc[-1] < sma_20.iloc[-1]:
                        signals['ma'] = 'SELL'
                    else:
                        signals['ma'] = 'HOLD'
            
            # Volume Signal
            volume_indicators = self.calculate_volume_indicators(data)
            volume_ratio = volume_indicators['volume_ratio'].iloc[-1] if not volume_indicators['volume_ratio'].empty else 1
            
            if volume_ratio > 1.5:
                signals['volume'] = 'STRONG'
            elif volume_ratio < 0.5:
                signals['volume'] = 'WEAK'
            else:
                signals['volume'] = 'NORMAL'
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            signals = {'error': 'CALCULATION_ERROR'}
        
        return signals