# agents/short_term_agent.py
"""
Short-term Trading Agent
Focuses on intraday and short-term price movements
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from datetime import datetime, timedelta
import asyncio
import logging

from .base_agent import BaseAgent, TradingSignal, MarketData
from utils.technical_analysis import TechnicalAnalyzer
from utils.llm_handler import LLMHandler

class ShortTermAgent(BaseAgent):
    """
    Short-term trading agent focused on scalping and day trading
    Uses technical indicators and momentum analysis
    """
    
    def __init__(self):
        super().__init__(name="Short-Term Agent", timeframe="1h")
        self.technical_analyzer = TechnicalAnalyzer()
        self.llm_handler = LLMHandler()
        self.strategy_type = "Momentum & Scalping"
        self.min_profit_target = 0.02  # 2% minimum profit target
        self.max_hold_time = timedelta(hours=4)  # Max 4 hours hold
        
    async def analyze(self, symbol: str, market_data: MarketData) -> TradingSignal:
        """
        Analyze symbol for short-term trading opportunities
        
        Args:
            symbol: Stock symbol to analyze
            market_data: Current market data
            
        Returns:
            TradingSignal: Short-term trading signal
        """
        try:
            # Get historical data for analysis
            historical_data = await self._get_historical_data(symbol, "1h", 100)
            
            # Perform technical analysis
            technical_signals = self._analyze_technical_indicators(historical_data)
            
            # Analyze volume patterns
            volume_analysis = self._analyze_volume_patterns(historical_data)
            
            # Check momentum indicators
            momentum_signals = self._analyze_momentum(historical_data)
            
            # Combine all signals
            combined_signal = self._combine_signals(
                technical_signals, 
                volume_analysis, 
                momentum_signals,
                market_data
            )
            
            # Get LLM insight for final validation
            llm_insight = await self._get_llm_analysis(symbol, combined_signal, market_data)
            
            # Generate final trading signal
            signal = self._generate_signal(symbol, combined_signal, llm_insight, market_data)
            
            self.log_signal(signal)
            return signal
            
        except Exception as e:
            self.logger.error(f"Error analyzing {symbol}: {e}")
            return self._create_hold_signal(symbol, f"Analysis error: {str(e)}")
    
    def _analyze_technical_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze technical indicators for short-term signals"""
        signals = {}
        
        try:
            # RSI analysis
            rsi = self.technical_analyzer.calculate_rsi(data['close'], period=14)
            current_rsi = rsi.iloc[-1]
            signals['rsi'] = {
                'value': current_rsi,
                'signal': 'BUY' if current_rsi < 30 else 'SELL' if current_rsi > 70 else 'HOLD',
                'strength': abs(50 - current_rsi) / 50
            }
            
            # MACD analysis
            macd_line, signal_line, histogram = self.technical_analyzer.calculate_macd(data['close'])
            macd_signal = 'BUY' if macd_line.iloc[-1] > signal_line.iloc[-1] else 'SELL'
            signals['macd'] = {
                'signal': macd_signal,
                'histogram': histogram.iloc[-1],
                'strength': abs(histogram.iloc[-1]) / data['close'].iloc[-1] * 100
            }
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self.technical_analyzer.calculate_bollinger_bands(data['close'])
            current_price = data['close'].iloc[-1]
            bb_position = (current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
            
            signals['bollinger'] = {
                'position': bb_position,
                'signal': 'SELL' if bb_position > 0.8 else 'BUY' if bb_position < 0.2 else 'HOLD',
                'strength': abs(0.5 - bb_position) * 2
            }
            
            # Moving Average Crossover
            ema_fast = self.technical_analyzer.calculate_ema(data['close'], 9)
            ema_slow = self.technical_analyzer.calculate_ema(data['close'], 21)
            
            ma_signal = 'BUY' if ema_fast.iloc[-1] > ema_slow.iloc[-1] else 'SELL'
            ma_strength = abs(ema_fast.iloc[-1] - ema_slow.iloc[-1]) / ema_slow.iloc[-1]
            
            signals['moving_average'] = {
                'signal': ma_signal,
                'strength': ma_strength,
                'fast_ma': ema_fast.iloc[-1],
                'slow_ma': ema_slow.iloc[-1]
            }
            
        except Exception as e:
            self.logger.error(f"Technical analysis error: {e}")
            
        return signals
    
    def _analyze_volume_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume patterns for confirmation"""
        try:
            # Volume moving average
            volume_ma = data['volume'].rolling(window=20).mean()
            current_volume = data['volume'].iloc[-1]
            avg_volume = volume_ma.iloc[-1]
            
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Price-Volume correlation
            price_change = data['close'].pct_change()
            volume_correlation = price_change.corr(data['volume'].pct_change())
            
            return {
                'volume_ratio': volume_ratio,
                'volume_signal': 'STRONG' if volume_ratio > 1.5 else 'WEAK' if volume_ratio < 0.5 else 'NORMAL',
                'price_volume_correlation': volume_correlation,
                'confirmation': volume_ratio > 1.2  # Volume confirms move
            }
            
        except Exception as e:
            self.logger.error(f"Volume analysis error: {e}")
            return {'confirmation': False}
    
    def _analyze_momentum(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze momentum indicators"""
        try:
            # Rate of Change (ROC)
            roc = ((data['close'] - data['close'].shift(10)) / data['close'].shift(10) * 100)
            current_roc = roc.iloc[-1]
            
            # Stochastic Oscillator
            stoch_k, stoch_d = self.technical_analyzer.calculate_stochastic(data, period=14)
            
            # Williams %R
            williams_r = self.technical_analyzer.calculate_williams_r(data, period=14)
            
            momentum_score = 0
            if current_roc > 2:
                momentum_score += 1
            elif current_roc < -2:
                momentum_score -= 1
                
            if stoch_k.iloc[-1] > 80:
                momentum_score -= 1
            elif stoch_k.iloc[-1] < 20:
                momentum_score += 1
                
            return {
                'roc': current_roc,
                'stochastic_k': stoch_k.iloc[-1],
                'stochastic_d': stoch_d.iloc[-1],
                'williams_r': williams_r.iloc[-1],
                'momentum_score': momentum_score,
                'signal': 'BUY' if momentum_score > 0 else 'SELL' if momentum_score < 0 else 'HOLD'
            }
            
        except Exception as e:
            self.logger.error(f"Momentum analysis error: {e}")
            return {'momentum_score': 0, 'signal': 'HOLD'}
    
    def _combine_signals(self, technical: Dict, volume: Dict, momentum: Dict, market_data: MarketData) -> Dict[str, Any]:
        """Combine all analysis signals into overall assessment"""
        
        buy_votes = 0
        sell_votes = 0
        total_strength = 0
        
        # Technical indicators votes
        for indicator, data in technical.items():
            if data.get('signal') == 'BUY':
                buy_votes += 1
                total_strength += data.get('strength', 0.5)
            elif data.get('signal') == 'SELL':
                sell_votes += 1
                total_strength += data.get('strength', 0.5)
        
        # Momentum vote
        momentum_signal = momentum.get('signal', 'HOLD')
        if momentum_signal == 'BUY':
            buy_votes += 1
        elif momentum_signal == 'SELL':
            sell_votes += 1
        
        # Volume confirmation
        volume_confirmation = volume.get('confirmation', False)
        if volume_confirmation:
            if buy_votes > sell_votes:
                buy_votes += 0.5
            elif sell_votes > buy_votes:
                sell_votes += 0.5
        
        # Determine overall signal
        total_votes = buy_votes + sell_votes
        if total_votes == 0:
            overall_signal = 'HOLD'
            confidence = 0.3
        else:
            if buy_votes > sell_votes:
                overall_signal = 'BUY'
                confidence = buy_votes / (buy_votes + sell_votes)
            elif sell_votes > buy_votes:
                overall_signal = 'SELL'
                confidence = sell_votes / (buy_votes + sell_votes)
            else:
                overall_signal = 'HOLD'
                confidence = 0.4
        
        # Adjust confidence based on signal strength
        avg_strength = total_strength / max(total_votes, 1)
        confidence = min(confidence * (0.5 + avg_strength), 0.95)
        
        return {
            'signal': overall_signal,
            'confidence': confidence,
            'buy_votes': buy_votes,
            'sell_votes': sell_votes,
            'technical_analysis': technical,
            'volume_analysis': volume,
            'momentum_analysis': momentum,
            'volume_confirmation': volume_confirmation
        }
    
    async def _get_llm_analysis(self, symbol: str, signals: Dict, market_data: MarketData) -> str:
        """Get LLM analysis for additional insight"""
        try:
            prompt = f"""
            As a short-term trading expert, analyze this data for {symbol}:
            
            Current Price: ${market_data.price:.2f}
            Technical Signal: {signals['signal']} (Confidence: {signals['confidence']:.1%})
            Volume Confirmation: {signals['volume_confirmation']}
            
            Technical Analysis Summary:
            - RSI: {signals['technical_analysis'].get('rsi', {}).get('value', 'N/A')}
            - MACD Signal: {signals['technical_analysis'].get('macd', {}).get('signal', 'N/A')}
            - Bollinger Position: {signals['technical_analysis'].get('bollinger', {}).get('position', 'N/A')}
            - Moving Average: {signals['technical_analysis'].get('moving_average', {}).get('signal', 'N/A')}
            
            Provide a brief analysis focusing on:
            1. Short-term price direction (next 1-4 hours)
            2. Key risk factors
            3. Entry/exit strategy
            
            Keep response under 200 words.
            """
            
            analysis = await self.llm_handler.get_analysis(prompt)
            return analysis
            
        except Exception as e:
            self.logger.error(f"LLM analysis error: {e}")
            return "Technical analysis suggests mixed signals. Consider market conditions and volume confirmation."
    
    def _generate_signal(self, symbol: str, combined_signal: Dict, llm_insight: str, market_data: MarketData) -> TradingSignal:
        """Generate final trading signal"""
        
        action = combined_signal['signal']
        confidence = combined_signal['confidence']
        
        # Create reasoning
        reasoning = f"Short-term analysis: {action} signal with {confidence:.1%} confidence. "
        reasoning += f"Technical votes - Buy: {combined_signal['buy_votes']}, Sell: {combined_signal['sell_votes']}. "
        reasoning += f"Volume confirmation: {combined_signal['volume_confirmation']}. "
        reasoning += f"AI insight: {llm_insight[:100]}..."
        
        # Calculate target and stop loss for buy signals
        target_price = None
        stop_loss = None
        
        if action == 'BUY':
            target_price = market_data.price * (1 + self.min_profit_target)
            stop_loss = market_data.price * (1 - 0.015)  # 1.5% stop loss
        elif action == 'SELL':
            target_price = market_data.price * (1 - self.min_profit_target)
            stop_loss = market_data.price * (1 + 0.015)
        
        return TradingSignal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            target_price=target_price,
            stop_loss=stop_loss,
            timeframe=self.timeframe,
            timestamp=datetime.now()
        )
    
    def _create_hold_signal(self, symbol: str, reason: str) -> TradingSignal:
        """Create a HOLD signal with given reason"""
        return TradingSignal(
            symbol=symbol,
            action='HOLD',
            confidence=0.3,
            reasoning=f"Short-term hold: {reason}",
            timeframe=self.timeframe,
            timestamp=datetime.now()
        )
    
    async def _get_historical_data(self, symbol: str, timeframe: str, periods: int) -> pd.DataFrame:
        """Get historical market data (mock implementation)"""
        # This would typically fetch real market data
        # For now, return mock data
        dates = pd.date_range(end=datetime.now(), periods=periods, freq='H')
        
        # Generate realistic price data
        np.random.seed(hash(symbol) % 2**32)
        base_price = 100 + (hash(symbol) % 200)
        
        prices = []
        price = base_price
        for _ in range(periods):
            change = np.random.normal(0, 0.02)  # 2% volatility
            price = price * (1 + change)
            prices.append(max(price, 1))  # Minimum price of $1
        
        volumes = np.random.randint(1000000, 10000000, periods)
        
        # Create OHLC data
        data = []
        for i, price in enumerate(prices):
            high = price * (1 + abs(np.random.normal(0, 0.01)))
            low = price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = prices[i-1] if i > 0 else price
            
            data.append({
                'timestamp': dates[i],
                'open': open_price,
                'high': high,
                'low': low,
                'close': price,
                'volume': volumes[i]
            })
        
        return pd.DataFrame(data)
    
    def get_strategy_description(self) -> str:
        """Return strategy description"""
        return """
        Short-Term Trading Strategy:
        - Timeframe: 1-4 hours
        - Focus: Technical momentum and volume analysis
        - Indicators: RSI, MACD, Bollinger Bands, Moving Averages
        - Risk: 1.5% stop loss, 2% profit target
        - Volume confirmation required for stronger signals
        - AI-enhanced pattern recognition
        """