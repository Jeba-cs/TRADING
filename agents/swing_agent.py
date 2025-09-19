# agents/swing_agent.py
"""
Swing Trading Agent
Focuses on medium-term price movements (1-10 days)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from datetime import datetime, timedelta
import asyncio
import logging

from .base_agent import BaseAgent, TradingSignal, MarketData
from utils.technical_analysis import TechnicalAnalyzer
from utils.fundamental_analysis import FundamentalAnalyzer
from utils.llm_handler import LLMHandler

class SwingAgent(BaseAgent):
    """
    Swing trading agent focused on capturing medium-term price swings
    Combines technical and fundamental analysis
    """
    
    def __init__(self):
        super().__init__(name="Swing Agent", timeframe="1d")
        self.technical_analyzer = TechnicalAnalyzer()
        self.fundamental_analyzer = FundamentalAnalyzer()
        self.llm_handler = LLMHandler()
        self.strategy_type = "Swing Trading"
        self.min_profit_target = 0.05  # 5% minimum profit target
        self.max_hold_time = timedelta(days=10)  # Max 10 days hold
        self.lookback_period = 50  # Days of data to analyze
        
    async def analyze(self, symbol: str, market_data: MarketData) -> TradingSignal:
        """
        Analyze symbol for swing trading opportunities
        
        Args:
            symbol: Stock symbol to analyze
            market_data: Current market data
            
        Returns:
            TradingSignal: Swing trading signal
        """
        try:
            # Get daily historical data
            daily_data = await self._get_historical_data(symbol, "1d", self.lookback_period)
            
            # Technical analysis
            technical_signals = self._analyze_swing_technicals(daily_data)
            
            # Trend analysis
            trend_analysis = self._analyze_trend_structure(daily_data)
            
            # Support/Resistance levels
            sr_levels = self._identify_support_resistance(daily_data)
            
            # Pattern recognition
            patterns = self._detect_chart_patterns(daily_data)
            
            # Fundamental check (basic)
            fundamental_score = await self._quick_fundamental_check(symbol)
            
            # Market sentiment (if available)
            sentiment_score = await self._get_market_sentiment(symbol)
            
            # Combine all analyses
            combined_signal = self._combine_swing_signals(
                technical_signals,
                trend_analysis,
                sr_levels,
                patterns,
                fundamental_score,
                sentiment_score,
                market_data
            )
            
            # Get LLM validation
            llm_insight = await self._get_llm_swing_analysis(symbol, combined_signal, market_data)
            
            # Generate final signal
            signal = self._generate_swing_signal(symbol, combined_signal, llm_insight, market_data)
            
            self.log_signal(signal)
            return signal
            
        except Exception as e:
            self.logger.error(f"Error in swing analysis for {symbol}: {e}")
            return self._create_hold_signal(symbol, f"Analysis error: {str(e)}")
    
    def _analyze_swing_technicals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze technical indicators for swing trading"""
        signals = {}
        
        try:
            # Longer-period moving averages
            sma_20 = self.technical_analyzer.calculate_sma(data['close'], 20)
            sma_50 = self.technical_analyzer.calculate_sma(data['close'], 50)
            ema_12 = self.technical_analyzer.calculate_ema(data['close'], 12)
            ema_26 = self.technical_analyzer.calculate_ema(data['close'], 26)
            
            current_price = data['close'].iloc[-1]
            
            # Moving average signals
            ma_signal = 'BUY' if (current_price > sma_20.iloc[-1] and 
                                 sma_20.iloc[-1] > sma_50.iloc[-1]) else 'SELL'
            
            signals['moving_averages'] = {
                'signal': ma_signal,
                'sma_20': sma_20.iloc[-1],
                'sma_50': sma_50.iloc[-1],
                'price_vs_sma20': (current_price - sma_20.iloc[-1]) / sma_20.iloc[-1],
                'strength': abs((sma_20.iloc[-1] - sma_50.iloc[-1]) / sma_50.iloc[-1])
            }
            
            # RSI with swing levels
            rsi = self.technical_analyzer.calculate_rsi(data['close'], period=21)  # Longer period for swing
            current_rsi = rsi.iloc[-1]
            
            if current_rsi < 35:
                rsi_signal = 'BUY'
            elif current_rsi > 65:
                rsi_signal = 'SELL'
            else:
                rsi_signal = 'HOLD'
            
            signals['rsi'] = {
                'value': current_rsi,
                'signal': rsi_signal,
                'strength': abs(50 - current_rsi) / 50
            }
            
            # MACD for trend confirmation
            macd_line, signal_line, histogram = self.technical_analyzer.calculate_macd(
                data['close'], fast=12, slow=26, signal=9
            )
            
            macd_signal = 'BUY' if (macd_line.iloc[-1] > signal_line.iloc[-1] and 
                                   histogram.iloc[-1] > 0) else 'SELL'
            
            signals['macd'] = {
                'signal': macd_signal,
                'macd_line': macd_line.iloc[-1],
                'signal_line': signal_line.iloc[-1],
                'histogram': histogram.iloc[-1],
                'strength': abs(histogram.iloc[-1]) / current_price * 100
            }
            
            # ADX for trend strength
            adx = self.technical_analyzer.calculate_adx(data, period=14)
            current_adx = adx.iloc[-1]
            
            trend_strength = 'STRONG' if current_adx > 25 else 'WEAK' if current_adx < 20 else 'MODERATE'
            
            signals['adx'] = {
                'value': current_adx,
                'trend_strength': trend_strength,
                'trending': current_adx > 25
            }
            
            # Stochastic for swing entries
            stoch_k, stoch_d = self.technical_analyzer.calculate_stochastic(data, period=14)
            
            if stoch_k.iloc[-1] < 20 and stoch_k.iloc[-1] > stoch_d.iloc[-1]:
                stoch_signal = 'BUY'
            elif stoch_k.iloc[-1] > 80 and stoch_k.iloc[-1] < stoch_d.iloc[-1]:
                stoch_signal = 'SELL'
            else:
                stoch_signal = 'HOLD'
            
            signals['stochastic'] = {
                'k': stoch_k.iloc[-1],
                'd': stoch_d.iloc[-1],
                'signal': stoch_signal
            }
            
        except Exception as e:
            self.logger.error(f"Technical analysis error: {e}")
            
        return signals
    
    def _analyze_trend_structure(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze overall trend structure"""
        try:
            # Calculate trend using multiple timeframes
            short_trend = self._calculate_trend(data['close'].tail(10))  # 10 days
            medium_trend = self._calculate_trend(data['close'].tail(20))  # 20 days
            long_trend = self._calculate_trend(data['close'].tail(50))   # 50 days
            
            # Higher highs and higher lows analysis
            highs = data['high'].tail(20)
            lows = data['low'].tail(20)
            
            recent_highs = highs.tail(5)
            recent_lows = lows.tail(5)
            
            higher_highs = recent_highs.max() > highs.head(15).max()
            higher_lows = recent_lows.min() > lows.head(15).min()
            
            if higher_highs and higher_lows:
                trend_structure = 'UPTREND'
            elif not higher_highs and not higher_lows:
                trend_structure = 'DOWNTREND'
            else:
                trend_structure = 'SIDEWAYS'
            
            # Trend consistency score
            trend_scores = [short_trend, medium_trend, long_trend]
            trend_consistency = len([t for t in trend_scores if t == trend_scores[0]]) / len(trend_scores)
            
            return {
                'short_trend': short_trend,
                'medium_trend': medium_trend,
                'long_trend': long_trend,
                'trend_structure': trend_structure,
                'trend_consistency': trend_consistency,
                'higher_highs': higher_highs,
                'higher_lows': higher_lows,
                'signal': 'BUY' if trend_structure == 'UPTREND' else 'SELL' if trend_structure == 'DOWNTREND' else 'HOLD'
            }
            
        except Exception as e:
            self.logger.error(f"Trend analysis error: {e}")
            return {'signal': 'HOLD', 'trend_consistency': 0.0}
    
    def _calculate_trend(self, prices: pd.Series) -> str:
        """Calculate trend direction for given price series"""
        if len(prices) < 2:
            return 'NEUTRAL'
            
        # Linear regression to determine trend
        x = np.arange(len(prices))
        slope = np.polyfit(x, prices.values, 1)[0]
        
        if slope > prices.mean() * 0.001:  # 0.1% threshold
            return 'UP'
        elif slope < -prices.mean() * 0.001:
            return 'DOWN'
        else:
            return 'NEUTRAL'
    
    def _identify_support_resistance(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Identify key support and resistance levels"""
        try:
            highs = data['high'].values
            lows = data['low'].values
            close = data['close'].values
            
            # Find local maxima and minima
            resistance_levels = []
            support_levels = []
            
            # Simple peak/trough detection
            for i in range(2, len(highs) - 2):
                # Resistance (peaks)
                if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and 
                    highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                    resistance_levels.append(highs[i])
                
                # Support (troughs)
                if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and 
                    lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                    support_levels.append(lows[i])
            
            current_price = close[-1]
            
            # Find nearest levels
            if resistance_levels:
                nearest_resistance = min([r for r in resistance_levels if r > current_price], 
                                       default=max(resistance_levels))
            else:
                nearest_resistance = current_price * 1.05
            
            if support_levels:
                nearest_support = max([s for s in support_levels if s < current_price], 
                                    default=min(support_levels))
            else:
                nearest_support = current_price * 0.95
            
            # Distance to levels
            resistance_distance = (nearest_resistance - current_price) / current_price
            support_distance = (current_price - nearest_support) / current_price
            
            return {
                'nearest_resistance': nearest_resistance,
                'nearest_support': nearest_support,
                'resistance_distance': resistance_distance,
                'support_distance': support_distance,
                'at_support': support_distance < 0.02,  # Within 2% of support
                'at_resistance': resistance_distance < 0.02,  # Within 2% of resistance
                'all_resistance_levels': resistance_levels,
                'all_support_levels': support_levels
            }
            
        except Exception as e:
            self.logger.error(f"S/R analysis error: {e}")
            return {
                'nearest_resistance': data['close'].iloc[-1] * 1.05,
                'nearest_support': data['close'].iloc[-1] * 0.95,
                'at_support': False,
                'at_resistance': False
            }
    
    def _detect_chart_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Simple chart pattern detection"""
        try:
            patterns_detected = []
            
            # Double bottom pattern (simplified)
            lows = data['low'].tail(20)
            if len(lows) >= 10:
                recent_low = lows.iloc[-5:].min()
                earlier_low = lows.iloc[-15:-10].min()
                
                if abs(recent_low - earlier_low) / earlier_low < 0.02:  # Within 2%
                    patterns_detected.append('DOUBLE_BOTTOM')
            
            # Ascending triangle (simplified)
            highs = data['high'].tail(15)
            if len(highs) >= 10:
                recent_highs = highs.tail(5)
                if recent_highs.std() / recent_highs.mean() < 0.01:  # Low variance
                    patterns_detected.append('ASCENDING_TRIANGLE')
            
            # Head and shoulders (very simplified)
            if len(data) >= 20:
                recent_highs = data['high'].tail(20)
                if len(recent_highs) >= 15:
                    max_idx = recent_highs.idxmax()
                    left_shoulder = recent_highs.iloc[:5].max()
                    head = recent_highs.max()
                    right_shoulder = recent_highs.iloc[-5:].max()
                    
                    if (head > left_shoulder * 1.02 and head > right_shoulder * 1.02 and
                        abs(left_shoulder - right_shoulder) / left_shoulder < 0.05):
                        patterns_detected.append('HEAD_AND_SHOULDERS')
            
            return {
                'patterns': patterns_detected,
                'bullish_patterns': len([p for p in patterns_detected if p in ['DOUBLE_BOTTOM', 'ASCENDING_TRIANGLE']]),
                'bearish_patterns': len([p for p in patterns_detected if p in ['HEAD_AND_SHOULDERS']]),
                'pattern_signal': ('BUY' if 'DOUBLE_BOTTOM' in patterns_detected or 'ASCENDING_TRIANGLE' in patterns_detected
                                 else 'SELL' if 'HEAD_AND_SHOULDERS' in patterns_detected else 'HOLD')
            }
            
        except Exception as e:
            self.logger.error(f"Pattern detection error: {e}")
            return {'patterns': [], 'pattern_signal': 'HOLD'}
    
    async def _quick_fundamental_check(self, symbol: str) -> float:
        """Quick fundamental analysis score"""
        try:
            # This would integrate with fundamental data APIs
            # For now, return a mock score based on symbol
            return 0.6 + (hash(symbol) % 100) / 250  # Score between 0.2-1.0
        except Exception:
            return 0.5  # Neutral score
    
    async def _get_market_sentiment(self, symbol: str) -> float:
        """Get market sentiment score"""
        try:
            # This would integrate with news/sentiment APIs
            # For now, return a mock sentiment
            return 0.4 + (hash(symbol) % 200) / 500  # Score between 0.0-0.8
        except Exception:
            return 0.5  # Neutral sentiment
    
    def _combine_swing_signals(self, technical: Dict, trend: Dict, sr_levels: Dict, 
                              patterns: Dict, fundamental: float, sentiment: float, 
                              market_data: MarketData) -> Dict[str, Any]:
        """Combine all swing analysis signals"""
        
        buy_score = 0
        sell_score = 0
        
        # Technical signals (40% weight)
        technical_weight = 0.4
        tech_signals = technical.values()
        for signal_data in tech_signals:
            if isinstance(signal_data, dict) and 'signal' in signal_data:
                if signal_data['signal'] == 'BUY':
                    buy_score += technical_weight / len(tech_signals)
                elif signal_data['signal'] == 'SELL':
                    sell_score += technical_weight / len(tech_signals)
        
        # Trend analysis (25% weight)
        trend_weight = 0.25
        if trend['signal'] == 'BUY':
            buy_score += trend_weight
        elif trend['signal'] == 'SELL':
            sell_score += trend_weight
        
        # Support/Resistance (15% weight)
        sr_weight = 0.15
        if sr_levels['at_support']:
            buy_score += sr_weight
        elif sr_levels['at_resistance']:
            sell_score += sr_weight
        
        # Chart patterns (10% weight)
        pattern_weight = 0.10
        if patterns['pattern_signal'] == 'BUY':
            buy_score += pattern_weight
        elif patterns['pattern_signal'] == 'SELL':
            sell_score += pattern_weight
        
        # Fundamental score (5% weight)
        fund_weight = 0.05
        if fundamental > 0.6:
            buy_score += fund_weight * (fundamental - 0.5) * 2
        elif fundamental < 0.4:
            sell_score += fund_weight * (0.5 - fundamental) * 2
        
        # Sentiment (5% weight)
        sent_weight = 0.05
        if sentiment > 0.6:
            buy_score += sent_weight * (sentiment - 0.5) * 2
        elif sentiment < 0.4:
            sell_score += sent_weight * (0.5 - sentiment) * 2
        
        # Determine overall signal
        total_score = buy_score + sell_score
        if total_score == 0:
            overall_signal = 'HOLD'
            confidence = 0.3
        else:
            if buy_score > sell_score:
                overall_signal = 'BUY'
                confidence = buy_score / max(buy_score + sell_score, 1)
            elif sell_score > buy_score:
                overall_signal = 'SELL'
                confidence = sell_score / max(buy_score + sell_score, 1)
            else:
                overall_signal = 'HOLD'
                confidence = 0.4
        
        # Apply trend consistency multiplier
        confidence *= trend.get('trend_consistency', 0.5)
        confidence = min(confidence, 0.95)
        
        return {
            'signal': overall_signal,
            'confidence': confidence,
            'buy_score': buy_score,
            'sell_score': sell_score,
            'components': {
                'technical': technical,
                'trend': trend,
                'support_resistance': sr_levels,
                'patterns': patterns,
                'fundamental_score': fundamental,
                'sentiment_score': sentiment
            }
        }
    
    async def _get_llm_swing_analysis(self, symbol: str, signals: Dict, market_data: MarketData) -> str:
        """Get LLM analysis for swing trading context"""
        try:
            prompt = f"""
            As a swing trading expert, analyze this data for {symbol}:
            
            Current Price: ${market_data.price:.2f}
            Overall Signal: {signals['signal']} (Confidence: {signals['confidence']:.1%})
            
            Technical Analysis:
            - Trend: {signals['components']['trend']['trend_structure']}
            - At Support: {signals['components']['support_resistance']['at_support']}
            - At Resistance: {signals['components']['support_resistance']['at_resistance']}
            - Patterns: {signals['components']['patterns']['patterns']}
            
            Provide swing trading analysis focusing on:
            1. Medium-term direction (1-10 days)
            2. Risk/reward ratio
            3. Optimal entry and exit points
            4. Position sizing recommendations
            
            Keep response under 200 words.
            """
            
            analysis = await self.llm_handler.get_analysis(prompt)
            return analysis
            
        except Exception as e:
            self.logger.error(f"LLM swing analysis error: {e}")
            return "Swing analysis shows mixed signals. Consider waiting for clearer trend confirmation."
    
    def _generate_swing_signal(self, symbol: str, combined_signal: Dict, llm_insight: str, market_data: MarketData) -> TradingSignal:
        """Generate final swing trading signal"""
        
        action = combined_signal['signal']
        confidence = combined_signal['confidence']
        
        # Create detailed reasoning
        reasoning = f"Swing analysis: {action} signal with {confidence:.1%} confidence. "
        reasoning += f"Buy score: {combined_signal['buy_score']:.2f}, Sell score: {combined_signal['sell_score']:.2f}. "
        
        # Add component insights
        trend = combined_signal['components']['trend']
        reasoning += f"Trend: {trend['trend_structure']} (consistency: {trend['trend_consistency']:.1%}). "
        
        sr = combined_signal['components']['support_resistance']
        if sr['at_support']:
            reasoning += "Near support level. "
        elif sr['at_resistance']:
            reasoning += "Near resistance level. "
        
        patterns = combined_signal['components']['patterns']['patterns']
        if patterns:
            reasoning += f"Patterns: {', '.join(patterns)}. "
        
        reasoning += f"AI insight: {llm_insight[:100]}..."
        
        # Calculate target and stop loss
        target_price = None
        stop_loss = None
        
        if action == 'BUY':
            # Target based on resistance level or minimum profit
            resistance = sr.get('nearest_resistance', market_data.price * 1.05)
            min_target = market_data.price * (1 + self.min_profit_target)
            target_price = max(resistance * 0.95, min_target)  # 5% below resistance or min target
            
            # Stop loss at support or 3% below
            support = sr.get('nearest_support', market_data.price * 0.97)
            stop_loss = min(support * 1.01, market_data.price * 0.97)  # 1% above support or 3% stop
            
        elif action == 'SELL':
            # Target based on support level
            support = sr.get('nearest_support', market_data.price * 0.95)
            min_target = market_data.price * (1 - self.min_profit_target)
            target_price = min(support * 1.05, min_target)
            
            # Stop loss at resistance
            resistance = sr.get('nearest_resistance', market_data.price * 1.03)
            stop_loss = max(resistance * 0.99, market_data.price * 1.03)
        
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
            reasoning=f"Swing hold: {reason}",
            timeframe=self.timeframe,
            timestamp=datetime.now()
        )
    
    async def _get_historical_data(self, symbol: str, timeframe: str, periods: int) -> pd.DataFrame:
        """Get historical market data for swing analysis"""
        # Mock implementation - in production, would fetch real data
        dates = pd.date_range(end=datetime.now(), periods=periods, freq='D')
        
        np.random.seed(hash(symbol) % 2**32)
        base_price = 50 + (hash(symbol) % 100)
        
        prices = []
        price = base_price
        for _ in range(periods):
            # Add some trend and mean reversion
            trend = 0.001 if len(prices) < periods//2 else -0.001
            change = np.random.normal(trend, 0.025)  # 2.5% daily volatility with trend
            price = price * (1 + change)
            prices.append(max(price, 1))
        
        volumes = np.random.randint(500000, 5000000, periods)
        
        data = []
        for i, price in enumerate(prices):
            daily_volatility = 0.02
            high = price * (1 + abs(np.random.normal(0, daily_volatility/2)))
            low = price * (1 - abs(np.random.normal(0, daily_volatility/2)))
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
        """Return swing strategy description"""
        return """
        Swing Trading Strategy:
        - Timeframe: 1-10 days
        - Focus: Technical patterns, trend structure, and S/R levels
        - Indicators: Moving averages, RSI, MACD, ADX, Stochastic
        - Risk: 3% stop loss, 5%+ profit target
        - Pattern recognition and fundamental confirmation
        - Support/resistance level analysis
        - AI-enhanced decision validation
        """