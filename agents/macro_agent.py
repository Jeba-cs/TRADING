# agents/macro_agent.py
"""
Macro Trading Agent
Focuses on long-term trends and macro-economic factors
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from datetime import datetime, timedelta
import asyncio
import logging

from .base_agent import BaseAgent, TradingSignal, MarketData
from utils.fundamental_analysis import FundamentalAnalyzer
from utils.llm_handler import LLMHandler

class MacroAgent(BaseAgent):
    """
    Macro trading agent focused on long-term trends and economic factors
    Uses fundamental analysis, sector rotation, and macro indicators
    """
    
    def __init__(self):
        super().__init__(name="Macro Agent", timeframe="1w")
        self.fundamental_analyzer = FundamentalAnalyzer()
        self.llm_handler = LLMHandler()
        self.strategy_type = "Macro & Long-term Trends"
        self.min_profit_target = 0.15  # 15% minimum profit target
        self.max_hold_time = timedelta(days=180)  # Max 6 months hold
        self.lookback_period = 252  # 1 year of data
        
    async def analyze(self, symbol: str, market_data: MarketData) -> TradingSignal:
        """
        Analyze symbol for macro trading opportunities
        
        Args:
            symbol: Stock symbol to analyze
            market_data: Current market data
            
        Returns:
            TradingSignal: Macro trading signal
        """
        try:
            # Get long-term data
            weekly_data = await self._get_historical_data(symbol, "1w", 52)  # 1 year weekly
            monthly_data = await self._get_historical_data(symbol, "1M", 24)  # 2 years monthly
            
            # Long-term trend analysis
            long_trend = self._analyze_long_term_trends(weekly_data, monthly_data)
            
            # Economic cycle analysis
            cycle_analysis = await self._analyze_economic_cycle(symbol)
            
            # Fundamental analysis
            fundamental_score = await self._comprehensive_fundamental_analysis(symbol)
            
            # Sector analysis
            sector_analysis = await self._analyze_sector_trends(symbol)
            
            # Market regime analysis
            market_regime = self._analyze_market_regime(weekly_data)
            
            # Valuation analysis
            valuation = await self._analyze_valuation_metrics(symbol, market_data)
            
            # Combine macro signals
            combined_signal = self._combine_macro_signals(
                long_trend,
                cycle_analysis,
                fundamental_score,
                sector_analysis,
                market_regime,
                valuation,
                market_data
            )
            
            # Get comprehensive LLM analysis
            llm_insight = await self._get_llm_macro_analysis(symbol, combined_signal, market_data)
            
            # Generate macro signal
            signal = self._generate_macro_signal(symbol, combined_signal, llm_insight, market_data)
            
            self.log_signal(signal)
            return signal
            
        except Exception as e:
            self.logger.error(f"Error in macro analysis for {symbol}: {e}")
            return self._create_hold_signal(symbol, f"Macro analysis error: {str(e)}")
    
    def _analyze_long_term_trends(self, weekly_data: pd.DataFrame, monthly_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze long-term price trends"""
        try:
            signals = {}
            
            # Weekly trend analysis
            weekly_closes = weekly_data['close']
            weekly_sma_13 = weekly_closes.rolling(window=13).mean()  # ~3 months
            weekly_sma_52 = weekly_closes.rolling(window=52).mean()  # ~1 year
            
            current_price = weekly_closes.iloc[-1]
            
            # Long-term moving average signals
            above_yearly_ma = current_price > weekly_sma_52.iloc[-1]
            above_quarterly_ma = current_price > weekly_sma_13.iloc[-1]
            ma_trend = 'BULLISH' if (above_yearly_ma and above_quarterly_ma) else 'BEARISH'
            
            signals['moving_averages'] = {
                'above_yearly_ma': above_yearly_ma,
                'above_quarterly_ma': above_quarterly_ma,
                'trend': ma_trend,
                'yearly_ma': weekly_sma_52.iloc[-1],
                'quarterly_ma': weekly_sma_13.iloc[-1]
            }
            
            # Monthly momentum
            if len(monthly_data) >= 12:
                monthly_closes = monthly_data['close']
                yearly_return = (monthly_closes.iloc[-1] / monthly_closes.iloc[-12] - 1)
                six_month_return = (monthly_closes.iloc[-1] / monthly_closes.iloc[-6] - 1) if len(monthly_closes) >= 6 else 0
                three_month_return = (monthly_closes.iloc[-1] / monthly_closes.iloc[-3] - 1) if len(monthly_closes) >= 3 else 0
                
                signals['returns'] = {
                    'yearly_return': yearly_return,
                    'six_month_return': six_month_return,
                    'three_month_return': three_month_return,
                    'momentum_score': self._calculate_momentum_score(yearly_return, six_month_return, three_month_return)
                }
            else:
                signals['returns'] = {'momentum_score': 0}
            
            # Volatility analysis
            weekly_returns = weekly_closes.pct_change().dropna()
            volatility = weekly_returns.std() * np.sqrt(52)  # Annualized volatility
            avg_volatility = 0.25  # Assume average stock volatility of 25%
            
            vol_regime = 'HIGH' if volatility > avg_volatility * 1.5 else 'LOW' if volatility < avg_volatility * 0.5 else 'NORMAL'
            
            signals['volatility'] = {
                'current_vol': volatility,
                'vol_regime': vol_regime,
                'vol_percentile': min(volatility / (avg_volatility * 2), 1.0)
            }
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Long-term trend analysis error: {e}")
            return {'moving_averages': {'trend': 'NEUTRAL'}, 'returns': {'momentum_score': 0}}
    
    def _calculate_momentum_score(self, yearly: float, six_month: float, three_month: float) -> float:
        """Calculate momentum score from different timeframe returns"""
        # Weight recent returns more heavily
        score = (yearly * 0.3 + six_month * 0.4 + three_month * 0.3)
        return max(min(score, 1.0), -1.0)  # Normalize to [-1, 1]
    
    async def _analyze_economic_cycle(self, symbol: str) -> Dict[str, Any]:
        """Analyze current economic cycle position"""
        try:
            # This would integrate with economic data APIs
            # For now, simulate economic indicators
            
            # Mock economic indicators (in production, fetch real data)
            gdp_growth = np.random.normal(2.5, 1.0)  # GDP growth rate
            inflation_rate = np.random.normal(2.0, 0.5)  # Inflation
            unemployment = np.random.normal(4.0, 1.0)  # Unemployment rate
            interest_rates = np.random.normal(3.0, 1.0)  # Interest rates
            
            # Determine cycle phase
            if gdp_growth > 3 and unemployment < 4:
                cycle_phase = 'EXPANSION'
            elif gdp_growth < 1 and unemployment > 6:
                cycle_phase = 'RECESSION'
            elif gdp_growth > gdp_growth * 0.8:  # Recovery
                cycle_phase = 'RECOVERY'
            else:
                cycle_phase = 'SLOWDOWN'
            
            # Score based on cycle favorability for stocks
            cycle_scores = {
                'RECOVERY': 0.8,
                'EXPANSION': 0.6,
                'SLOWDOWN': 0.3,
                'RECESSION': 0.1
            }
            
            return {
                'cycle_phase': cycle_phase,
                'cycle_score': cycle_scores.get(cycle_phase, 0.5),
                'gdp_growth': gdp_growth,
                'inflation': inflation_rate,
                'unemployment': unemployment,
                'interest_rates': interest_rates,
                'favorable_for_stocks': cycle_phase in ['RECOVERY', 'EXPANSION']
            }
            
        except Exception as e:
            self.logger.error(f"Economic cycle analysis error: {e}")
            return {'cycle_phase': 'UNKNOWN', 'cycle_score': 0.5}
    
    async def _comprehensive_fundamental_analysis(self, symbol: str) -> Dict[str, Any]:
        """Comprehensive fundamental analysis"""
        try:
            # Mock fundamental metrics (in production, fetch real financial data)
            pe_ratio = 15 + np.random.normal(0, 5)
            pb_ratio = 2 + np.random.normal(0, 1)
            roe = 0.12 + np.random.normal(0, 0.05)
            debt_to_equity = 0.4 + np.random.normal(0, 0.2)
            revenue_growth = 0.08 + np.random.normal(0, 0.1)
            earnings_growth = 0.10 + np.random.normal(0, 0.15)
            
            # Score each metric
            scores = {}
            
            # P/E ratio (lower is better, but not too low)
            if 8 <= pe_ratio <= 20:
                scores['pe'] = 0.8
            elif 20 < pe_ratio <= 30:
                scores['pe'] = 0.6
            else:
                scores['pe'] = 0.3
            
            # P/B ratio
            scores['pb'] = 0.8 if pb_ratio < 2 else 0.5 if pb_ratio < 4 else 0.2
            
            # ROE
            scores['roe'] = min(roe * 5, 1.0)  # Scale ROE to 0-1
            
            # Debt to Equity
            scores['debt'] = 0.8 if debt_to_equity < 0.3 else 0.6 if debt_to_equity < 0.6 else 0.3
            
            # Growth rates
            scores['revenue_growth'] = min(max(revenue_growth * 5, 0), 1.0)
            scores['earnings_growth'] = min(max(earnings_growth * 3, 0), 1.0)
            
            # Overall fundamental score
            fundamental_score = np.mean(list(scores.values()))
            
            return {
                'pe_ratio': pe_ratio,
                'pb_ratio': pb_ratio,
                'roe': roe,
                'debt_to_equity': debt_to_equity,
                'revenue_growth': revenue_growth,
                'earnings_growth': earnings_growth,
                'component_scores': scores,
                'overall_score': fundamental_score,
                'quality_score': (scores['roe'] + scores['debt']) / 2,
                'growth_score': (scores['revenue_growth'] + scores['earnings_growth']) / 2,
                'value_score': (scores['pe'] + scores['pb']) / 2
            }
            
        except Exception as e:
            self.logger.error(f"Fundamental analysis error: {e}")
            return {'overall_score': 0.5}
    
    async def _analyze_sector_trends(self, symbol: str) -> Dict[str, Any]:
        """Analyze sector and industry trends"""
        try:
            # Mock sector analysis (in production, would use sector data)
            sectors = ['Technology', 'Healthcare', 'Finance', 'Consumer', 'Industrial', 'Energy']
            symbol_sector = sectors[hash(symbol) % len(sectors)]
            
            # Mock sector performance
            sector_performance = {
                'Technology': 0.12,
                'Healthcare': 0.08,
                'Finance': 0.06,
                'Consumer': 0.05,
                'Industrial': 0.07,
                'Energy': 0.15
            }
            
            # Sector rotation analysis
            sector_momentum = sector_performance.get(symbol_sector, 0.08)
            sector_rank = sorted(sector_performance.values(), reverse=True).index(sector_momentum) + 1
            
            # Sector relative strength
            market_return = 0.08  # Assume 8% market return
            relative_strength = sector_momentum - market_return
            
            return {
                'sector': symbol_sector,
                'sector_performance': sector_momentum,
                'sector_rank': sector_rank,
                'relative_strength': relative_strength,
                'outperforming_market': relative_strength > 0,
                'in_leading_sectors': sector_rank <= 3,
                'sector_score': min(max((sector_momentum - 0.05) * 10, 0), 1)  # Normalize
            }
            
        except Exception as e:
            self.logger.error(f"Sector analysis error: {e}")
            return {'sector_score': 0.5}
    
    def _analyze_market_regime(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze current market regime"""
        try:
            closes = data['close']
            returns = closes.pct_change().dropna()
            
            # Volatility regime
            vol = returns.std() * np.sqrt(52)
            vol_regime = 'HIGH_VOL' if vol > 0.3 else 'LOW_VOL' if vol < 0.15 else 'NORMAL_VOL'
            
            # Trend regime (using 26-week moving average)
            ma_26 = closes.rolling(window=26).mean()
            current_price = closes.iloc[-1]
            
            if len(ma_26) >= 26:
                trend_regime = 'BULL_MARKET' if current_price > ma_26.iloc[-1] else 'BEAR_MARKET'
                trend_strength = abs(current_price - ma_26.iloc[-1]) / ma_26.iloc[-1]
            else:
                trend_regime = 'NEUTRAL'
                trend_strength = 0
            
            # Market breadth (mock)
            market_breadth = np.random.uniform(0.3, 0.8)  # % of stocks above MA
            
            # Risk-on/Risk-off sentiment
            if vol_regime == 'LOW_VOL' and trend_regime == 'BULL_MARKET':
                risk_sentiment = 'RISK_ON'
            elif vol_regime == 'HIGH_VOL' and trend_regime == 'BEAR_MARKET':
                risk_sentiment = 'RISK_OFF'
            else:
                risk_sentiment = 'MIXED'
            
            return {
                'volatility_regime': vol_regime,
                'trend_regime': trend_regime,
                'trend_strength': trend_strength,
                'market_breadth': market_breadth,
                'risk_sentiment': risk_sentiment,
                'favorable_for_longs': trend_regime == 'BULL_MARKET' and risk_sentiment in ['RISK_ON', 'MIXED'],
                'regime_score': 0.8 if risk_sentiment == 'RISK_ON' else 0.2 if risk_sentiment == 'RISK_OFF' else 0.5
            }
            
        except Exception as e:
            self.logger.error(f"Market regime analysis error: {e}")
            return {'regime_score': 0.5}
    
    async def _analyze_valuation_metrics(self, symbol: str, market_data: MarketData) -> Dict[str, Any]:
        """Analyze valuation relative to historical norms"""
        try:
            # Mock valuation analysis
            current_price = market_data.price
            
            # Estimate fair value (mock calculation)
            estimated_fair_value = current_price * (0.9 + np.random.uniform(0, 0.2))
            
            # Price to fair value ratio
            price_to_fair_value = current_price / estimated_fair_value
            
            # Historical price percentile (mock)
            price_percentile = np.random.uniform(0.2, 0.9)
            
            # Valuation classification
            if price_to_fair_value < 0.8:
                valuation = 'UNDERVALUED'
            elif price_to_fair_value > 1.2:
                valuation = 'OVERVALUED'
            else:
                valuation = 'FAIRLY_VALUED'
            
            # Valuation score (higher is better for buying)
            if valuation == 'UNDERVALUED':
                val_score = 0.9
            elif valuation == 'OVERVALUED':
                val_score = 0.1
            else:
                val_score = 0.5
            
            return {
                'current_price': current_price,
                'estimated_fair_value': estimated_fair_value,
                'price_to_fair_value': price_to_fair_value,
                'price_percentile': price_percentile,
                'valuation': valuation,
                'valuation_score': val_score,
                'attractive_entry': valuation == 'UNDERVALUED'
            }
            
        except Exception as e:
            self.logger.error(f"Valuation analysis error: {e}")
            return {'valuation_score': 0.5}
    
    def _combine_macro_signals(self, long_trend: Dict, cycle: Dict, fundamental: Dict,
                              sector: Dict, market_regime: Dict, valuation: Dict,
                              market_data: MarketData) -> Dict[str, Any]:
        """Combine all macro analysis components"""
        
        # Weighted scoring system
        weights = {
            'long_trend': 0.25,      # 25% - Long-term trend
            'cycle': 0.20,           # 20% - Economic cycle
            'fundamental': 0.20,     # 20% - Fundamentals
            'sector': 0.15,          # 15% - Sector trends
            'market_regime': 0.10,   # 10% - Market regime
            'valuation': 0.10        # 10% - Valuation
        }
        
        # Calculate component scores
        scores = {}
        
        # Long-term trend score
        trend_score = 0.8 if long_trend['moving_averages']['trend'] == 'BULLISH' else 0.2
        momentum_score = (long_trend.get('returns', {}).get('momentum_score', 0) + 1) / 2  # Scale to 0-1
        scores['long_trend'] = (trend_score + momentum_score) / 2
        
        # Cycle score
        scores['cycle'] = cycle.get('cycle_score', 0.5)
        
        # Fundamental score
        scores['fundamental'] = fundamental.get('overall_score', 0.5)
        
        # Sector score
        scores['sector'] = sector.get('sector_score', 0.5)
        
        # Market regime score
        scores['market_regime'] = market_regime.get('regime_score', 0.5)
        
        # Valuation score
        scores['valuation'] = valuation.get('valuation_score', 0.5)
        
        # Calculate weighted overall score
        overall_score = sum(scores[component] * weights[component] for component in weights.keys())
        
        # Determine signal based on score
        if overall_score >= 0.65:
            signal = 'BUY'
            confidence = min(overall_score, 0.95)
        elif overall_score <= 0.35:
            signal = 'SELL'
            confidence = min(1 - overall_score, 0.95)
        else:
            signal = 'HOLD'
            confidence = 0.4
        
        # Quality adjustment - higher confidence for quality stocks
        quality_multiplier = 1 + (fundamental.get('quality_score', 0.5) - 0.5) * 0.2
        confidence *= quality_multiplier
        confidence = min(confidence, 0.95)
        
        return {
            'signal': signal,
            'confidence': confidence,
            'overall_score': overall_score,
            'component_scores': scores,
            'components': {
                'long_trend': long_trend,
                'economic_cycle': cycle,
                'fundamental': fundamental,
                'sector': sector,
                'market_regime': market_regime,
                'valuation': valuation
            }
        }
    
    async def _get_llm_macro_analysis(self, symbol: str, signals: Dict, market_data: MarketData) -> str:
        """Get comprehensive LLM macro analysis"""
        try:
            prompt = f"""
            As a macro investment strategist, provide comprehensive analysis for {symbol}:
            
            Current Price: ${market_data.price:.2f}
            Macro Signal: {signals['signal']} (Confidence: {signals['confidence']:.1%})
            Overall Score: {signals['overall_score']:.2f}
            
            Component Analysis:
            - Economic Cycle: {signals['components']['economic_cycle']['cycle_phase']}
            - Sector Performance: {signals['components']['sector']['sector']} (Rank: {signals['components']['sector'].get('sector_rank', 'N/A')})
            - Valuation: {signals['components']['valuation']['valuation']}
            - Market Regime: {signals['components']['market_regime']['trend_regime']}
            - Fundamental Score: {signals['components']['fundamental']['overall_score']:.2f}
            
            Provide macro investment analysis covering:
            1. Long-term investment thesis (6-18 months)
            2. Key macro risks and catalysts
            3. Position sizing and timeline recommendations
            4. Exit strategy considerations
            5. Portfolio allocation suggestions
            
            Keep response comprehensive but under 300 words.
            """
            
            analysis = await self.llm_handler.get_analysis(prompt)
            return analysis
            
        except Exception as e:
            self.logger.error(f"LLM macro analysis error: {e}")
            return "Macro analysis shows mixed signals across economic, fundamental, and technical factors. Consider diversification and gradual position building."
    
    def _generate_macro_signal(self, symbol: str, combined_signal: Dict, llm_insight: str, market_data: MarketData) -> TradingSignal:
        """Generate comprehensive macro trading signal"""
        
        action = combined_signal['signal']
        confidence = combined_signal['confidence']
        
        # Build detailed reasoning
        reasoning = f"Macro analysis: {action} signal with {confidence:.1%} confidence (Score: {combined_signal['overall_score']:.2f}). "
        
        # Add component insights
        components = combined_signal['components']
        
        # Economic cycle
        cycle = components['economic_cycle']
        reasoning += f"Economic cycle: {cycle['cycle_phase']} (favorable: {cycle.get('favorable_for_stocks', False)}). "
        
        # Sector
        sector = components['sector']
        reasoning += f"Sector ({sector['sector']}): rank #{sector.get('sector_rank', 'N/A')}, outperforming: {sector.get('outperforming_market', False)}. "
        
        # Valuation
        valuation = components['valuation']
        reasoning += f"Valuation: {valuation['valuation']} (Fair value: ${valuation['estimated_fair_value']:.2f}). "
        
        # Market regime
        regime = components['market_regime']
        reasoning += f"Market regime: {regime['trend_regime']}, sentiment: {regime['risk_sentiment']}. "
        
        # Fundamentals
        fundamental = components['fundamental']
        reasoning += f"Fundamentals: Overall {fundamental['overall_score']:.2f} (Quality: {fundamental.get('quality_score', 0.5):.2f}, Growth: {fundamental.get('growth_score', 0.5):.2f}). "
        
        reasoning += f"AI insight: {llm_insight[:150]}..."
        
        # Calculate long-term targets
        target_price = None
        stop_loss = None
        
        if action == 'BUY':
            # Conservative long-term target
            fair_value = valuation.get('estimated_fair_value', market_data.price * 1.15)
            target_price = max(fair_value, market_data.price * (1 + self.min_profit_target))
            
            # Wider stop loss for macro positions
            stop_loss = market_data.price * 0.85  # 15% stop loss
            
        elif action == 'SELL':
            # Short target or reduce allocation
            fair_value = valuation.get('estimated_fair_value', market_data.price * 0.85)
            target_price = min(fair_value, market_data.price * (1 - self.min_profit_target))
            
            # Stop loss for short positions
            stop_loss = market_data.price * 1.15  # 15% stop loss
        
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
            confidence=0.4,
            reasoning=f"Macro hold: {reason}",
            timeframe=self.timeframe,
            timestamp=datetime.now()
        )
    
    async def _get_historical_data(self, symbol: str, timeframe: str, periods: int) -> pd.DataFrame:
        """Get historical data for macro analysis"""
        # Mock implementation for different timeframes
        if timeframe == "1w":
            freq = 'W'
        elif timeframe == "1M":
            freq = 'M'
        else:
            freq = 'D'
        
        dates = pd.date_range(end=datetime.now(), periods=periods, freq=freq)
        
        np.random.seed(hash(symbol) % 2**32)
        base_price = 75 + (hash(symbol) % 150)
        
        # Generate long-term trending data
        prices = []
        price = base_price
        long_term_trend = 0.0015  # Slight upward bias
        
        for i in range(periods):
            # Add cyclical components for macro timeframes
            cycle_component = 0.001 * np.sin(2 * np.pi * i / 26)  # ~6 month cycle
            noise = np.random.normal(0, 0.03)  # 3% volatility
            
            change = long_term_trend + cycle_component + noise
            price = price * (1 + change)
            prices.append(max(price, 1))
        
        volumes = np.random.randint(200000, 2000000, periods)
        
        data = []
        for i, price in enumerate(prices):
            volatility = 0.04 if timeframe == "1w" else 0.06 if timeframe == "1M" else 0.02
            high = price * (1 + abs(np.random.normal(0, volatility/2)))
            low = price * (1 - abs(np.random.normal(0, volatility/2)))
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
        """Return macro strategy description"""
        return """
        Macro Trading Strategy:
        - Timeframe: 6-18 months
        - Focus: Economic cycles, fundamental analysis, sector rotation
        - Analysis: Comprehensive fundamental metrics, valuation models
        - Risk: 15% stop loss, 15%+ profit targets
        - Economic cycle positioning and sector leadership
        - Long-term value and growth assessment
        - Market regime and risk sentiment analysis
        - AI-enhanced macro thesis development
        """