# agents/consensus_controller.py
"""
Consensus Controller - Orchestrates all trading agents and makes final decisions
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np

from .base_agent import TradingSignal, MarketData
from .short_term_agent import ShortTermAgent
from .swing_agent import SwingAgent
from .macro_agent import MacroAgent
from utils.llm_handler import LLMHandler
from memory.memory_manager import MemoryManager

class ConsensusController:
    """
    Orchestrates multiple trading agents and creates consensus decisions
    Implements the debate and decision-making process
    """
    
    def __init__(self):
        self.logger = logging.getLogger("ConsensusController")
        
        # Initialize agents
        self.agents = {
            'short_term': ShortTermAgent(),
            'swing': SwingAgent(),
            'macro': MacroAgent()
        }
        
        self.llm_handler = LLMHandler()
        self.memory_manager = MemoryManager()
        
        # Consensus parameters
        self.consensus_threshold = 0.6
        self.min_confidence = 0.5
        self.max_debate_rounds = 3
        
        # Agent weights (can be dynamically adjusted based on performance)
        self.agent_weights = {
            'short_term': 0.3,
            'swing': 0.4,
            'macro': 0.3
        }
        
        # Performance tracking
        self.agent_performance = {
            'short_term': {'total': 0, 'correct': 0, 'win_rate': 0.5},
            'swing': {'total': 0, 'correct': 0, 'win_rate': 0.5},
            'macro': {'total': 0, 'correct': 0, 'win_rate': 0.5}
        }
    
    async def get_consensus(self, symbol: str, market_data: Optional[MarketData] = None) -> Dict[str, Any]:
        """
        Get consensus trading decision from all agents
        
        Args:
            symbol: Stock symbol to analyze
            market_data: Current market data (if None, will fetch)
            
        Returns:
            Dict: Consensus decision with agent breakdown
        """
        try:
            # Get or create market data
            if market_data is None:
                market_data = await self._get_market_data(symbol)
            
            # Get individual agent signals
            agent_signals = await self._gather_agent_signals(symbol, market_data)
            
            # Initial consensus calculation
            initial_consensus = self._calculate_initial_consensus(agent_signals)
            
            # If consensus is weak, initiate debate process
            if initial_consensus['confidence'] < self.consensus_threshold:
                debated_consensus = await self._conduct_agent_debate(
                    symbol, market_data, agent_signals, initial_consensus
                )
                final_consensus = debated_consensus
            else:
                final_consensus = initial_consensus
            
            # Final validation with LLM
            validated_consensus = await self._llm_validate_consensus(
                symbol, market_data, agent_signals, final_consensus
            )
            
            # Store decision in memory
            await self._store_consensus_decision(symbol, validated_consensus, agent_signals)
            
            # Update agent weights based on recent performance
            self._update_agent_weights()
            
            return validated_consensus
            
        except Exception as e:
            self.logger.error(f"Error getting consensus for {symbol}: {e}")
            return self._create_error_consensus(symbol, str(e))
    
    async def _gather_agent_signals(self, symbol: str, market_data: MarketData) -> Dict[str, TradingSignal]:
        """Gather signals from all agents concurrently"""
        self.logger.info(f"Gathering agent signals for {symbol}")
        
        tasks = {}
        for agent_name, agent in self.agents.items():
            tasks[agent_name] = asyncio.create_task(
                agent.analyze(symbol, market_data)
            )
        
        # Wait for all agents to complete
        results = {}
        for agent_name, task in tasks.items():
            try:
                results[agent_name] = await task
                self.logger.info(f"{agent_name} signal: {results[agent_name].action} ({results[agent_name].confidence:.1%})")
            except Exception as e:
                self.logger.error(f"Error from {agent_name}: {e}")
                # Create fallback signal
                results[agent_name] = TradingSignal(
                    symbol=symbol,
                    action='HOLD',
                    confidence=0.3,
                    reasoning=f"Agent error: {str(e)}",
                    timestamp=datetime.now()
                )
        
        return results
    
    def _calculate_initial_consensus(self, agent_signals: Dict[str, TradingSignal]) -> Dict[str, Any]:
        """Calculate initial consensus from agent signals"""
        
        # Count votes
        votes = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        weighted_confidence = 0
        total_weight = 0
        
        agent_details = {}
        
        for agent_name, signal in agent_signals.items():
            weight = self.agent_weights.get(agent_name, 0.33)
            performance_multiplier = self.agent_performance[agent_name]['win_rate']
            
            # Adjust weight by performance
            adjusted_weight = weight * (0.5 + performance_multiplier)
            
            votes[signal.action] += adjusted_weight
            weighted_confidence += signal.confidence * adjusted_weight
            total_weight += adjusted_weight
            
            agent_details[agent_name] = {
                'action': signal.action,
                'confidence': signal.confidence,
                'weight': adjusted_weight,
                'reasoning': signal.reasoning[:100] + "..." if len(signal.reasoning) > 100 else signal.reasoning
            }
        
        # Determine consensus action
        if total_weight == 0:
            consensus_action = 'HOLD'
            consensus_confidence = 0.3
        else:
            consensus_action = max(votes, key=votes.get)
            consensus_confidence = weighted_confidence / total_weight
            
            # Adjust confidence based on vote distribution
            max_votes = votes[consensus_action]
            vote_ratio = max_votes / total_weight
            
            # Penalize weak consensus
            if vote_ratio < 0.6:
                consensus_confidence *= vote_ratio / 0.6
        
        return {
            'action': consensus_action,
            'confidence': consensus_confidence,
            'vote_distribution': votes,
            'agent_details': agent_details,
            'total_weight': total_weight,
            'consensus_strength': votes[consensus_action] / total_weight if total_weight > 0 else 0
        }
    
    async def _conduct_agent_debate(self, symbol: str, market_data: MarketData, 
                                  agent_signals: Dict[str, TradingSignal], 
                                  initial_consensus: Dict[str, Any]) -> Dict[str, Any]:
        """Conduct debate between agents to resolve disagreements"""
        
        self.logger.info(f"Conducting agent debate for {symbol} (weak consensus: {initial_consensus['confidence']:.1%})")
        
        debate_history = []
        current_consensus = initial_consensus
        
        for round_num in range(self.max_debate_rounds):
            # Identify disagreeing agents
            majority_action = current_consensus['action']
            disagreeing_agents = []
            agreeing_agents = []
            
            for agent_name, signal in agent_signals.items():
                if signal.action != majority_action:
                    disagreeing_agents.append((agent_name, signal))
                else:
                    agreeing_agents.append((agent_name, signal))
            
            if not disagreeing_agents:
                break  # Full consensus reached
            
            # Generate debate prompt
            debate_prompt = self._create_debate_prompt(
                symbol, market_data, majority_action, 
                agreeing_agents, disagreeing_agents, round_num
            )
            
            # Get LLM moderated debate
            debate_result = await self._llm_moderate_debate(debate_prompt)
            debate_history.append({
                'round': round_num + 1,
                'prompt': debate_prompt,
                'result': debate_result
            })
            
            # Adjust agent confidences based on debate
            current_consensus = self._adjust_consensus_after_debate(
                current_consensus, debate_result, agent_signals
            )
            
            # Check if consensus improved
            if current_consensus['confidence'] >= self.consensus_threshold:
                self.logger.info(f"Consensus reached after round {round_num + 1}")
                break
        
        current_consensus['debate_history'] = debate_history
        current_consensus['debate_rounds'] = len(debate_history)
        
        return current_consensus
    
    def _create_debate_prompt(self, symbol: str, market_data: MarketData, 
                            majority_action: str, agreeing_agents: List, 
                            disagreeing_agents: List, round_num: int) -> str:
        """Create prompt for agent debate"""
        
        prompt = f"""
        AGENT DEBATE - Round {round_num + 1} for {symbol}
        Current Price: ${market_data.price:.2f}
        Majority Position: {majority_action}
        
        AGREEING AGENTS ({majority_action}):
        """
        
        for agent_name, signal in agreeing_agents:
            prompt += f"\n{agent_name}: {signal.reasoning[:150]}..."
        
        prompt += f"\n\nDISAGREEING AGENTS:"
        
        for agent_name, signal in disagreeing_agents:
            prompt += f"\n{agent_name} ({signal.action}): {signal.reasoning[:150]}..."
        
        prompt += f"""
        
        As an AI trading mediator, analyze this disagreement:
        1. What are the key points of contention?
        2. Which arguments are most compelling?
        3. Are there any overlooked factors?
        4. What additional analysis might resolve this?
        5. Recommend confidence adjustments for each position
        
        Provide structured analysis focusing on evidence quality and logical consistency.
        """
        
        return prompt
    
    async def _llm_moderate_debate(self, debate_prompt: str) -> Dict[str, Any]:
        """Use LLM to moderate agent debate"""
        try:
            moderation = await self.llm_handler.get_analysis(debate_prompt)
            
            # Parse moderation result (in production, would use structured output)
            return {
                'moderation': moderation,
                'timestamp': datetime.now(),
                'key_insights': self._extract_key_insights(moderation)
            }
            
        except Exception as e:
            self.logger.error(f"LLM debate moderation error: {e}")
            return {
                'moderation': "Unable to moderate debate due to technical issues.",
                'key_insights': []
            }
    
    def _extract_key_insights(self, moderation_text: str) -> List[str]:
        """Extract key insights from moderation text"""
        # Simple keyword extraction (in production, would use NLP)
        insights = []
        
        key_phrases = [
            "key point", "compelling", "overlooked", "evidence", 
            "risk", "opportunity", "momentum", "valuation"
        ]
        
        sentences = moderation_text.split('.')
        for sentence in sentences:
            for phrase in key_phrases:
                if phrase.lower() in sentence.lower():
                    insights.append(sentence.strip())
                    break
        
        return insights[:5]  # Return top 5 insights
    
    def _adjust_consensus_after_debate(self, consensus: Dict[str, Any], 
                                     debate_result: Dict[str, Any], 
                                     agent_signals: Dict[str, TradingSignal]) -> Dict[str, Any]:
        """Adjust consensus based on debate outcome"""
        
        # Simple adjustment based on debate insights
        moderation = debate_result.get('moderation', '').lower()
        
        # Look for confidence adjustment signals in moderation
        confidence_adjustments = {}
        
        for agent_name in agent_signals.keys():
            if agent_name.lower() in moderation:
                if any(word in moderation for word in ['strong', 'compelling', 'correct']):
                    confidence_adjustments[agent_name] = 1.1  # Boost confidence
                elif any(word in moderation for word in ['weak', 'flawed', 'questionable']):
                    confidence_adjustments[agent_name] = 0.9  # Reduce confidence
        
        # Recalculate consensus with adjustments
        adjusted_consensus = consensus.copy()
        
        if confidence_adjustments:
            # Recalculate weighted consensus
            votes = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
            weighted_confidence = 0
            total_weight = 0
            
            for agent_name, signal in agent_signals.items():
                weight = self.agent_weights.get(agent_name, 0.33)
                adjustment = confidence_adjustments.get(agent_name, 1.0)
                adjusted_weight = weight * adjustment
                
                votes[signal.action] += adjusted_weight
                weighted_confidence += signal.confidence * adjusted_weight
                total_weight += adjusted_weight
            
            if total_weight > 0:
                new_action = max(votes, key=votes.get)
                new_confidence = weighted_confidence / total_weight
                
                adjusted_consensus.update({
                    'action': new_action,
                    'confidence': new_confidence,
                    'vote_distribution': votes,
                    'confidence_adjustments': confidence_adjustments
                })
        
        return adjusted_consensus
    
    async def _llm_validate_consensus(self, symbol: str, market_data: MarketData,
                                    agent_signals: Dict[str, TradingSignal],
                                    consensus: Dict[str, Any]) -> Dict[str, Any]:
        """Final LLM validation of consensus decision"""
        
        try:
            validation_prompt = f"""
            FINAL CONSENSUS VALIDATION for {symbol}
            Current Price: ${market_data.price:.2f}
            
            CONSENSUS DECISION: {consensus['action']} (Confidence: {consensus['confidence']:.1%})
            
            AGENT BREAKDOWN:
            """
            
            for agent_name, details in consensus['agent_details'].items():
                validation_prompt += f"\n{agent_name}: {details['action']} ({details['confidence']:.1%}) - {details['reasoning']}"
            
            if 'debate_history' in consensus:
                validation_prompt += f"\n\nDEBATE CONDUCTED: {consensus['debate_rounds']} rounds"
            
            validation_prompt += f"""
            
            As a senior portfolio manager, provide final validation:
            1. Is this consensus decision sound?
            2. Are there any critical risks overlooked?
            3. What is the appropriate position size?
            4. Any timing considerations?
            5. Risk management recommendations?
            
            Provide concise validation with specific recommendations.
            """
            
            validation = await self.llm_handler.get_analysis(validation_prompt)
            
            # Extract position size recommendation (simple parsing)
            position_size = self._extract_position_size(validation)
            
            final_consensus = consensus.copy()
            final_consensus.update({
                'llm_validation': validation,
                'recommended_position_size': position_size,
                'final_timestamp': datetime.now(),
                'validation_score': self._calculate_validation_score(validation)
            })
            
            return final_consensus
            
        except Exception as e:
            self.logger.error(f"LLM validation error: {e}")
            consensus['llm_validation'] = f"Validation error: {str(e)}"
            consensus['validation_score'] = 0.5
            return consensus
    
    def _extract_position_size(self, validation_text: str) -> float:
        """Extract position size recommendation from validation text"""
        # Simple pattern matching for position size
        import re
        
        patterns = [
            r'(\d+(?:\.\d+)?)%.*position',
            r'position.*(\d+(?:\.\d+)?)%',
            r'size.*(\d+(?:\.\d+)?)%'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, validation_text.lower())
            if match:
                try:
                    return float(match.group(1)) / 100
                except ValueError:
                    pass
        
        # Default position size based on confidence
        return 0.05  # 5% default
    
    def _calculate_validation_score(self, validation_text: str) -> float:
        """Calculate validation score based on LLM response"""
        positive_words = ['sound', 'good', 'solid', 'recommend', 'favorable', 'strong']
        negative_words = ['risky', 'caution', 'weak', 'avoid', 'concerning', 'uncertain']
        
        text_lower = validation_text.lower()
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count + negative_count == 0:
            return 0.5  # Neutral
        
        score = positive_count / (positive_count + negative_count)
        return score
    
    async def _store_consensus_decision(self, symbol: str, consensus: Dict[str, Any], 
                                      agent_signals: Dict[str, TradingSignal]):
        """Store consensus decision in memory for learning"""
        
        decision_record = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'consensus': consensus,
            'agent_signals': {
                name: {
                    'action': signal.action,
                    'confidence': signal.confidence,
                    'reasoning': signal.reasoning
                }
                for name, signal in agent_signals.items()
            }
        }
        
        await self.memory_manager.store_decision(decision_record)
    
    def _update_agent_weights(self):
        """Update agent weights based on recent performance"""
        
        total_performance = sum(
            perf['win_rate'] for perf in self.agent_performance.values()
        )
        
        if total_performance > 0:
            for agent_name in self.agent_weights:
                performance_ratio = self.agent_performance[agent_name]['win_rate'] / total_performance
                
                # Gradually adjust weights toward performance
                current_weight = self.agent_weights[agent_name]
                target_weight = performance_ratio
                
                # Smooth adjustment (10% step toward target)
                self.agent_weights[agent_name] = current_weight * 0.9 + target_weight * 0.1
        
        # Normalize weights
        total_weight = sum(self.agent_weights.values())
        if total_weight > 0:
            for agent_name in self.agent_weights:
                self.agent_weights[agent_name] /= total_weight
    
    def update_agent_performance(self, agent_name: str, was_correct: bool):
        """Update agent performance metrics"""
        if agent_name in self.agent_performance:
            perf = self.agent_performance[agent_name]
            perf['total'] += 1
            if was_correct:
                perf['correct'] += 1
            
            perf['win_rate'] = perf['correct'] / perf['total'] if perf['total'] > 0 else 0.5
    
    async def _get_market_data(self, symbol: str) -> MarketData:
        """Get current market data for symbol"""
        # Mock implementation - in production, would fetch real data
        
        # Generate realistic mock data
        np.random.seed(hash(symbol) % 2**32)
        base_price = 100 + (hash(symbol) % 300)
        
        # Add some intraday movement
        price_change = np.random.normal(0, 0.02)  # 2% volatility
        current_price = base_price * (1 + price_change)
        
        high = current_price * (1 + abs(np.random.normal(0, 0.01)))
        low = current_price * (1 - abs(np.random.normal(0, 0.01)))
        
        return MarketData(
            symbol=symbol,
            price=current_price,
            volume=np.random.randint(1000000, 10000000),
            high=high,
            low=low,
            open=base_price,
            close=current_price,
            timestamp=datetime.now()
        )
    
    def _create_error_consensus(self, symbol: str, error_msg: str) -> Dict[str, Any]:
        """Create error consensus when analysis fails"""
        return {
            'action': 'HOLD',
            'confidence': 0.2,
            'vote_distribution': {'HOLD': 1.0, 'BUY': 0.0, 'SELL': 0.0},
            'agent_details': {
                'error': {
                    'action': 'HOLD',
                    'confidence': 0.2,
                    'reasoning': f"System error: {error_msg}",
                    'weight': 1.0
                }
            },
            'error': True,
            'error_message': error_msg,
            'timestamp': datetime.now()
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and performance metrics"""
        return {
            'agents_active': {name: agent.is_active for name, agent in self.agents.items()},
            'agent_weights': self.agent_weights.copy(),
            'agent_performance': self.agent_performance.copy(),
            'consensus_threshold': self.consensus_threshold,
            'total_decisions': sum(perf['total'] for perf in self.agent_performance.values()),
            'system_uptime': datetime.now(),  # In production, track actual uptime
            'last_updated': datetime.now()
        }