# agents/chatbot_agent.py
"""
Trading Chatbot Agent
Provides conversational interface for trading advice and market analysis
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import re

from .base_agent import BaseAgent, TradingSignal, MarketData
from utils.llm_handler import LLMHandler
from data.data_manager import DataManager
from memory.conversation_history import ConversationHistory

class TradingChatbot:
    """
    AI-powered trading chatbot that provides personalized trading advice,
    market analysis, and answers trading-related questions
    """
    
    def __init__(self):
        self.logger = logging.getLogger("TradingChatbot")
        self.llm_handler = LLMHandler()
        self.data_manager = DataManager()
        self.conversation_history = ConversationHistory()
        
        # Chatbot personality and expertise
        self.personality = {
            'name': 'TradeGPT',
            'expertise': 'Professional Trading & Market Analysis',
            'style': 'Helpful, analytical, risk-aware',
            'specialties': [
                'Technical Analysis', 'Fundamental Analysis', 'Risk Management',
                'Portfolio Optimization', 'Market Psychology', 'Trading Strategies'
            ]
        }
        
        # Pre-defined responses for common queries
        self.quick_responses = {
            'greeting': [
                "Hello! I'm TradeGPT, your AI trading assistant. How can I help you with your trading decisions today?",
                "Hi there! Ready to analyze some markets? What stock or trading question can I help you with?",
                "Welcome! I'm here to help with trading analysis, market insights, and investment advice. What's on your mind?"
            ],
            'help': [
                "I can help you with:\n• Stock analysis and recommendations\n• Market trend analysis\n• Risk assessment\n• Trading strategy advice\n• Portfolio suggestions\n• Technical and fundamental analysis",
                "Ask me about any stock symbol, market trends, trading strategies, or investment decisions. I'm here to provide data-driven insights!"
            ]
        }
        
        # Context tracking
        self.current_context = {
            'symbols_discussed': [],
            'last_analysis_type': None,
            'user_risk_tolerance': 'moderate',
            'preferred_timeframe': 'swing',
            'conversation_theme': None
        }
    
    async def get_response(self, user_input: str, user_id: str = "default") -> str:
        """
        Generate AI response to user input
        
        Args:
            user_input: User's message/question
            user_id: User identifier for conversation tracking
            
        Returns:
            str: AI response
        """
        try:
            # Clean and analyze user input
            cleaned_input = self._clean_input(user_input)
            intent = self._analyze_intent(cleaned_input)
            
            # Store user message in conversation history
            await self.conversation_history.add_message(
                user_id, 'user', user_input, intent
            )
            
            # Generate appropriate response based on intent
            response = await self._generate_response(cleaned_input, intent, user_id)
            
            # Store bot response
            await self.conversation_history.add_message(
                user_id, 'bot', response, 'response'
            )
            
            # Update context
            self._update_context(cleaned_input, intent)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating chatbot response: {e}")
            return self._get_error_response()
    
    def _clean_input(self, user_input: str) -> str:
        """Clean and normalize user input"""
        # Remove extra whitespace
        cleaned = re.sub(r'\\s+', ' ', user_input.strip())
        
        # Extract stock symbols
        symbols = re.findall(r'\\b[A-Z]{1,5}\\b', cleaned.upper())
        self.current_context['symbols_mentioned'] = symbols
        
        return cleaned
    
    def _analyze_intent(self, user_input: str) -> str:
        """Analyze user intent from input"""
        input_lower = user_input.lower()
        
        # Greeting patterns
        if any(word in input_lower for word in ['hello', 'hi', 'hey', 'greetings']):
            return 'greeting'
        
        # Help requests
        if any(word in input_lower for word in ['help', 'what can you', 'how to', 'guide']):
            return 'help'
        
        # Stock analysis requests
        if any(word in input_lower for word in ['analyze', 'analysis', 'what do you think', 'opinion']):
            return 'stock_analysis'
        
        # Buy/sell advice
        if any(word in input_lower for word in ['should i buy', 'should i sell', 'buy or sell', 'recommend']):
            return 'buy_sell_advice'
        
        # Market outlook
        if any(word in input_lower for word in ['market', 'outlook', 'trend', 'direction']):
            return 'market_outlook'
        
        # Risk assessment
        if any(word in input_lower for word in ['risk', 'safe', 'dangerous', 'volatile']):
            return 'risk_assessment'
        
        # Portfolio advice
        if any(word in input_lower for word in ['portfolio', 'diversification', 'allocation']):
            return 'portfolio_advice'
        
        # Technical analysis
        if any(word in input_lower for word in ['technical', 'chart', 'indicator', 'support', 'resistance']):
            return 'technical_analysis'
        
        # News/events
        if any(word in input_lower for word in ['news', 'earnings', 'event', 'catalyst']):
            return 'news_analysis'
        
        return 'general_question'
    
    async def _generate_response(self, user_input: str, intent: str, user_id: str) -> str:
        """Generate response based on intent"""
        
        if intent == 'greeting':
            return self._get_random_response('greeting')
        
        elif intent == 'help':
            return self._get_random_response('help')
        
        elif intent in ['stock_analysis', 'buy_sell_advice']:
            return await self._handle_stock_analysis(user_input, intent)
        
        elif intent == 'market_outlook':
            return await self._handle_market_outlook(user_input)
        
        elif intent == 'risk_assessment':
            return await self._handle_risk_assessment(user_input)
        
        elif intent == 'portfolio_advice':
            return await self._handle_portfolio_advice(user_input)
        
        elif intent == 'technical_analysis':
            return await self._handle_technical_analysis(user_input)
        
        elif intent == 'news_analysis':
            return await self._handle_news_analysis(user_input)
        
        else:
            return await self._handle_general_question(user_input, user_id)
    
    async def _handle_stock_analysis(self, user_input: str, intent: str) -> str:
        """Handle stock analysis requests"""
        
        symbols = self.current_context.get('symbols_mentioned', [])
        
        if not symbols:
            return "I'd be happy to analyze a stock for you! Please specify a stock symbol (like AAPL, TSLA, GOOGL) and I'll provide a comprehensive analysis."
        
        # Analyze the first mentioned symbol
        symbol = symbols[0]
        
        try:
            # Get market data
            market_data = await self.data_manager.get_current_data(symbol)
            
            # Create analysis prompt
            analysis_prompt = f"""
            Provide a comprehensive trading analysis for {symbol} at ${market_data.get('price', 'N/A')}:
            
            User request: "{user_input}"
            Intent: {intent}
            
            Include:
            1. Current price assessment
            2. Short-term outlook (1-5 days)
            3. Medium-term outlook (1-4 weeks)
            4. Key technical levels
            5. Risk factors
            6. {'Buy/sell recommendation with reasoning' if intent == 'buy_sell_advice' else 'Overall assessment'}
            
            Style: Conversational, analytical, balanced
            Length: 200-300 words
            """
            
            analysis = await self.llm_handler.get_analysis(analysis_prompt)
            
            # Add context and personality
            response = f"📊 **{symbol} Analysis**\\n\\n{analysis}\\n\\n"
            response += "💡 *Remember: This is analysis, not financial advice. Always do your own research and consider your risk tolerance!*"
            
            return response
            
        except Exception as e:
            self.logger.error(f"Stock analysis error: {e}")
            return f"I'm having trouble analyzing {symbol} right now. Could you try again in a moment? In the meantime, I recommend checking the latest market data and news for this stock."
    
    async def _handle_market_outlook(self, user_input: str) -> str:
        """Handle market outlook questions"""
        
        outlook_prompt = f"""
        Provide market outlook analysis based on: "{user_input}"
        
        Cover:
        1. Current market sentiment
        2. Key market drivers
        3. Sector performance highlights
        4. Risk factors to watch
        5. Trading opportunities
        6. Timeframe-specific outlook
        
        Style: Informative, balanced, actionable
        Length: 250-350 words
        """
        
        try:
            analysis = await self.llm_handler.get_analysis(outlook_prompt)
            
            response = f"🌐 **Market Outlook**\\n\\n{analysis}\\n\\n"
            response += "📈 *Market conditions can change rapidly. Stay informed and adjust your strategy accordingly!*"
            
            return response
            
        except Exception as e:
            self.logger.error(f"Market outlook error: {e}")
            return "I'm currently updating my market analysis. In the meantime, I recommend checking major market indices, sector rotations, and economic indicators for the latest market pulse."
    
    async def _handle_risk_assessment(self, user_input: str) -> str:
        """Handle risk assessment questions"""
        
        symbols = self.current_context.get('symbols_mentioned', [])
        
        risk_prompt = f"""
        Provide risk assessment for: "{user_input}"
        {"Focusing on symbols: " + ", ".join(symbols) if symbols else "General market risk"}
        
        Address:
        1. Volatility assessment
        2. Market risk factors
        3. Company-specific risks (if applicable)
        4. Portfolio impact considerations
        5. Risk mitigation strategies
        6. Position sizing recommendations
        
        Style: Risk-focused, practical, educational
        Length: 200-300 words
        """
        
        try:
            analysis = await self.llm_handler.get_analysis(risk_prompt)
            
            response = f"⚠️ **Risk Assessment**\\n\\n{analysis}\\n\\n"
            response += "🛡️ *Risk management is crucial for long-term trading success. Never risk more than you can afford to lose!*"
            
            return response
            
        except Exception as e:
            self.logger.error(f"Risk assessment error: {e}")
            return "Risk assessment is temporarily unavailable. As a general rule: diversify your holdings, use stop losses, position size appropriately, and never invest more than you can afford to lose."
    
    async def _handle_portfolio_advice(self, user_input: str) -> str:
        """Handle portfolio advice requests"""
        
        portfolio_prompt = f"""
        Provide portfolio advice for: "{user_input}"
        
        Include:
        1. Diversification strategies
        2. Asset allocation suggestions
        3. Sector balance recommendations
        4. Risk level considerations
        5. Rebalancing guidance
        6. Portfolio monitoring tips
        
        Style: Strategic, educational, personalized
        Length: 250-300 words
        """
        
        try:
            advice = await self.llm_handler.get_analysis(portfolio_prompt)
            
            response = f"💼 **Portfolio Strategy**\\n\\n{advice}\\n\\n"
            response += "📊 *Portfolio construction should align with your goals, timeline, and risk tolerance. Consider consulting a financial advisor for personalized advice.*"
            
            return response
            
        except Exception as e:
            self.logger.error(f"Portfolio advice error: {e}")
            return "Portfolio advice is temporarily unavailable. Key principles: diversify across sectors and asset classes, maintain appropriate risk levels, rebalance regularly, and align with your investment timeline."
    
    async def _handle_technical_analysis(self, user_input: str) -> str:
        """Handle technical analysis requests"""
        
        symbols = self.current_context.get('symbols_mentioned', [])
        
        if symbols:
            symbol = symbols[0]
            tech_prompt = f"""
            Provide technical analysis for {symbol} based on: "{user_input}"
            
            Cover:
            1. Chart pattern analysis
            2. Key support and resistance levels
            3. Technical indicators (RSI, MACD, etc.)
            4. Volume analysis
            5. Trend direction and strength
            6. Entry/exit points
            
            Style: Technical, specific, actionable
            Length: 200-300 words
            """
        else:
            tech_prompt = f"""
            Provide general technical analysis guidance for: "{user_input}"
            
            Explain:
            1. Relevant technical concepts
            2. How to apply these indicators
            3. What to look for in charts
            4. Common patterns and signals
            5. Risk management using technicals
            
            Style: Educational, practical
            Length: 200-300 words
            """
        
        try:
            analysis = await self.llm_handler.get_analysis(tech_prompt)
            
            response = f"📈 **Technical Analysis**\\n\\n{analysis}\\n\\n"
            response += "🎯 *Technical analysis is one tool among many. Combine with fundamental analysis and risk management for best results.*"
            
            return response
            
        except Exception as e:
            self.logger.error(f"Technical analysis error: {e}")
            return "Technical analysis is temporarily unavailable. Focus on key levels: support/resistance, trend lines, volume confirmation, and major moving averages for guidance."
    
    async def _handle_news_analysis(self, user_input: str) -> str:
        """Handle news and events analysis"""
        
        news_prompt = f"""
        Analyze news/events impact: "{user_input}"
        
        Discuss:
        1. Market impact assessment
        2. Affected sectors/stocks
        3. Timeline considerations
        4. Trading implications
        5. Risk/opportunity factors
        6. Historical context
        
        Style: Analytical, timely, strategic
        Length: 200-300 words
        """
        
        try:
            analysis = await self.llm_handler.get_analysis(news_prompt)
            
            response = f"📰 **News Impact Analysis**\\n\\n{analysis}\\n\\n"
            response += "⏰ *Market reactions to news can be swift and volatile. Stay updated and be prepared to adjust positions quickly.*"
            
            return response
            
        except Exception as e:
            self.logger.error(f"News analysis error: {e}")
            return "News analysis is temporarily unavailable. Monitor major financial news sources, earnings announcements, and economic data releases for market-moving events."
    
    async def _handle_general_question(self, user_input: str, user_id: str) -> str:
        """Handle general trading questions"""
        
        # Get conversation context
        recent_messages = await self.conversation_history.get_recent_messages(user_id, limit=5)
        context = self._build_conversation_context(recent_messages)
        
        general_prompt = f"""
        Answer this trading-related question: "{user_input}"
        
        Conversation context: {context}
        
        Provide:
        1. Direct answer to the question
        2. Relevant examples or context
        3. Practical applications
        4. Important considerations
        5. Follow-up suggestions
        
        Style: Helpful, educational, conversational
        Length: 150-250 words
        """
        
        try:
            answer = await self.llm_handler.get_analysis(general_prompt)
            
            response = f"💭 **Trading Insight**\\n\\n{answer}\\n\\n"
            response += "❓ *Have more questions? I'm here to help with any aspect of trading and investing!*"
            
            return response
            
        except Exception as e:
            self.logger.error(f"General question error: {e}")
            return "I'm having trouble processing your question right now. Could you rephrase it or try asking about a specific stock, trading concept, or market topic?"
    
    def _build_conversation_context(self, recent_messages: List[Dict]) -> str:
        """Build context from recent conversation"""
        if not recent_messages:
            return "New conversation"
        
        context_parts = []
        for msg in recent_messages[-3:]:  # Last 3 messages
            if msg['role'] == 'user':
                context_parts.append(f"User asked: {msg['content'][:100]}")
        
        return "; ".join(context_parts) if context_parts else "Ongoing conversation"
    
    def _update_context(self, user_input: str, intent: str):
        """Update conversation context"""
        
        # Update symbols discussed
        symbols = self.current_context.get('symbols_mentioned', [])
        for symbol in symbols:
            if symbol not in self.current_context['symbols_discussed']:
                self.current_context['symbols_discussed'].append(symbol)
        
        # Update analysis type
        self.current_context['last_analysis_type'] = intent
        
        # Update conversation theme
        if intent in ['stock_analysis', 'buy_sell_advice', 'technical_analysis']:
            self.current_context['conversation_theme'] = 'stock_focused'
        elif intent in ['market_outlook', 'news_analysis']:
            self.current_context['conversation_theme'] = 'market_focused'
        elif intent in ['portfolio_advice', 'risk_assessment']:
            self.current_context['conversation_theme'] = 'strategy_focused'
    
    def _get_random_response(self, response_type: str) -> str:
        """Get random response from predefined responses"""
        import random
        responses = self.quick_responses.get(response_type, ["I'm here to help with your trading questions!"])
        return random.choice(responses)
    
    def _get_error_response(self) -> str:
        """Get error response"""
        return """
        I'm experiencing some technical difficulties right now. Here's what I can still help you with:
        
        • Ask about specific stock symbols (AAPL, TSLA, etc.)
        • General trading strategies and concepts
        • Risk management principles
        • Market analysis basics
        
        Please try rephrasing your question, and I'll do my best to assist you!
        """
    
    def get_conversation_summary(self, user_id: str) -> Dict[str, Any]:
        """Get summary of current conversation context"""
        return {
            'symbols_discussed': self.current_context['symbols_discussed'],
            'conversation_theme': self.current_context.get('conversation_theme'),
            'last_analysis_type': self.current_context.get('last_analysis_type'),
            'chatbot_info': self.personality,
            'capabilities': [
                'Stock Analysis & Recommendations',
                'Market Outlook & Trends',
                'Technical Analysis',
                'Risk Assessment',
                'Portfolio Strategy',
                'News Impact Analysis',
                'Trading Education'
            ]
        }