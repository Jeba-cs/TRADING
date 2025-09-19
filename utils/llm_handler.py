# utils/llm_handler.py
"""
LLM Handler for AI-powered analysis and insights
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
import json
import openai
from datetime import datetime

class LLMHandler:
    """
    Handles interactions with Large Language Models for trading analysis
    Supports multiple LLM providers (OpenAI, Google, Anthropic)
    """
    
    def __init__(self, provider: str = "openai", model: str = "gpt-4"):
        self.logger = logging.getLogger("LLMHandler")
        self.provider = provider.lower()
        self.model = model
        
        # Initialize API clients based on provider
        self._initialize_provider()
        
        # Analysis templates
        self.analysis_templates = {
            'stock_analysis': self._get_stock_analysis_template(),
            'market_outlook': self._get_market_outlook_template(),
            'risk_assessment': self._get_risk_assessment_template(),
            'technical_analysis': self._get_technical_analysis_template()
        }
    
    def _initialize_provider(self):
        """Initialize the selected LLM provider"""
        try:
            if self.provider == "openai":
                # OpenAI configuration
                openai.api_key = "your-openai-api-key-here"  # Replace with actual key
                self.client = openai
                
            elif self.provider == "google":
                # Google Gemini configuration
                import google.generativeai as genai
                genai.configure(api_key="your-google-api-key-here")
                self.client = genai.GenerativeModel(self.model)
                
            elif self.provider == "anthropic":
                # Anthropic Claude configuration
                import anthropic
                self.client = anthropic.Anthropic(api_key="your-anthropic-api-key-here")
                
            else:
                raise ValueError(f"Unsupported LLM provider: {self.provider}")
                
            self.logger.info(f"Initialized LLM provider: {self.provider} with model: {self.model}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM provider {self.provider}: {e}")
            # Fallback to mock mode
            self.provider = "mock"
            self.client = None
    
    async def get_analysis(self, prompt: str, analysis_type: str = "general", 
                          max_tokens: int = 2000, temperature: float = 0.3) -> str:
        """
        Get AI analysis for given prompt
        
        Args:
            prompt: Analysis prompt
            analysis_type: Type of analysis for template selection
            max_tokens: Maximum response tokens
            temperature: Response creativity (0.0-1.0)
            
        Returns:
            str: AI analysis response
        """
        try:
            # Enhance prompt with template if available
            enhanced_prompt = self._enhance_prompt(prompt, analysis_type)
            
            # Get response based on provider
            if self.provider == "openai":
                response = await self._get_openai_response(enhanced_prompt, max_tokens, temperature)
            elif self.provider == "google":
                response = await self._get_google_response(enhanced_prompt, max_tokens, temperature)
            elif self.provider == "anthropic":
                response = await self._get_anthropic_response(enhanced_prompt, max_tokens, temperature)
            else:
                # Mock response for testing/demo
                response = self._get_mock_response(enhanced_prompt, analysis_type)
            
            # Log the interaction
            self.logger.info(f"Generated {analysis_type} analysis (length: {len(response)} chars)")
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating LLM analysis: {e}")
            return self._get_fallback_response(analysis_type)
    
    def _enhance_prompt(self, prompt: str, analysis_type: str) -> str:
        """Enhance prompt with appropriate template"""
        
        if analysis_type in self.analysis_templates:
            template = self.analysis_templates[analysis_type]
            enhanced_prompt = f"{template}\\n\\nSpecific Request: {prompt}"
        else:
            # General trading analysis template
            enhanced_prompt = f"""
            You are an expert financial analyst and trading advisor with deep knowledge of:
            - Technical and fundamental analysis
            - Risk management and portfolio theory
            - Market psychology and behavioral finance
            - Global economics and market dynamics
            
            Provide professional, balanced analysis that:
            - Is data-driven and objective
            - Acknowledges risks and uncertainties
            - Offers actionable insights
            - Maintains appropriate disclaimers
            
            Request: {prompt}
            
            Format your response to be clear, structured, and professional.
            """
        
        return enhanced_prompt
    
    async def _get_openai_response(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Get response from OpenAI API"""
        try:
            response = await asyncio.to_thread(
                self.client.ChatCompletion.create,
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional financial analyst and trading advisor."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                frequency_penalty=0.1,
                presence_penalty=0.1
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            raise
    
    async def _get_google_response(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Get response from Google Gemini API"""
        try:
            # Configure generation parameters
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "top_p": 0.9,
                "top_k": 40
            }
            
            response = await asyncio.to_thread(
                self.client.generate_content,
                prompt,
                generation_config=generation_config
            )
            
            return response.text.strip()
            
        except Exception as e:
            self.logger.error(f"Google API error: {e}")
            raise
    
    async def _get_anthropic_response(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Get response from Anthropic Claude API"""
        try:
            response = await asyncio.to_thread(
                self.client.messages.create,
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.content[0].text.strip()
            
        except Exception as e:
            self.logger.error(f"Anthropic API error: {e}")
            raise
    
    def _get_mock_response(self, prompt: str, analysis_type: str) -> str:
        """Generate mock response for testing/demo purposes"""
        
        mock_responses = {
            'stock_analysis': """
            **Technical Analysis Summary:**
            The stock is currently trading near key resistance levels with mixed momentum indicators. 
            RSI shows moderate overbought conditions while MACD indicates potential bullish crossover.
            
            **Fundamental Outlook:**
            Strong earnings growth and solid balance sheet support long-term value. However, 
            current valuation appears stretched compared to industry peers.
            
            **Risk Assessment:**
            Moderate risk profile with key support at $XX. Watch for volume confirmation 
            on any breakout attempts. Market volatility remains a concern.
            
            **Recommendation:**
            Consider position sizing carefully. Good for swing trading with defined stops.
            Long-term investors may want to wait for better entry points.
            
            *This is sample analysis for demonstration purposes.*
            """,
            
            'market_outlook': """
            **Current Market Environment:**
            Markets are showing resilience despite economic headwinds. Technology and 
            healthcare sectors continue to outperform while energy faces pressure.
            
            **Key Factors:**
            - Central bank policy remains accommodative
            - Earnings growth moderating but still positive  
            - Geopolitical tensions creating selective volatility
            
            **Outlook:**
            Cautiously optimistic for next 3-6 months. Favor quality growth stocks
            with strong fundamentals. Maintain defensive positioning in uncertain times.
            
            *This is sample analysis for demonstration purposes.*
            """,
            
            'risk_assessment': """
            **Risk Profile Analysis:**
            Current market conditions suggest elevated volatility ahead. Key risks include:
            
            1. **Market Risk:** Increased correlation during stress events
            2. **Liquidity Risk:** Potential for rapid spread widening
            3. **Concentration Risk:** Over-exposure to specific sectors
            
            **Mitigation Strategies:**
            - Implement proper position sizing (2-3% per position)
            - Use stop-loss orders effectively
            - Maintain portfolio diversification
            - Consider hedging with inverse ETFs or options
            
            **Risk-Adjusted Returns:**
            Focus on Sharpe ratio optimization rather than pure returns.
            
            *This is sample analysis for demonstration purposes.*
            """,
            
            'general': """
            Based on current market conditions and the specific factors you mentioned, 
            here's a comprehensive analysis:
            
            **Key Considerations:**
            The market environment presents both opportunities and risks that require 
            careful navigation. Technical indicators suggest mixed signals while 
            fundamental factors remain supportive.
            
            **Strategic Approach:**
            1. Maintain disciplined risk management
            2. Focus on high-conviction positions
            3. Stay flexible with market changes
            4. Monitor key support/resistance levels
            
            **Action Items:**
            Consider scaling into positions gradually and maintain appropriate 
            stop-loss levels. Market timing remains challenging, so focus on 
            risk-adjusted returns rather than trying to time perfect entries.
            
            *This analysis is for educational purposes and not financial advice.*
            """
        }
        
        return mock_responses.get(analysis_type, mock_responses['general'])
    
    def _get_fallback_response(self, analysis_type: str) -> str:
        """Get fallback response when LLM fails"""
        return f"""
        I'm currently experiencing technical difficulties with AI analysis. 
        
        For {analysis_type}, I recommend:
        - Consulting multiple data sources
        - Applying fundamental risk management principles
        - Considering current market volatility
        - Maintaining appropriate position sizing
        
        Please try again shortly, or proceed with manual analysis using 
        standard technical and fundamental indicators.
        """
    
    def _get_stock_analysis_template(self) -> str:
        """Template for stock analysis"""
        return """
        You are conducting a comprehensive stock analysis. Structure your response as:
        
        1. **Current Price Action & Technical Setup**
        2. **Fundamental Strengths & Weaknesses**  
        3. **Risk Factors & Catalysts**
        4. **Short-term vs Long-term Outlook**
        5. **Trading Strategy Recommendations**
        
        Keep analysis balanced, acknowledge uncertainties, and provide specific price levels where relevant.
        """
    
    def _get_market_outlook_template(self) -> str:
        """Template for market outlook"""
        return """
        Provide a market outlook analysis covering:
        
        1. **Current Market Regime & Sentiment**
        2. **Key Economic & Policy Drivers**
        3. **Sector Rotation & Leadership**
        4. **Technical Market Structure**
        5. **Risk Scenarios & Probability Assessment**
        6. **Strategic Positioning Recommendations**
        
        Focus on actionable insights for different time horizons (short/medium/long-term).
        """
    
    def _get_risk_assessment_template(self) -> str:
        """Template for risk assessment"""
        return """
        Conduct thorough risk assessment including:
        
        1. **Volatility & Drawdown Analysis**
        2. **Correlation & Concentration Risks**
        3. **Liquidity & Market Structure Risks**
        4. **Macro & Event Risks**
        5. **Position Sizing & Risk Budget**
        6. **Hedging & Risk Mitigation Strategies**
        
        Provide specific, actionable risk management recommendations.
        """
    
    def _get_technical_analysis_template(self) -> str:
        """Template for technical analysis"""
        return """
        Perform technical analysis covering:
        
        1. **Trend Analysis & Market Structure**
        2. **Key Support & Resistance Levels**
        3. **Momentum & Oscillator Signals**
        4. **Volume & Price Action Patterns**
        5. **Chart Patterns & Formations**
        6. **Entry/Exit Points & Risk Management**
        
        Provide specific price targets and stop-loss levels where applicable.
        """
    
    async def get_batch_analysis(self, requests: List[Dict[str, Any]]) -> List[str]:
        """
        Process multiple analysis requests concurrently
        
        Args:
            requests: List of analysis requests with 'prompt' and 'type' keys
            
        Returns:
            List[str]: List of analysis responses
        """
        tasks = []
        for request in requests:
            task = asyncio.create_task(
                self.get_analysis(
                    request.get('prompt', ''),
                    request.get('type', 'general'),
                    request.get('max_tokens', 1000),
                    request.get('temperature', 0.3)
                )
            )
            tasks.append(task)
        
        try:
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions in responses
            processed_responses = []
            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    self.logger.error(f"Batch request {i} failed: {response}")
                    processed_responses.append(self._get_fallback_response(
                        requests[i].get('type', 'general')
                    ))
                else:
                    processed_responses.append(response)
            
            return processed_responses
            
        except Exception as e:
            self.logger.error(f"Batch analysis error: {e}")
            return [self._get_fallback_response('general')] * len(requests)
    
    def get_provider_status(self) -> Dict[str, Any]:
        """Get current LLM provider status"""
        return {
            'provider': self.provider,
            'model': self.model,
            'status': 'active' if self.client else 'inactive',
            'supported_types': list(self.analysis_templates.keys()),
            'last_updated': datetime.now().isoformat()
        }