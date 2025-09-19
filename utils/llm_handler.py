# utils/llm_handler.py
"""
LLM Handler supporting Google Gemini and OpenAI
"""
import os
import google.generativeai as genai
from typing import Optional


class LLMHandler:
    def __init__(self, provider: str = "google", model: str = "gemini-pro"):
        self.provider = provider.lower()
        self.model = model
        self.client = None
        self._initialize_provider()

    def _initialize_provider(self):
        """Initialize the selected LLM provider"""
        try:
            if self.provider == "google":
                api_key = os.getenv("GEMINI_API_KEY")
                if not api_key:
                    raise ValueError("GEMINI_API_KEY not found in environment variables")
                genai.configure(api_key=api_key)
                self.client = genai.GenerativeModel(self.model)
                print(f"✅ Initialized Google Gemini with model: {self.model}")

            elif self.provider == "openai":
                try:
                    import openai
                    api_key = os.getenv("OPENAI_API_KEY")
                    if not api_key:
                        raise ValueError("OPENAI_API_KEY not found in environment variables")
                    openai.api_key = api_key
                    self.client = openai
                    print(f"✅ Initialized OpenAI with model: {self.model}")
                except ImportError:
                    raise ImportError("OpenAI library not installed. Run: pip install openai")

            else:
                raise ValueError(f"Unsupported provider: {self.provider}")

        except Exception as e:
            print(f"❌ Error initializing {self.provider}: {str(e)}")
            self.client = None

    def generate_response(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate response using the configured LLM provider"""
        if not self.client:
            return "❌ LLM not properly initialized. Check your API key configuration."

        try:
            if self.provider == "google":
                response = self.client.generate_content(prompt)
                return response.text if response.text else "No response generated."

            elif self.provider == "openai":
                response = self.client.Completion.create(
                    engine=self.model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=0.7
                )
                return response.choices[0].text.strip()

        except Exception as e:
            return f"❌ Error generating response: {str(e)}"

    def analyze_trading_question(self, question: str, market_data: Optional[dict] = None) -> str:
        """Analyze trading-related questions with market context"""
        context = ""
        if market_data:
            context = f"\nMarket Data Context: {market_data}"

        prompt = f"""
        You are an expert AI trading analyst. Answer the following trading question with:
        1. Clear, actionable insights
        2. Risk considerations
        3. Market analysis if relevant
        4. Educational context

        Question: {question}
        {context}

        Provide a comprehensive but concise response (max 300 words):
        """

        return self.generate_response(prompt)
