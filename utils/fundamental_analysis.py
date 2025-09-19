# utils/fundamental_analysis.py
"""
Fundamental Analysis utilities
Provides basic fundamental metrics and scoring
"""

import yfinance as yf

class FundamentalAnalyzer:
    def __init__(self):
        pass

    def get_fundamental_data(self, symbol):
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return info

    def analyze_fundamentals(self, symbol):
        info = self.get_fundamental_data(symbol)
        pe_ratio = info.get("trailingPE", None)
        pb_ratio = info.get("priceToBook", None)
        roe = info.get("returnOnEquity", None)
        debt_equity = info.get("debtToEquity", None)
        revenue_growth = info.get("revenueGrowth", None)
        earnings_growth = info.get("earningsQuarterlyGrowth", None)

        scores = {
            "pe_score": self._score_pe(pe_ratio),
            "pb_score": self._score_pb(pb_ratio),
            "roe_score": self._score_roe(roe),
            "debt_equity_score": self._score_debt_equity(debt_equity),
            "revenue_growth_score": self._score_growth(revenue_growth),
            "earnings_growth_score": self._score_growth(earnings_growth),
        }

        overall_score = sum(scores.values()) / len(scores)
        return overall_score, scores

    def _score_pe(self, pe):
        if pe is None:
            return 0.5
        if pe < 15:
            return 1.0
        if pe < 25:
            return 0.7
        return 0.3

    def _score_pb(self, pb):
        if pb is None:
            return 0.5
        if pb < 1.5:
            return 1.0
        if pb < 3:
            return 0.6
        return 0.3

    def _score_roe(self, roe):
        if roe is None:
            return 0.5
        if roe > 0.15:
            return 1.0
        if roe > 0.1:
            return 0.7
        return 0.3

    def _score_debt_equity(self, de):
        if de is None:
            return 0.5
        if de < 0.3:
            return 1.0
        if de < 0.6:
            return 0.7
        return 0.2

    def _score_growth(self, growth):
        if growth is None:
            return 0.5
        if growth > 0.1:
            return 1.0
        if growth > 0.05:
            return 0.7
        return 0.4
