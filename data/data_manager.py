# data/data_manager.py
"""
Data manager for fetching market data, news, sentiment, and fundamentals
"""

import yfinance as yf
import pandas as pd

class DataManager:
    def __init__(self):
        pass

    def get_price_data(self, symbol, period="1y", interval="1d"):
        data = yf.download(symbol, period=period, interval=interval)
        return data

    def get_current_price(self, symbol):
        ticker = yf.Ticker(symbol)
        price = ticker.info.get('currentPrice', None)
        return price
