# execution/trade_executor.py
"""
Simple trade executor mock for paper trades
"""

class TradeExecutor:
    def __init__(self):
        self.trades = []

    def execute_trade(self, symbol, action, quantity):
        trade = {
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'status': 'filled'
        }
        self.trades.append(trade)
        return trade
