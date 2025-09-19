# models/portfolio.py
from datetime import datetime, timedelta
class Portfolio:
    def __init__(self):
        self.positions = {}
        self.cash = 100000  # starting cash in USD

    def add_position(self, symbol, quantity, price):
        if symbol in self.positions:
            self.positions[symbol]['quantity'] += quantity
            # Average price update (simplified)
            self.positions[symbol]['price'] = price
        else:
            self.positions[symbol] = {'quantity': quantity, 'price': price}
        self.cash -= quantity * price

    def remove_position(self, symbol, quantity, price):
        if symbol in self.positions:
            self.positions[symbol]['quantity'] -= quantity
            if self.positions[symbol]['quantity'] <= 0:
                del self.positions[symbol]
            self.cash += quantity * price

    def get_position(self, symbol):
        return self.positions.get(symbol, {'quantity': 0, 'price': 0})

    def get_portfolio_value(self, current_prices):
        total_value = self.cash
        for symbol, pos in self.positions.items():
            total_value += pos['quantity'] * current_prices.get(symbol, 0)
        return total_value

    def get_total_value(self, current_prices=None):
        """
        Returns total portfolio value (cash + all positions marked to market).

        Args:
            current_prices (dict): symbol -> price

        If current_prices is None, uses the last purchase price for each symbol.
        """
        total = self.cash
        for symbol, pos in self.positions.items():
            price = pos['price']
            if current_prices and symbol in current_prices:
                price = current_prices[symbol]
            total += pos['quantity'] * price
        return total

    def get_daily_pnl(self, price_history=None):
        """
        Returns a dictionary of date -> daily PnL for the portfolio.
        Expects price_history: dict of symbol -> pandas Series (date->price)
        If price_history is None, returns last 7 days PnL as zeroes.
        """
        import pandas as pd

        if price_history is None or not price_history:
            # Return zeroed dummy data for last 7 days
            today = datetime.now().date()
            dates = [today - timedelta(days=i) for i in range(7)][::-1]
            return {d: 0.0 for d in dates}

        # Build PnL by summing daily change for each position
        pnl = {}
        # Make sure all series align to full date index
        date_index = None
        for series in price_history.values():
            if date_index is None or len(series) > len(date_index):
                date_index = series.index

        for date in date_index:
            day_pnl = 0.0
            for symbol, pos in self.positions.items():
                qty = pos['quantity']
                ser = price_history.get(symbol)
                if ser is not None and date in ser.index:
                    price_now = ser.loc[date]
                    # If price history covers previous day, calc change
                    prev_idx = ser.index.get_loc(date)
                    if prev_idx > 0:
                        price_prev = ser.iloc[prev_idx - 1]
                        change = (price_now - price_prev) * qty
                        day_pnl += change
            pnl[date.date()] = day_pnl

        return pnl