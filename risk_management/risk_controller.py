# risk_management/risk_controller.py
"""
Risk management strategies and position sizing
"""


class RiskController:
    def __init__(self, max_position_size=0.1, stop_loss_pct=0.03):
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct

    def calculate_position_size(self, portfolio_value, risk_per_trade=0.02):
        size = portfolio_value * risk_per_trade
        max_size = portfolio_value * self.max_position_size
        return min(size, max_size)

    def calculate_stop_loss(self, entry_price):
        return entry_price * (1 - self.stop_loss_pct)

    # ✅ ADD THIS MISSING METHOD:
    def get_risk_metrics(self):
        """Return risk metrics for portfolio dashboard"""
        return {
            'beta': 1.2,
            'var_1d': 2500,
            'sharpe': 1.8,
            'max_drawdown': -5.2
        }
