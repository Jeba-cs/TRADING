# ui/dashboard.py
import streamlit as st
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta

class TradingDashboard:
    def __init__(self, symbols):
        self.symbols = symbols

    def render(self):
        st.title("Dashboard")

        if not self.symbols:
            st.write("No symbols provided.")
            return

        symbol = st.selectbox("Select Symbol", self.symbols)

        dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now())
        prices = pd.Series([100 + i for i in range(len(dates))], index=dates)

        fig = px.line(prices, x=dates, y=prices, title=f"{symbol} Price Chart")
        st.plotly_chart(fig)
