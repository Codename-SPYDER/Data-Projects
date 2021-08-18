import yfinance as yf
import streamlit as st
import pandas as pd

st.write("""

# NVDA Stock Ticker

The last 3 years of NVDA closing price is shown below. NIVIDIA CORP = 'NVDA'

""")

tickerSymbol = 'NVDA'

tickerData = yf.Ticker(tickerSymbol)

tickerDf = tickerData.history(period = '1d', start = '2018-1-1', end = '2021-1-1')
#.Open, .Close, .Volume, .Dividends, .StockSplit 

st.line_chart(tickerDf.Close)

st.write("""

The last 3 years of trade volume for NVDA is shown below.

""")

st.line_chart(tickerDf.Volume)
