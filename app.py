import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, GRU
from sklearn.preprocessing import MinMaxScaler
import datetime
# import pickle
import streamlit as st
import model_building as m
import technical_analysis as t
import matplotlib.pyplot as plt
import mpld3
import streamlit.components.v1 as components
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands


with st.sidebar:

    st.markdown("# Stock Analysis & Forecasting")
    user_input = st.text_input('Enter Stock Name', "ADANIENT.NS")
    window = st.sidebar.slider('Select SMA Window (Days)', min_value=10, max_value=100, value=50)
    st.markdown("### Choose Date for your anaylsis")
    date_from = st.date_input("From",datetime.date(2020, 1, 1))
    date_to = st.date_input("To",datetime.date(2023, 2, 25))
    btn = st.button('Submit') 

#adding a button
if btn:
    df = yf.download(user_input, start=date_from, end=date_to)
    plotdf, future_predicted_values =m.create_model(df)


    st.markdown("### Original vs predicted close price")
    fig= plt.figure(figsize=(20,10))
    sns.lineplot(data=plotdf)
    st.pyplot(fig)

    st.markdown("### Next 10 days forecast")
    list_of_days = ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7","Day 8", "Day 9", "Day 10"]

    for i,j in zip(st.tabs(list_of_days),range(10)):
        with i:
            st.write(future_predicted_values.iloc[j:j+1])


    st.markdown("### Adj Close Price")
    fig= plt.figure(figsize=(20,10))
    t.last_2_years_price_plot(df)
    st.pyplot(fig)

    st.markdown("### Daily Percentage Changes")
    fig= plt.figure(figsize=(20,10))
    t.daily_percent_change_plot(df)
    st.pyplot(fig)
    
    st.markdown("### Daily Percentage Changes Histogram")
    fig= plt.figure(figsize=(20,10))
    t.daily_percent_change_histogram(df)
    st.pyplot(fig)

    st.markdown("### Trend Analysis")
    fig= plt.figure(figsize=(20,10))
    t.trend_pie_chart(df)
    st.pyplot(fig)

    st.markdown("### Volume Plot")
    fig= plt.figure(figsize=(20,10))
    t.volume_plot(df)
    st.pyplot(fig)

    st.markdown("### Volume Plot")
    fig= plt.figure(figsize=(20,10))
    t.correlation_plot(df)
    st.pyplot(fig)

    st.markdown("### Volatility Plot")
    fig= plt.figure(figsize=(20,10))
    t.volatility_plot(df)
    st.pyplot(fig)


    st.markdown("# Technical Analysis")

    st.markdown("## MACD Indicator")
    
    fig= plt.figure(figsize=(20,10))
    t.plot_price_and_signals(t.get_macd(df),'MACD')
    st.pyplot(fig)

    fig= plt.figure(figsize=(20,10))
    t.plot_macd(df)
    st.pyplot(fig)

    st.write(" ***:blue[Strategy:]:***")
    st.write(":red[Sell  Signal:] The cross over: When the MACD line is below the signal line.")
    st.write(":green[Buy Signal:] The cross over: When the MACD line is above the signal line.")

    st.markdown("## RSI Indicator")

    fig= plt.figure(figsize=(20,10))
    t.plot_price_and_signals(t.get_rsi(df),'RSI')
    st.pyplot(fig)

    fig= plt.figure(figsize=(20,10))
    t.plot_rsi(df)
    st.pyplot(fig)

    st.write(" ***:blue[Strategy:]:***")
    st.write(":red[Sell  Signal:] When RSI increases above 70%")
    st.write(":green[Buy Signal:] When RSI decreases below 30%.")


    st.markdown("## Bollinger Indicator")

    fig= plt.figure(figsize=(20,10))
    t.plot_price_and_signals(t.get_bollinger_bands(df),'Bollinger_Bands')
    st.pyplot(fig)

    fig= plt.figure(figsize=(20,10))
    t.plot_bollinger_bands(df)
    st.pyplot(fig)

    st.write(" ***:blue[Strategy:]:***")
    st.write(":red[Sell  Signal:] As soon as the market price touches the upper Bollinger band")
    st.write(":green[Buy Signal:] As soon as the market price touches the lower Bollinger band")
    

    st.markdown("## SMA Indicator")
    st.title('Simple Moving Average (SMA) Plot')
    st.write(f'Ticker Symbol: {user_input}')
    st.write(f'SMA Window: {window} Days')
    fig = t.sma_plot(user_input, window)
    st.pyplot(fig)
    st.write(" ***:blue[Strategy:]:***")
    st.write(":red[Sell  Signal:] If 50-day moving average < 200-day moving average, it is a bearish signal")
    st.write(":green[Buy Signal:] If 50-day moving average > 200-day moving average, it is a bullish signal")


    st.markdown("## EMA Indicator")
    st.title('Exponential Moving Average (EMA) Plot')
    st.write(f'Ticker Symbol: {user_input}')
    fig = t.ema_plot(user_input, date_from, date_to)
    st.pyplot(fig)
    st.write(" ***:blue[Strategy:]:***")
    st.write(":red[Sell  Signal:] When the 50-day EMA crosses below the 200-day EMA, it is considered a bearish signal, indicating that the stock price may continue to fall. This is because the recent prices are lower than the longer-term prices")
    st.write(":green[Buy Signal:] When the 50-day EMA crosses above the 200-day EMA, it is considered a bullish signal, indicating that the stock price may continue to rise. This is because the recent prices are higher than the longer-term prices")

else:
    st.write('Please click on the submit to get the analysis') #displayed when the button is unclicked

