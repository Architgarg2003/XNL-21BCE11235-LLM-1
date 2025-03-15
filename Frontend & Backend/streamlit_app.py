# import streamlit as st
# import yfinance as yf
# import pandas as pd
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# from gnews import GNews
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# import plotly.express as px
# from datetime import datetime, timedelta

# # Set up Streamlit page
# st.set_page_config(page_title="Advanced Stock & News Dashboard", layout="wide")
# st.title("📈 Advanced Real-Time Stock & Sentiment Dashboard")

# # User Inputs
# col1, col2 = st.columns(2)
# with col1:
#     stock_symbol = st.text_input("Enter Stock Symbol (e.g., TSLA, AAPL):", "TSLA").upper()
# with col2:
#     time_period = st.selectbox("Select Time Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y"])

# # Initialize sentiment analyzer
# analyzer = SentimentIntensityAnalyzer()

# # Function to get enhanced stock data
# def get_stock_data(symbol, period):
#     stock = yf.Ticker(symbol)
#     hist = stock.history(period=period)
    
#     # Calculate technical indicators
#     hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
#     hist['RSI'] = 100 - (100 / (1 + hist['Close'].diff(1).clip(lower=0).rolling(14).mean() / 
#                           hist['Close'].diff(1).clip(upper=0).abs().rolling(14).mean()))
#     return hist

# # Function to fetch recent news using GNews
# def fetch_news(symbol):
#     google_news = GNews()
#     google_news.period = '7d'  # Last 7 days
#     google_news.max_results = 20  # Limit results
#     try:
#         news = google_news.get_news(f'{symbol} stock')
#         return [(article['title'], 
#                 article['published date'], 
#                 article['url']) for article in news][:10]  # Return top 10
#     except:
#         return []

# # Enhanced sentiment analysis with VADER
# def analyze_sentiment(headlines):
#     results = []
#     for headline, date, url in headlines:
#         vs = analyzer.polarity_scores(headline)
#         results.append({
#             "headline": headline,
#             "date": date,
#             "url": url,
#             "positive": vs["pos"],
#             "neutral": vs["neu"],
#             "negative": vs["neg"],
#             "compound": vs["compound"]
#         })
#     return results

# # Display fundamental data
# def show_fundamentals(symbol):
#     stock = yf.Ticker(symbol)
#     info = stock.info
#     col1, col2, col3, col4 = st.columns(4)
#     fundamentals = {
#         "Market Cap": f"${info.get('marketCap', 'N/A')/1e9:.1f}B" if info.get('marketCap') else 'N/A',
#         "PE Ratio": info.get('trailingPE', 'N/A'),
#         "52 Week High": info.get('fiftyTwoWeekHigh', 'N/A'),
#         "Volume Avg.": f"{info.get('averageVolume', 'N/A'):,}"
#     }
#     for i, (k, v) in enumerate(fundamentals.items()):
#         [col1, col2, col3, col4][i].metric(k, v)

# # Main Dashboard
# if st.button("Analyze Market Data"):
#     st.header(f"📊 {stock_symbol} Comprehensive Analysis")
    
#     # Fundamentals Section
#     show_fundamentals(stock_symbol)
    
#     # Stock Data Visualization
#     st.subheader("Technical Analysis Chart")
#     data = get_stock_data(stock_symbol, time_period)
    
#     # Create subplots
#     fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
#                        vertical_spacing=0.05,
#                        row_heights=[0.5, 0.25, 0.25])
    
#     # Candlestick chart
#     fig.add_trace(go.Candlestick(x=data.index,
#                                 open=data['Open'],
#                                 high=data['High'],
#                                 low=data['Low'],
#                                 close=data['Close'],
#                                 name='Price'), row=1, col=1)
    
#     # SMA
#     fig.add_trace(go.Scatter(x=data.index, 
#                             y=data['SMA_20'],
#                             line=dict(color='orange', width=2),
#                             name='20 SMA'), row=1, col=1)
    
#     # Volume
#     fig.add_trace(go.Bar(x=data.index, 
#                         y=data['Volume'],
#                         name='Volume'), row=2, col=1)
    
#     # RSI
#     fig.add_trace(go.Scatter(x=data.index, 
#                             y=data['RSI'],
#                             line=dict(color='purple', width=2),
#                             name='RSI'), row=3, col=1)
    
#     fig.update_layout(height=800, showlegend=False)
#     st.plotly_chart(fig, use_container_width=True)

#     # News Analysis Section
#     st.subheader("📰 Latest News Analysis")
#     news_articles = fetch_news(stock_symbol)
    
#     if news_articles:
#         sentiment_data = analyze_sentiment(news_articles)
#         df = pd.DataFrame(sentiment_data)
        
#         # Sentiment Metrics
#         avg_sentiment = df['compound'].mean()
#         sentiment_category = "Positive" if avg_sentiment > 0.05 else "Negative" if avg_sentiment < -0.05 else "Neutral"
        
#         col1, col2 = st.columns(2)
#         col1.metric("Average Sentiment Score", f"{avg_sentiment:.2f}", sentiment_category)
#         col2.metric("Total Articles Analyzed", len(df))
        
#         # Sentiment Visualization
#         fig = px.pie(df, names=['Positive', 'Neutral', 'Negative'], 
#                     values=[df['positive'].mean(), df['neutral'].mean(), df['negative'].mean()],
#                     title="Sentiment Distribution",
#                     color_discrete_sequence=['green', 'gray', 'red'])
#         st.plotly_chart(fig, use_container_width=True)
        
#         # News List
#         st.subheader("Recent News Headlines")
#         for idx, article in enumerate(sentiment_data):
#             with st.expander(f"{article['headline']} (Sentiment: {article['compound']:.2f})"):
#                 st.caption(f"Published: {article['date']}")
#                 st.markdown(f"[Read more]({article['url']})")
#     else:
#         st.warning("No recent news articles found for this stock")

# # Add a refresh button
# st.sidebar.markdown("---")
# if st.sidebar.button("🔄 Refresh Data"):
#     st.rerun()
# st.sidebar.info("Note: Data updates may take a few seconds")






# import streamlit as st
# import yfinance as yf
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# from gnews import GNews
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# import plotly.express as px
# from datetime import datetime, timedelta
# from sklearn.linear_model import LinearRegression

# # Add Prophet with error handling
# try:
#     from fbprophet import Prophet
# except ImportError:
#     st.error("Please install fbprophet: pip install fbprophet")

# # Set up Streamlit page
# st.set_page_config(page_title="Advanced Stock & News Dashboard", layout="wide")
# st.title("📈 Advanced Real-Time Stock & Sentiment Dashboard")

# # User Inputs
# col1, col2, col3 = st.columns(3)
# with col1:
#     stock_symbol = st.text_input("Enter Stock Symbol (e.g., TSLA, AAPL):", "TSLA").upper()
# with col2:
#     time_period = st.selectbox("Select Time Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y"])
# with col3:
#     forecast_days = st.number_input("Forecast Period (days)", min_value=1, max_value=30, value=7)

# # Initialize sentiment analyzer
# analyzer = SentimentIntensityAnalyzer()

# # Function to get enhanced stock data
# def get_stock_data(symbol, period):
#     stock = yf.Ticker(symbol)
#     hist = stock.history(period=period)
    
#     # Calculate technical indicators
#     hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
#     hist['RSI'] = 100 - (100 / (1 + hist['Close'].diff(1).clip(lower=0).rolling(14).mean() / 
#                           hist['Close'].diff(1).clip(upper=0).abs().rolling(14).mean()))
#     return hist

# # Function to fetch recent news using GNews
# def fetch_news(symbol):
#     google_news = GNews()
#     google_news.period = '7d'  # Last 7 days
#     google_news.max_results = 20  # Limit results
#     try:
#         news = google_news.get_news(f'{symbol} stock')
#         return [(article['title'], 
#                 article['published date'], 
#                 article['url']) for article in news][:10]  # Return top 10
#     except:
#         return []

# # Enhanced sentiment analysis with VADER
# def analyze_sentiment(headlines):
#     results = []
#     for headline, date, url in headlines:
#         vs = analyzer.polarity_scores(headline)
#         results.append({
#             "headline": headline,
#             "date": date,
#             "url": url,
#             "positive": vs["pos"],
#             "neutral": vs["neu"],
#             "negative": vs["neg"],
#             "compound": vs["compound"]
#         })
#     return results

# # Display fundamental data
# def show_fundamentals(symbol):
#     stock = yf.Ticker(symbol)
#     info = stock.info
#     col1, col2, col3, col4 = st.columns(4)
#     fundamentals = {
#         "Market Cap": f"${info.get('marketCap', 'N/A')/1e9:.1f}B" if info.get('marketCap') else 'N/A',
#         "PE Ratio": info.get('trailingPE', 'N/A'),
#         "52 Week High": info.get('fiftyTwoWeekHigh', 'N/A'),
#         "Volume Avg.": f"{info.get('averageVolume', 'N/A'):,}"
#     }
#     for i, (k, v) in enumerate(fundamentals.items()):
#         [col1, col2, col3, col4][i].metric(k, v)

# # Main Dashboard
# if st.button("Analyze Market Data"):
#     st.header(f"📊 {stock_symbol} Comprehensive Analysis")
    
#     # Fundamentals Section
#     show_fundamentals(stock_symbol)
    
#     # Stock Data Visualization
#     st.subheader("Technical Analysis Chart")
#     data = get_stock_data(stock_symbol, time_period)
    
#     # Create subplots
#     fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
#                        vertical_spacing=0.05,
#                        row_heights=[0.5, 0.25, 0.25])
    
#     # Candlestick chart
#     fig.add_trace(go.Candlestick(x=data.index,
#                                 open=data['Open'],
#                                 high=data['High'],
#                                 low=data['Low'],
#                                 close=data['Close'],
#                                 name='Price'), row=1, col=1)
    
#     # SMA
#     fig.add_trace(go.Scatter(x=data.index, 
#                             y=data['SMA_20'],
#                             line=dict(color='orange', width=2),
#                             name='20 SMA'), row=1, col=1)
    
#     # Volume
#     fig.add_trace(go.Bar(x=data.index, 
#                         y=data['Volume'],
#                         name='Volume'), row=2, col=1)
    
#     # RSI
#     fig.add_trace(go.Scatter(x=data.index, 
#                             y=data['RSI'],
#                             line=dict(color='purple', width=2),
#                             name='RSI'), row=3, col=1)
    
#     fig.update_layout(height=800, showlegend=False)
#     st.plotly_chart(fig, use_container_width=True)

#     # News Analysis Section
#     st.subheader("📰 Latest News Analysis")
#     news_articles = fetch_news(stock_symbol)
    
#     if news_articles:
#         sentiment_data = analyze_sentiment(news_articles)
#         df = pd.DataFrame(sentiment_data)
        
#         # Sentiment Metrics
#         avg_sentiment = df['compound'].mean()
#         sentiment_category = "Positive" if avg_sentiment > 0.05 else "Negative" if avg_sentiment < -0.05 else "Neutral"
        
#         col1, col2 = st.columns(2)
#         col1.metric("Average Sentiment Score", f"{avg_sentiment:.2f}", sentiment_category)
#         col2.metric("Total Articles Analyzed", len(df))
        
#         # Sentiment Visualization
#         fig = px.pie(df, names=['Positive', 'Neutral', 'Negative'], 
#                     values=[df['positive'].mean(), df['neutral'].mean(), df['negative'].mean()],
#                     title="Sentiment Distribution",
#                     color_discrete_sequence=['green', 'gray', 'red'])
#         st.plotly_chart(fig, use_container_width=True)
        
#         # News List
#         st.subheader("Recent News Headlines")
#         for idx, article in enumerate(sentiment_data):
#             with st.expander(f"{article['headline']} (Sentiment: {article['compound']:.2f})"):
#                 st.caption(f"Published: {article['date']}")
#                 st.markdown(f"[Read more]({article['url']})")
#     else:
#         st.warning("No recent news articles found for this stock")

#     # Market Trend Prediction Section
#     st.header("📈 Market Trend Prediction")
    
#     # NLP-Based Trend Prediction
#     st.subheader("NLP-Based Trend Analysis")
#     if news_articles:
#         try:
#             df_sentiment = pd.DataFrame(sentiment_data)
#             df_sentiment['date'] = pd.to_datetime(df_sentiment['date']).dt.date
#             daily_sentiment = df_sentiment.groupby('date')['compound'].mean().reset_index()
#             daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
#             daily_sentiment = daily_sentiment.sort_values('date')

#             if len(daily_sentiment) >= 2:
#                 # Fit linear regression to determine trend
#                 X = np.arange(len(daily_sentiment)).reshape(-1, 1)
#                 y = daily_sentiment['compound'].values
#                 model_lr = LinearRegression().fit(X, y)
#                 slope = model_lr.coef_[0]
                
#                 # Determine trend direction
#                 if slope > 0.05:
#                     trend = "Strong Upward"
#                     color = "green"
#                 elif slope > 0:
#                     trend = "Mild Upward"
#                     color = "lightgreen"
#                 elif slope < -0.05:
#                     trend = "Strong Downward"
#                     color = "red"
#                 else:
#                     trend = "Mild Downward"
#                     color = "pink"
                
#                 st.markdown(f"**Sentiment Trend:** :{color}[{trend} Trend] (Slope: {slope:.2f})")
                
#                 # Plot sentiment trend
#                 fig_sent = px.line(daily_sentiment, x='date', y='compound', 
#                                  title='Daily Average Sentiment Trend',
#                                  markers=True)
#                 fig_sent.update_layout(yaxis_range=[-1,1])
#                 st.plotly_chart(fig_sent, use_container_width=True)
#             else:
#                 st.warning("Insufficient data to determine NLP-based trend. At least 2 days of news required.")
        # except Exception as e:
        #     st.error(f"Error in NLP trend analysis: {str(e)}")
#     else:
#         st.warning("No news articles available for NLP-based trend prediction")

#     # Time Series Forecasting with Prophet
#     st.subheader("Price Forecast using Prophet")
#     try:
#         # Prepare data for Prophet
#         prophet_df = data.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
#         prophet_df = prophet_df.dropna()
        
#         if len(prophet_df) < 2:
#             st.error("Not enough historical data for forecasting")
#         else:
#             with st.spinner("Training forecasting model..."):
#                 model = Prophet()
#                 model.fit(prophet_df)
#                 future = model.make_future_dataframe(periods=forecast_days)
#                 forecast = model.predict(future)

#             # Plot forecast
#             fig_forecast = go.Figure()
#             fig_forecast.add_trace(go.Scatter(
#                 x=forecast['ds'], 
#                 y=forecast['yhat'],
#                 name='Forecast',
#                 line=dict(color='blue', width=2)
#             ))
#             fig_forecast.add_trace(go.Scatter(
#                 x=prophet_df['ds'],
#                 y=prophet_df['y'],
#                 name='Historical',
#                 line=dict(color='orange', width=2)
#             ))
#             fig_forecast.add_trace(go.Scatter(
#                 x=forecast['ds'],
#                 y=forecast['yhat_upper'],
#                 fill=None,
#                 mode='lines',
#                 line=dict(color='rgba(0,100,80,0.2)'),
#                 name='Upper Bound'
#             ))
#             fig_forecast.add_trace(go.Scatter(
#                 x=forecast['ds'],
#                 y=forecast['yhat_lower'],
#                 fill='tonexty',
#                 mode='lines',
#                 line=dict(color='rgba(0,100,80,0.2)'),
#                 name='Lower Bound'
#             ))
#             fig_forecast.update_layout(
#                 title=f'{stock_symbol} Price Forecast ({forecast_days} days)',
#                 xaxis_title='Date',
#                 yaxis_title='Price',
#                 hovermode='x unified'
#             )
#             st.plotly_chart(fig_forecast, use_container_width=True)
            
#             # Show forecast components
#             with st.expander("Show Forecast Components"):
#                 fig_components = model.plot_components(forecast)
#                 st.pyplot(fig_components)
                
#     except Exception as e:
#         st.error(f"Error in forecasting: {str(e)}")

# # Add a refresh button
# st.sidebar.markdown("---")
# if st.sidebar.button("🔄 Refresh Data"):
#     st.rerun()
# st.sidebar.info("Note: Data updates may take a few seconds")







import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from gnews import GNews
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Try to import Prophet, but fall back to statsmodels if unavailable
try:
    from prophet import Prophet
    has_prophet = True
except ImportError:
    has_prophet = False
    st.warning("Prophet not installed. Using ARIMA model instead. For better forecasts, install Prophet: pip install prophet")

# Set up Streamlit page
st.set_page_config(page_title="Advanced Stock & News Dashboard", layout="wide")
st.title("📈 FinHub by XNL")

# User Inputs
col1, col2, col3 = st.columns(3)
with col1:
    stock_symbol = st.text_input("Enter Stock Symbol (e.g., TSLA, AAPL):", "TSLA").upper()
with col2:
    time_period = st.selectbox("Select Time Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y"])
with col3:
    forecast_days = st.number_input("Forecast Period (days)", min_value=1, max_value=30, value=7)

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to get enhanced stock data
def get_stock_data(symbol, period):
    stock = yf.Ticker(symbol)
    hist = stock.history(period=period)
    
    # Calculate technical indicators
    hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
    hist['RSI'] = 100 - (100 / (1 + hist['Close'].diff(1).clip(lower=0).rolling(14).mean() / 
                          hist['Close'].diff(1).clip(upper=0).abs().rolling(14).mean()))
    return hist

# Function to fetch recent news using GNews
def fetch_news(symbol):
    google_news = GNews()
    google_news.period = '7d'  # Last 7 days
    google_news.max_results = 20  # Limit results
    try:
        news = google_news.get_news(f'{symbol} stock')
        return [(article['title'], 
                article['published date'], 
                article['url']) for article in news][:10]  # Return top 10
    except:
        return []

# Enhanced sentiment analysis with VADER
def analyze_sentiment(headlines):
    results = []
    for headline, date, url in headlines:
        vs = analyzer.polarity_scores(headline)
        results.append({
            "headline": headline,
            "date": date,
            "url": url,
            "positive": vs["pos"],
            "neutral": vs["neu"],
            "negative": vs["neg"],
            "compound": vs["compound"]
        })
    return results

# Display fundamental data
def show_fundamentals(symbol):
    stock = yf.Ticker(symbol)
    info = stock.info
    col1, col2, col3, col4 = st.columns(4)
    fundamentals = {
        "Market Cap": f"${info.get('marketCap', 'N/A')/1e9:.1f}B" if info.get('marketCap') else 'N/A',
        "PE Ratio": info.get('trailingPE', 'N/A'),
        "52 Week High": info.get('fiftyTwoWeekHigh', 'N/A'),
        "Volume Avg.": f"{info.get('averageVolume', 'N/A'):,}"
    }
    for i, (k, v) in enumerate(fundamentals.items()):
        [col1, col2, col3, col4][i].metric(k, v)

# ARIMA Forecasting function (alternative to Prophet)
def forecast_with_arima(historical_data, forecast_days):
    # Prepare data for ARIMA
    data = historical_data['Close'].values
    
    # Fit ARIMA model
    model = ARIMA(data, order=(5,1,0))
    model_fit = model.fit()
    
    # Forecast
    forecast = model_fit.forecast(steps=forecast_days)
    forecast_index = pd.date_range(start=historical_data.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
    
    # Calculate confidence intervals
    # Assuming 95% confidence interval
    conf_int = model_fit.get_forecast(steps=forecast_days).conf_int()
    lower_bound = conf_int.iloc[:, 0].values
    upper_bound = conf_int.iloc[:, 1].values
    
    return forecast, forecast_index, lower_bound, upper_bound

# Main Dashboard
if st.button("Analyze Market Data"):
    st.header(f"📊 {stock_symbol} Comprehensive Analysis")
    
    # Fundamentals Section
    show_fundamentals(stock_symbol)
    
    # Stock Data Visualization
    st.subheader("Technical Analysis Chart")
    data = get_stock_data(stock_symbol, time_period)
    
    # Create subplots
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.05,
                       row_heights=[0.5, 0.25, 0.25])
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(x=data.index,
                                open=data['Open'],
                                high=data['High'],
                                low=data['Low'],
                                close=data['Close'],
                                name='Price'), row=1, col=1)
    
    # SMA
    fig.add_trace(go.Scatter(x=data.index, 
                            y=data['SMA_20'],
                            line=dict(color='orange', width=2),
                            name='20 SMA'), row=1, col=1)
    
    # Volume
    fig.add_trace(go.Bar(x=data.index, 
                        y=data['Volume'],
                        name='Volume'), row=2, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=data.index, 
                            y=data['RSI'],
                            line=dict(color='purple', width=2),
                            name='RSI'), row=3, col=1)
    
    fig.update_layout(height=800, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # News Analysis Section
    st.subheader("📰 Latest News Analysis")
    news_articles = fetch_news(stock_symbol)
    
    if news_articles:
        sentiment_data = analyze_sentiment(news_articles)
        df = pd.DataFrame(sentiment_data)
        
        # Sentiment Metrics
        avg_sentiment = df['compound'].mean()
        sentiment_category = "Positive" if avg_sentiment > 0.05 else "Negative" if avg_sentiment < -0.05 else "Neutral"
        
        col1, col2 = st.columns(2)
        col1.metric("Average Sentiment Score", f"{avg_sentiment:.2f}", sentiment_category)
        col2.metric("Total Articles Analyzed", len(df))
        
        # Sentiment Visualization
        fig = px.pie(df, names=['Positive', 'Neutral', 'Negative'], 
                    values=[df['positive'].mean(), df['neutral'].mean(), df['negative'].mean()],
                    title="Sentiment Distribution",
                    color_discrete_sequence=['green', 'gray', 'red'])
        st.plotly_chart(fig, use_container_width=True)
        
        # News List
        st.subheader("Recent News Headlines")
        for idx, article in enumerate(sentiment_data):
            with st.expander(f"{article['headline']} (Sentiment: {article['compound']:.2f})"):
                st.caption(f"Published: {article['date']}")
                st.markdown(f"[Read more]({article['url']})")
    else:
        st.warning("No recent news articles found for this stock")

    # Market Trend Prediction Section
    st.header("📈 Market Trend Prediction")
    
    # NLP-Based Trend Prediction
    st.subheader("NLP-Based Trend Analysis")
    if news_articles:
        try:
            df_sentiment = pd.DataFrame(sentiment_data)
            df_sentiment['date'] = pd.to_datetime(df_sentiment['date']).dt.date
            daily_sentiment = df_sentiment.groupby('date')['compound'].mean().reset_index()
            daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
            daily_sentiment = daily_sentiment.sort_values('date')

            if len(daily_sentiment) >= 2:
                # Fit linear regression to determine trend
                X = np.arange(len(daily_sentiment)).reshape(-1, 1)
                y = daily_sentiment['compound'].values
                model_lr = LinearRegression().fit(X, y)
                slope = model_lr.coef_[0]
                
                # Determine trend direction
                if slope > 0.05:
                    trend = "Strong Upward"
                    color = "green"
                elif slope > 0:
                    trend = "Mild Upward"
                    color = "lightgreen"
                elif slope < -0.05:
                    trend = "Strong Downward"
                    color = "red"
                else:
                    trend = "Mild Downward"
                    color = "pink"
                
                st.markdown(f"**Sentiment Trend:** :{color}[{trend} Trend] (Slope: {slope:.2f})")
                
                # Plot sentiment trend
                fig_sent = px.line(daily_sentiment, x='date', y='compound', 
                                 title='Daily Average Sentiment Trend',
                                 markers=True)
                fig_sent.update_layout(yaxis_range=[-1,1])
                st.plotly_chart(fig_sent, use_container_width=True)
            else:
                st.warning("Insufficient data to determine NLP-based trend. At least 2 days of news required.")
        except Exception as e:
            st.error(f"Error in NLP trend analysis: {str(e)}")
    else:
        st.warning("No news articles available for NLP-based trend prediction")

    # Time Series Forecasting (Prophet or ARIMA)
    st.subheader("Price Forecast")
    try:
        # Check if we have enough data
        if len(data) < 2:
            st.error("Not enough historical data for forecasting")
        else:
            # Choose forecasting method based on available libraries
            if has_prophet:
                with st.spinner("Training Prophet forecasting model..."):
                    # Prepare data for Prophet
                    # Make a copy of the data to avoid modifying the original
                    prophet_data = data.copy()
                    
                    # Reset index and ensure Date doesn't have timezone info
                    prophet_data = prophet_data.reset_index()
                    
                    # Remove timezone from Date column
                    if hasattr(prophet_data['Date'].iloc[0], 'tz'):
                        prophet_data['Date'] = prophet_data['Date'].dt.tz_localize(None)
                    
                    # Rename columns for Prophet
                    prophet_df = prophet_data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
                    prophet_df = prophet_df.dropna()
                    
                    model = Prophet(daily_seasonality=True)
                    model.fit(prophet_df)
                    future = model.make_future_dataframe(periods=forecast_days)
                    forecast = model.predict(future)
                    
                    # Plot forecast
                    fig_forecast = go.Figure()
                    fig_forecast.add_trace(go.Scatter(
                        x=forecast['ds'], 
                        y=forecast['yhat'],
                        name='Forecast',
                        line=dict(color='blue', width=2)
                    ))
                    fig_forecast.add_trace(go.Scatter(
                        x=prophet_df['ds'],
                        y=prophet_df['y'],
                        name='Historical',
                        line=dict(color='orange', width=2)
                    ))
                    fig_forecast.add_trace(go.Scatter(
                        x=forecast['ds'],
                        y=forecast['yhat_upper'],
                        fill=None,
                        mode='lines',
                        line=dict(color='rgba(0,100,80,0.2)'),
                        name='Upper Bound'
                    ))
                    fig_forecast.add_trace(go.Scatter(
                        x=forecast['ds'],
                        y=forecast['yhat_lower'],
                        fill='tonexty',
                        mode='lines',
                        line=dict(color='rgba(0,100,80,0.2)'),
                        name='Lower Bound'
                    ))
                    fig_forecast.update_layout(
                        title=f'{stock_symbol} Price Forecast ({forecast_days} days)',
                        xaxis_title='Date',
                        yaxis_title='Price',
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig_forecast, use_container_width=True)
                    
                    # Show forecast components
                    with st.expander("Show Forecast Components"):
                        fig_components = model.plot_components(forecast)
                        st.pyplot(fig_components)
            else:
                # Use ARIMA as fallback
                with st.spinner("Training ARIMA forecasting model..."):
                    forecast_values, forecast_dates, lower_bound, upper_bound = forecast_with_arima(data, forecast_days)
                    
                    # Plot forecast
                    fig_forecast = go.Figure()
                    fig_forecast.add_trace(go.Scatter(
                        x=forecast_dates, 
                        y=forecast_values,
                        name='ARIMA Forecast',
                        line=dict(color='blue', width=2)
                    ))
                    fig_forecast.add_trace(go.Scatter(
                        x=data.index,
                        y=data['Close'],
                        name='Historical',
                        line=dict(color='orange', width=2)
                    ))
                    fig_forecast.add_trace(go.Scatter(
                        x=forecast_dates,
                        y=upper_bound,
                        fill=None,
                        mode='lines',
                        line=dict(color='rgba(0,100,80,0.2)'),
                        name='Upper Bound'
                    ))
                    fig_forecast.add_trace(go.Scatter(
                        x=forecast_dates,
                        y=lower_bound,
                        fill='tonexty',
                        mode='lines',
                        line=dict(color='rgba(0,100,80,0.2)'),
                        name='Lower Bound'
                    ))
                    fig_forecast.update_layout(
                        title=f'{stock_symbol} Price Forecast ({forecast_days} days) - ARIMA Model',
                        xaxis_title='Date',
                        yaxis_title='Price',
                        hovermode='x unified'
                    )
                    st.plot

    except Exception as e:
            st.error(f"Error in NLP trend analysis: {str(e)}")