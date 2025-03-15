# FinHub By XNL

🚀 An advanced end to end Fintech tool

## 📌 Project Overview

This is a real-time financial analysis and sentiment tracking system that integrates various components to provide live stock data, market sentiment analysis, technical indicators, AI-powered forecasting, and an interactive financial dashboard.

## 🔹 Technologies Used:

✅ **FastAPI** for AI model routing and API backend  
✅ **Streamlit** for an interactive frontend  
✅ **Alpha Vantage, Yahoo Finance & Google News API** for data sourcing  
✅ **VaderSentiment** for sentiment analysis  
✅ **Prophet, ARIMA** for stock price forecasting  
✅ **Together AI & OpenAI (GPT-4o-mini, Llama-3.3-70B)** for AI-powered insights  

## 📂 Repository Structure

This repository consists of four main components:

### 1️⃣ XNL - FinTech API (Stock & Market Data Processing)
📍 This module fetches real-time stock data using **Alpha Vantage & Yahoo Finance API** and provides structured financial data in JSON format.

#### ➡️ Key Features:
✅ Fetches live stock market data  
✅ Retrieves historical data for trend analysis  
✅ Provides JSON-formatted API responses  
✅ Computes **technical indicators** (SMA, RSI, MACD)  

📌 Location: `FinTech API/`

---

### 2️⃣ XNL - Market Sentiment Analysis (News Sentiment Tracking)
📍 This module analyzes financial news sentiment and assigns a sentiment score using **VaderSentiment**.

#### ➡️ Key Features:
✅ Fetches latest financial news from **Google News API**  
✅ Applies **VaderSentiment Analysis** for polarity detection  
✅ Provides structured sentiment scores (**Positive, Negative, Neutral**)  

📌 Location: `Market Sentiment Analysis/`

---

### 3️⃣ XNL - AI-Based Forecasting (Stock Price Predictions)
📍 This module forecasts future stock prices using **ARIMA & Prophet models**.

#### ➡️ Key Features:
✅ Predicts future stock prices based on historical trends  
✅ Uses **ARIMA (AutoRegressive Integrated Moving Average)** for time-series forecasting  
✅ Implements **Prophet (Meta’s forecasting model)** for better accuracy  
✅ Provides AI-powered market trend insights  



---

### 4️⃣ XNL - Dashboard (Frontend)
📍 This folder contains both frontend and backend components, integrating **FastAPI, Streamlit, and Next.js**.

#### 📌 Backend:
✅ Developed using **FastAPI**  
✅ Routes AI queries via **OpenAI (GPT-4o-mini) & Together AI (Llama-3.3-70B)**  
✅ Hosts **forecasting models** (ARIMA, Prophet)  
✅ Fetches stock data & news sentiment  
✅ Hosted on **Railway.app**  

#### 📌 Frontend:
✅ Built using **Next.js & Tailwind CSS**  
✅ Provides an interactive financial dashboard  
✅ Uses **Streamlit Cloud** for AI model visualization  

📌 Location: `Frontend/`

---

## 📜 Project Documentation
A detailed document explaining the entire process and work done in this project is available inside the repository.

📌 Location: `Project Documentation.pdf`

---

## 🚀 How to Run the Project Locally

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Architgarg2003/XNL-21BCE11235-LLM-1.git
cd XNL-21BCE11235-LLM-1
```

### 2️⃣ Set Up the Backend (FastAPI & AI Models)
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate    # On Windows
pip install -r requirements.txt
```

### 4️⃣ Set Up & Run the Frontend (Streamlit)
```bash
cd frontend
streamlit run streamlit_app.py
```

---

## 📌 Live Deployment Links
✅ **Frontend (Next.js, Python - Streamlit)**: [https://xnl-21bce11235-llm.streamlit.app])  

---

## 📬 Contact & Queries
For any queries or issues, please open a GitHub issue or contact me via email.

---

## 🎉 Acknowledgment & Gratitude
I would like to express my sincere gratitude to **XNL Innovations** for assigning me this opportunity to work on this project. This assignment was part of the placement process, and I deeply appreciate the learning experience it provided.

Thank you, **XNL Innovations**, for your guidance and support! 🎉

