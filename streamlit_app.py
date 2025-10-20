# FULL STREAMLIT APP — Optic Prophet: Advanced VSRP Scanner (v2)

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd

st.set_page_config(page_title="Optic Prophet: VSRP Scanner", page_icon="🧠")
st.title("🧠 Optic Prophet: VSRP Scanner")
st.subheader("Volatility Skew • Signal Intelligence • Option Sentiment")
st.markdown("---")

ticker = st.text_input("Enter Ticker (e.g. AAPL, TSLA, NVDA)")

# ---------- Core Interpretation Functions ----------
def interpret_bias(ratio):
    if ratio is None:
        return ("⚠️ Skew Undetectable", "gray")
    if ratio < 0.7:
        return ("📉 Cheap Premium — Buy Zone", "green")
    elif ratio > 1.3:
        return ("🔥 High Skew — Sell Zone", "red")
    return ("🟡 Neutral Skew", "yellow")

def expected_move(price, iv, days):
    return round(price * iv * np.sqrt(days / 365), 2)

def calculate_iv_rank(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist_iv = []
        for date in stock.options[:5]:
            chain = stock.option_chain(date)
            ivs = chain.calls['impliedVolatility'].dropna()
            hist_iv.extend(ivs.tolist())
        if not hist_iv:
            return "N/A"
        current_iv = np.mean(hist_iv[-10:])
        rank = 100 * (np.sum(np.array(hist_iv) < current_iv) / len(hist_iv))
        return round(rank, 1)
    except:
        return "N/A"

def calculate_pcr(calls, puts):
    call_vol = calls['volume'].sum()
    put_vol = puts['volume'].sum()
    if call_vol + put_vol == 0:
        return "N/A"
    return round(put_vol / call_vol, 2) if call_vol else "∞"

def calculate_skew(calls, last):
    atm_iv = calls.iloc[(calls['strike'] - last).abs().argsort()[:1]]['impliedVolatility'].values[0]
    otm_calls = calls[calls['strike'] > last * 1.05].head(3)
    if otm_calls.empty:
        return "N/A"
    avg_otm_iv = otm_calls['impliedVolatility'].mean()
    return round((avg_otm_iv - atm_iv) * 100, 2)

def score_signal(iv_rank, skew, pcr):
    score = 50
    if isinstance(iv_rank, (int, float)) and iv_rank > 70:
        score += 15
    if isinstance(skew, (int, float)) and skew > 3:
        score += 15
    if isinstance(pcr, (int, float)) and pcr > 1.2:
        score += 15
    return min(score, 100)

# ---------- Core Data Pull + Output ----------
def get_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="6mo")
        if hist.empty:
            return {"error": "No historical data found."}

        returns = hist["Close"].pct_change().dropna()
        rv = np.std(returns) * np.sqrt(252)
        last_price = hist["Close"].iloc[-1]

        options_dates = stock.options
        if not options_dates:
            return {"error": "No options available."}

        opt_chain = stock.option_chain(options_dates[0])
        calls = opt_chain.calls
        puts = opt_chain.puts

        atm_call = calls.iloc[(calls['strike'] - last_price).abs().argsort()[:1]]
        if atm_call.empty:
            return {"error": "No ATM option found."}

        iv = atm_call['impliedVolatility'].values[0]
        atm_strike = atm_call['strike'].values[0]
        volume = atm_call['volume'].values[0]
        oi = atm_call['openInterest'].values[0]

        ratio = iv / rv if rv else None
        bias_msg, bias_color = interpret_bias(ratio)

        iv_rank = calculate_iv_rank(ticker)
        pcr = calculate_pcr(calls, puts)
        skew = calculate_skew(calls, last_price)
        score = score_signal(iv_rank, skew, pcr)

        return {
            "Price": round(last_price, 2),
            "ATM Strike": round(atm_strike, 2),
            "IV (ATM)": round(iv, 3),
            "RV (6mo)": round(rv, 3),
            "IV/RV Ratio": round(ratio, 2) if ratio else "N/A",
            "Expected Move (1D)": expected_move(last_price, iv, 1),
            "Expected Move (1W)": expected_move(last_price, iv, 5),
            "Expected Move (1M)": expected_move(last_price, iv, 21),
            "Volume (ATM)": volume,
            "OI (ATM)": oi,
            "IV Rank": iv_rank,
            "Put/Call Ratio": pcr,
            "Skew % (OTM vs ATM)": skew,
            "Signal Score": score,
            "Bias": bias_msg,
            "Color": bias_color
        }

    except Exception as e:
        return {"error": str(e)}

# ---------- Display UI ----------
if ticker:
    result = get_data(ticker.upper())
    if "error" in result:
        st.error(result["error"])
    else:
        st.markdown("### 📊 Volatility Snapshot")
        st.write(f"Price: ${result['Price']} | ATM Strike: {result['ATM Strike']}")
        st.write(f"IV: {result['IV (ATM)']} | RV: {result['RV (6mo)']} | IV/RV: {result['IV/RV Ratio']}")
        st.markdown("---")

        st.markdown("### 🔭 Expected Moves")
        st.write(f"1D: ±${result['Expected Move (1D)']} | 1W: ±${result['Expected Move (1W)']} | 1M: ±${result['Expected Move (1M)']}")
        st.markdown("---")

        st.markdown("### 🧠 Bias Engine")
        st.markdown(f":{result['Color']}_circle: {result['Bias']}")
        st.markdown("---")

        st.markdown("### 📈 Sentiment Data")
        st.write(f"Volume (ATM): {result['Volume (ATM)']} | Open Interest (ATM): {result['OI (ATM)']}")
        st.write(f"Put/Call Volume Ratio: {result['Put/Call Ratio']}")
        st.write(f"Skew % (OTM > ATM): {result['Skew % (OTM vs ATM)']}%")
        st.markdown("---")

        st.markdown("### 🧪 Signal Intelligence")
        st.write(f"IV Rank: {result['IV Rank']} | Conviction Score: {result['Signal Score']}/100")
