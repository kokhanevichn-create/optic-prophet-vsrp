import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ✅ Streamlit must be imported before any st.* usage
import streamlit as st

# The rest of your imports follow
import math
from math import log, sqrt, exp
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.stats import norm, zscore
# ---- Journal / Heatmap (encapsulated so it's never executed before imports) ----
TIMEZONE_ET = ZoneInfo("America/New_York")   # keep near your other CONFIG constants
TRADE_LOG_CSV = "vsrp_trade_log.csv"         # keep consistent with the rest of the app

def render_journal_and_heatmap():
    st.divider()
    st.header("🧠 Discipline: Log trades → learn your kill zones")

    with st.expander("Log a trade (R-based)"):
        # Inputs
        colA, colB, colC = st.columns([1,1,2])
        t_ticker = colA.text_input("Ticker", value="MSFT")
        t_R = colB.number_input("R result", value=0.0, step=0.25, format="%.2f")

        # Prefer datetime_input if available; else fall back to date+time
        now_utc_naive = datetime.utcnow()
        if hasattr(st, "datetime_input"):
            exec_dt = colC.datetime_input("Execution time (UTC)", value=now_utc_naive)
        else:
            d = colC.date_input("Execution date (UTC)", value=now_utc_naive.date())
            t = colC.time_input("Execution time (UTC)", value=now_utc_naive.time())
            exec_dt = datetime.combine(d, t)

        # Force UTC aware
        if exec_dt.tzinfo is None:
            exec_dt = exec_dt.replace(tzinfo=timezone.utc)
        else:
            exec_dt = exec_dt.astimezone(timezone.utc)

        if st.button("Add to log"):
            try:
                logdf = pd.read_csv(TRADE_LOG_CSV)
            except Exception:
                logdf = pd.DataFrame(columns=["timestamp","ticker","R"])

            row = {"timestamp": exec_dt.isoformat(), "ticker": t_ticker.upper(), "R": t_R}
            logdf = pd.concat([logdf, pd.DataFrame([row])], ignore_index=True)
            logdf.to_csv(TRADE_LOG_CSV, index=False)
            st.success("Logged.")
            st.download_button("Download log CSV",
                               logdf.to_csv(index=False).encode(),
                               file_name="vsrp_trade_log.csv",
                               mime="text/csv")

    with st.expander("Show time‑of‑day heatmap data"):
        try:
            log = pd.read_csv(TRADE_LOG_CSV)
            # Parse as UTC-aware and then convert to ET
            log["timestamp"] = pd.to_datetime(log["timestamp"], utc=True, errors="coerce")
            log = log.dropna(subset=["timestamp"]).copy()
            log["et"] = log["timestamp"].dt.tz_convert(TIMEZONE_ET)
            log["hour"] = log["et"].dt.hour
            log["dow"]  = log["et"].dt.weekday

            agg = (log.groupby(["dow","hour"])
                       .agg(win_rate=("R", lambda x: float(np.mean(x > 0)) if len(x) else np.nan),
                            avg_R=("R","mean"),
                            n=("R","count"))
                       .reset_index()
                       .sort_values(["dow","hour"]))

            st.dataframe(agg, use_container_width=True, height=260)
        except Exception as e:
            st.info("No trade log yet — once you log, this will map your **win‑rate by hour** so you can hard‑code deploy windows.")
# ... after your scan table and spread details ...
render_journal_and_heatmap()
