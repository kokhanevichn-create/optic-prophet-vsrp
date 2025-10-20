# streamlit_app_vsrp.py
# Optic Prophet — VSRP Scanner (Complete)
# Built tight: IV/RV, RR25 skew, z-scores, momentum bias, IV Rank, regime overlay, dead-time gate,
# delta-based spread builder, trim/trigger guidance, journal+heatmap (side panel).

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ---- Streamlit must be imported before any st.* calls ----
import streamlit as st

import math
from math import log, sqrt, exp
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from typing import Optional, Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

# ============================ CONFIG ============================ #
DEFAULT_TICKERS = ["MSFT","META","GOOG","AMZN","TSLA","AAPL"]

TARGET_DTE = 30
MIN_DTE, MAX_DTE = 18, 45
OI_MIN = 50
VOL_MIN = 1
RISK_FREE_RATE = 0.045
DIV_YIELD = 0.0

IVRV_THRESHOLD = 1.35
SKEW_ZSCORE_THRESHOLD = 1.8
IVR_LOW_GUARD = 20.0
VEGA_FLIP_DELTA = 15.0

SPREAD_WIDTH_LIMIT = 0.10            # % of spot
DEBIT_CALL_DELTAS = (0.35, 0.20)     # (long, short)
DEBIT_PUT_DELTAS  = (-0.35,-0.20)
CREDIT_PUT_DELTAS = (-0.25,-0.10)    # (short, long)
CREDIT_CALL_DELTAS= (0.25, 0.10)

MOMENTUM_WINDOW = 90
TIMEZONE_ET = ZoneInfo("America/New_York")
DEPLOY_WINDOWS = {1:[("10:00","11:15")], 2:[("10:00","11:15")], 3:[("10:00","11:15")]}  # Tue-Thu

METRICS_CSV = "vsrp_metrics.csv"      # persistent IV history for IV Rank
TRADE_LOG_CSV = "vsrp_trade_log.csv"
SKEW_ZONES_CSV = "vsrp_skew_zones.csv"

MAX_WORKERS = 8

# ============================ PAGE SETUP ============================ #
st.set_page_config(page_title="Optic Prophet — VSRP Scanner", page_icon="🧠", layout="wide")
st.title("🧠 Optic Prophet: VSRP Scanner")
st.caption("We don’t predict price. We hunt **distortion** (IV vs RV) and **skew**. Every ARM is sniper-grade or it’s a KILL.")

# ============================ CACHED IO ============================ #
@st.cache_data(show_spinner=False, ttl=300)
def _yf_history(tkr: str, period="1y", interval="1d"):
    return yf.download(tkr, period=period, interval=interval, progress=False)

@st.cache_data(show_spinner=False, ttl=600)
def _list_option_expiries(tkr: str) -> List[str]:
    try:
        return yf.Ticker(tkr).options or []
    except Exception:
        return []

@st.cache_data(show_spinner=False, ttl=300)
def _fetch_chain(tkr: str, expiry: str):
    oc = yf.Ticker(tkr).option_chain(expiry)
    return oc.calls.copy(), oc.puts.copy()

# ============================ MATH HELPERS ============================ #
def _annualized_rv_from_prices(prices: pd.Series, window: int = 20) -> float:
    if prices is None or len(prices) < window + 2:
        return np.nan
    rets = np.log(prices).diff().dropna()
    return rets.tail(window).std() * sqrt(252)

def _choose_expiry(tkr: str, target: int = TARGET_DTE) -> Optional[str]:
    opts = _list_option_expiries(tkr)
    if not opts:
        return None
    today = datetime.now(timezone.utc).date()
    picks = []
    for d in opts:
        dt = datetime.strptime(d, "%Y-%m-%d").date()
        dte = (dt - today).days
        if MIN_DTE <= dte <= 365:
            picks.append((abs(dte - target), dte, d))
    if not picks:
        return None
    picks.sort(key=lambda x: x[0])
    return picks[0][2]

def _bs_delta(S, K, T, r, sigma, q, is_call):
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return np.nan
    d1 = (log(S/K) + (r - q + 0.5*sigma*sigma)*T)/(sigma*sqrt(T))
    return exp(-q*T)*norm.cdf(d1) if is_call else -exp(-q*T)*norm.cdf(-d1)

def _mid(row):
    bid = float(row.get("bid", np.nan))
    ask = float(row.get("ask", np.nan))
    last = float(row.get("lastPrice", np.nan))
    if np.isfinite(bid) and np.isfinite(ask) and ask >= bid and ask > 0:
        return (bid+ask)/2.0
    return last if np.isfinite(last) else np.nan

def _attach_delta(df: pd.DataFrame, S: float, T: float, is_call: bool) -> pd.DataFrame:
    if df is None or df.empty or "impliedVolatility" not in df.columns:
        return pd.DataFrame()
    w = df.copy()
    w = w[(w.get("openInterest", 0) >= OI_MIN) & (w.get("volume", 0) >= VOL_MIN)]
    if w.empty:
        return pd.DataFrame()
    sig = w["impliedVolatility"].astype(float).clip(lower=1e-6, upper=5.0)
    K = w["strike"].astype(float)
    w["delta"] = [_bs_delta(S, k, T, RISK_FREE_RATE, s, DIV_YIELD, is_call) for k, s in zip(K, sig)]
    w["mid"] = w.apply(_mid, axis=1)
    w = w.dropna(subset=["delta","impliedVolatility","mid"])
    return w

def _nearest_by_delta(df: pd.DataFrame, target_delta: float) -> Optional[pd.Series]:
    if df is None or df.empty: return None
    i = (df["delta"] - target_delta).abs().idxmin()
    return df.loc[i].copy()

def _atm_iv(calls: pd.DataFrame, puts: pd.DataFrame, S: float) -> float:
    ivs=[]
    for df in (calls, puts):
        if df is None or df.empty: continue
        idx = (df["strike"] - S).abs().idxmin()
        iv = float(df.loc[idx,"impliedVolatility"])
        if np.isfinite(iv): ivs.append(iv)
    return float(np.nanmean(ivs)) if ivs else np.nan

def _risk_reversal_25(calls: pd.DataFrame, puts: pd.DataFrame) -> float:
    c = _nearest_by_delta(calls, 0.25)
    p = _nearest_by_delta(puts, -0.25)
    if c is None or p is None: return np.nan
    return float(c["impliedVolatility"] - p["impliedVolatility"])

def _zscore_nan(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    mu = np.nanmean(a)
    sd = np.nanstd(a)
    if not np.isfinite(sd) or sd == 0:
        return np.full_like(a, np.nan)
    return (a - mu) / sd

def _momentum_bias(tkr: str) -> Optional[str]:
    df = _yf_history(tkr, period="6mo", interval="1d")
    if df is None or df.empty: return None
    df["SMA20"]=df["Close"].rolling(20).mean()
    df["SMA50"]=df["Close"].rolling(50).mean()
    r = df.iloc[-1]
    if r["Close"]>r["SMA20"]>r["SMA50"]: return "bull"
    if r["Close"]<r["SMA20"]<r["SMA50"]: return "bear"
    return "neutral"

def _get_spot(tkr: str) -> float:
    t = yf.Ticker(tkr)
    try:
        return float(t.fast_info["last_price"])
    except Exception:
        hist = t.history(period="1d")
        return float(hist["Close"][-1])

def _get_next_earnings_date(tkr: str):
    try:
        edf = yf.Ticker(tkr).get_earnings_dates(limit=1)
        if edf is None or edf.empty: return None
        return pd.to_datetime(edf.index[0]).date()
    except Exception:
        return None

def _today_et():
    return datetime.now(TIMEZONE_ET)

def _in_deploy_window() -> bool:
    now = _today_et()
    wd = now.weekday()  # Mon=0
    if wd not in DEPLOY_WINDOWS: return False
    t = now.strftime("%H:%M")
    for start,end in DEPLOY_WINDOWS[wd]:
        if start<=t<=end: return True
    return False

# ============================ IV RANK (Persistence) ============================ #
@st.cache_data(show_spinner=False)
def _load_metrics() -> pd.DataFrame:
    try:
        return pd.read_csv(METRICS_CSV, parse_dates=["date"])
    except Exception:
        return pd.DataFrame(columns=["date","ticker","atm_iv","ivrv","rr25"])

def _append_metric_row(row: dict):
    df = _load_metrics().copy()
    mask = (df["date"]==row["date"]) & (df["ticker"]==row["ticker"])
    if mask.any():
        for k,v in row.items(): df.loc[mask,k]=v
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(METRICS_CSV, index=False)
    _load_metrics.clear()

def _iv_rank_from_history(ticker: str, today_iv: float, date: pd.Timestamp, lookback_days=252) -> Tuple[float,float]:
    df = _load_metrics()
    tdf = df[df["ticker"]==ticker].sort_values("date")
    if tdf.empty or tdf["atm_iv"].dropna().shape[0] < 10:
        return np.nan, np.nan
    start_cut = date - pd.Timedelta(days=lookback_days*1.2)
    tdf = tdf[tdf["date"]>=start_cut]
    series = tdf["atm_iv"].astype(float).dropna()
    iv_min, iv_max = float(series.min()), float(series.max())
    if not np.isfinite(iv_min) or not np.isfinite(iv_max) or iv_max<=iv_min:
        return np.nan, np.nan
    rank_today = (today_iv - iv_min)/(iv_max - iv_min)*100.0
    prev_iv = float(series.iloc[-1])
    prev_rank = (prev_iv - iv_min)/(iv_max - iv_min)*100.0
    return float(rank_today), float(rank_today - prev_rank)

# ============================ MNQ/NQ REGIME ============================ #
@st.cache_data(show_spinner=False, ttl=180)
def _nq_regime():
    try:
        df = _yf_history("NQ=F", period="5d", interval="5m")
        label = "NQ=F (5m)"
    except Exception:
        df = _yf_history("^NDX", period="5d", interval="5m")
        label = "^NDX (5m)"
    if df is None or df.empty:
        return {"status":"unknown","label":label}
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    rets = (df["Close"].pct_change().dropna())
    intraday_rv = float(rets.std()*math.sqrt(78))  # ~78 5m bars/day
    trend = "risk_on" if df["EMA20"].iloc[-1] > df["EMA50"].iloc[-1] else "risk_off"
    hot = intraday_rv > 0.015
    regime = "OK" if (trend=="risk_on" and not hot) else ("Caution" if not hot else "Volatile")
    return {"status":regime,"trend":trend,"rv_5m":intraday_rv,"label":label}

# ============================ DATACLASSES ============================ #
@dataclass
class Signal:
    ticker: str
    price: float
    expiry: str
    dte: int
    atm_iv: float
    rv20: float
    ivrv_ratio: float
    rr25: float
    momentum: Optional[str]
    iv_rank: float = np.nan
    iv_rank_delta: float = np.nan
    skew_z: float = np.nan
    earnings_days: Optional[int] = None
    vega_flip: bool = False
    time_gate: bool = False
    play: str = "KILL"

@dataclass
class Spread:
    kind: str
    expiry: str
    width: float
    net: float
    max_gain: float
    max_loss: float
    breakeven: Optional[float]
    long_leg: Optional[dict]
    short_leg: Optional[dict]

# ============================ SPREAD BUILDER ============================ #
def _cap_width(width: float, spot: float, cap_pct: float) -> float:
    return min(width, spot*cap_pct)

def _select_vertical(calls, puts, S, expiry, kind, cap_pct):
    if kind=="Bull Call Debit":
        long = _nearest_by_delta(calls, DEBIT_CALL_DELTAS[0]); short=_nearest_by_delta(calls, DEBIT_CALL_DELTAS[1])
        if long is None or short is None: return None
        if float(short["strike"]) <= float(long["strike"]):
            cands = calls[calls["strike"]>long["strike"]]
            if cands.empty: return None
            short = cands.iloc[(cands["delta"]-DEBIT_CALL_DELTAS[1]).abs().idxmin()]
        width = _cap_width(float(short["strike"]-long["strike"]), S, cap_pct)
        net = float(long["mid"]-short["mid"]); max_gain = width - net; max_loss = net; be = float(long["strike"])+net
        return Spread(kind, expiry, width, net, max_gain, max_loss, be,
                      {"strike":float(long["strike"]), "delta":float(long["delta"]), "mid":float(long["mid"])},
                      {"strike":float(short["strike"]), "delta":float(short["delta"]), "mid":float(short["mid"])})
    if kind=="Bear Put Debit":
        long = _nearest_by_delta(puts, DEBIT_PUT_DELTAS[0]); short=_nearest_by_delta(puts, DEBIT_PUT_DELTAS[1])
        if long is None or short is None: return None
        if float(short["strike"]) >= float(long["strike"]):
            cands = puts[puts["strike"]<long["strike"]]
            if cands.empty: return None
            short = cands.iloc[(cands["delta"]-DEBIT_PUT_DELTAS[1]).abs().idxmin()]
        width = _cap_width(float(long["strike"]-short["strike"]), S, cap_pct)
        net = float(long["mid"]-short["mid"]); max_gain = width - net; max_loss = net; be = float(long["strike"])-net
        return Spread(kind, expiry, width, net, max_gain, max_loss, be,
                      {"strike":float(long["strike"]), "delta":float(long["delta"]), "mid":float(long["mid"])},
                      {"strike":float(short["strike"]), "delta":float(short["delta"]), "mid":float(short["mid"])})
    if kind=="Bull Put Credit":
        short=_nearest_by_delta(puts, CREDIT_PUT_DELTAS[0]); long=_nearest_by_delta(puts, CREDIT_PUT_DELTAS[1])
        if long is None or short is None: return None
        if float(long["strike"]) >= float(short["strike"]):
            cands = puts[puts["strike"]<short["strike"]]
            if cands.empty: return None
            long = cands.iloc[(cands["delta"]-CREDIT_PUT_DELTAS[1]).abs().idxmin()]
        width = _cap_width(float(short["strike"]-long["strike"]), S, cap_pct)
        net = float(short["mid"]-long["mid"]); max_gain = net; max_loss = width - net; be = float(short["strike"])-net
        return Spread(kind, expiry, width, net, max_gain, max_loss, be,
                      {"strike":float(long["strike"]), "delta":float(long["delta"]), "mid":float(long["mid"])},
                      {"strike":float(short["strike"]), "delta":float(short["delta"]), "mid":float(short["mid"])})
    if kind=="Bear Call Credit":
        short=_nearest_by_delta(calls, CREDIT_CALL_DELTAS[0]); long=_nearest_by_delta(calls, CREDIT_CALL_DELTAS[1])
        if long is None or short is None: return None
        if float(long["strike"]) <= float(short["strike"]):
            cands = calls[calls["strike"]>short["strike"]]
            if cands.empty: return None
            long = cands.iloc[(cands["delta"]-CREDIT_CALL_DELTAS[1]).abs().idxmin()]
        width = _cap_width(float(long["strike"]-short["strike"]), S, cap_pct)
        net = float(short["mid"]-long["mid"]); max_gain = net; max_loss = width - net; be = float(short["strike"])+net
        return Spread(kind, expiry, width, net, max_gain, max_loss, be,
                      {"strike":float(long["strike"]), "delta":float(long["delta"]), "mid":float(long["mid"])},
                      {"strike":float(short["strike"]), "delta":float(short["delta"]), "mid":float(short["mid"])})
    return None

# ============================ CORE COLLECTION ============================ #
def _collect_snapshot(tkr: str, target_dte: int) -> Optional[Tuple[Signal, pd.DataFrame, pd.DataFrame, float, float]]:
    try:
        spot = _get_spot(tkr)
        expiry = _choose_expiry(tkr, target_dte)
        if expiry is None:
            return None
        today = datetime.now(timezone.utc).date()
        dte = (datetime.strptime(expiry, "%Y-%m-%d").date() - today).days
        T = max(dte,1)/365.0

        calls_raw, puts_raw = _fetch_chain(tkr, expiry)
        calls = _attach_delta(calls_raw, spot, T, True)
        puts  = _attach_delta(puts_raw,  spot, T, False)

        atm_iv = _atm_iv(calls, puts, spot)
        rr25   = _risk_reversal_25(calls, puts)

        px = _yf_history(tkr, period="1y", interval="1d")["Close"]
        rv20 = _annualized_rv_from_prices(px, 20)
        ivrv = atm_iv/rv20 if np.isfinite(atm_iv) and np.isfinite(rv20) and rv20>0 else np.nan

        mom = _momentum_bias(tkr)
        edate = _get_next_earnings_date(tkr)
        e_days = (edate - today).days if edate else None

        sig = Signal(ticker=tkr, price=spot, expiry=expiry, dte=dte,
                     atm_iv=atm_iv, rv20=rv20, ivrv_ratio=ivrv, rr25=rr25,
                     momentum=mom, earnings_days=e_days)
        return sig, calls, puts, spot, T
    except Exception:
        return None

def _decide_play(sig: Signal, ivrv_gate: float, skew_gate: float, ivr_credit_guard: float) -> str:
    if not sig.time_gate:
        return "KILL (dead time)"
    if not np.isfinite(sig.ivrv_ratio) or sig.ivrv_ratio < ivrv_gate:
        return "KILL (IV not rich)"
    if not np.isfinite(sig.skew_z) or abs(sig.skew_z) < skew_gate:
        return "KILL (skew not extreme)"
    if sig.momentum=="bull" and sig.rr25>0:
        return "Bull Put Credit" if np.isfinite(sig.iv_rank) and sig.iv_rank < ivr_credit_guard else "Bull Call Debit"
    if sig.momentum=="bear" and sig.rr25<0:
        return "Bear Call Credit" if np.isfinite(sig.iv_rank) and sig.iv_rank < ivr_credit_guard else "Bear Put Debit"
    return "KILL (bias disagree)"

def _scan_watchlist(tickers: List[str], target_dte: int, ivrv_gate: float, skew_gate: float, ivr_credit_guard: float, width_cap: float):
    pods=[]
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(tickers))) as ex:
        futs = {ex.submit(_collect_snapshot, tkr, target_dte): tkr for tkr in tickers}
        for f in as_completed(futs):
            res = f.result()
            if res is not None:
                pods.append(res)
    if not pods:
        return pd.DataFrame()

    sigs=[p[0] for p in pods]
    rr_vals=np.array([s.rr25 for s in sigs], dtype=float)
    rr_z=_zscore_nan(rr_vals)
    now_utc=pd.Timestamp.utcnow().normalize()

    rows=[]
    for (sig,calls,puts,spot,T), z in zip(pods, rr_z):
        sig.skew_z=float(z) if np.isfinite(z) else np.nan
        sig.time_gate = _in_deploy_window()
        ivr, ivr_delta = _iv_rank_from_history(sig.ticker, sig.atm_iv, now_utc)
        sig.iv_rank, sig.iv_rank_delta = ivr, ivr_delta
        sig.vega_flip = bool(np.isfinite(ivr_delta) and ivr_delta >= VEGA_FLIP_DELTA)
        sig.play = _decide_play(sig, ivrv_gate, skew_gate, ivr_credit_guard)

        _append_metric_row({"date":now_utc, "ticker":sig.ticker, "atm_iv":sig.atm_iv, "ivrv":sig.ivrv_ratio, "rr25":sig.rr25})

        spread=None
        if "Debit" in sig.play or "Credit" in sig.play:
            spread=_select_vertical(calls, puts, spot, sig.expiry, sig.play, width_cap)

        earnings_trap = (sig.earnings_days is not None and sig.earnings_days <= 10 and np.isfinite(sig.atm_iv) and sig.atm_iv >= 0.80)

        # CALL/PUT + LONG/SHORT flags for quick read
        call_put = "CALL" if sig.rr25>0 else ("PUT" if sig.rr25<0 else "—")
        long_short = "LONG" if sig.momentum=="bull" else ("SHORT" if sig.momentum=="bear" else "NEUTRAL")

        # Trim/Trigger guidance (simple, rules-based)
        trim = "Trim 50% on Vega‑Flip (IVR Δ≥15). Tighten if skew_z falls <1.0."
        if earnings_trap:
            trim = "Avoid new debit risk into earnings (IV≥0.80, ≤10d). Prefer credits or wait."

        rows.append({
            "ticker":sig.ticker, "price":round(sig.price,2),"expiry":sig.expiry,"dte":sig.dte,
            "ivrv_ratio":round(sig.ivrv_ratio,2) if np.isfinite(sig.ivrv_ratio) else np.nan,
            "rr25":round(sig.rr25,4) if np.isfinite(sig.rr25) else np.nan,
            "skew_z":round(sig.skew_z,2) if np.isfinite(sig.skew_z) else np.nan,
            "iv_rank":round(sig.iv_rank,1) if np.isfinite(sig.iv_rank) else np.nan,
            "ivr_delta":round(sig.iv_rank_delta,1) if np.isfinite(sig.iv_rank_delta) else np.nan,
            "vega_flip":sig.vega_flip, "momentum":sig.momentum, "time_gate":sig.time_gate,
            "earnings_trap":earnings_trap, "play":sig.play, "call_put":call_put, "dir":long_short,
            "trim_note":trim, "spread": spread.__dict__ if spread else None
        })
    df=pd.DataFrame(rows).sort_values(by=["ivrv_ratio"], ascending=False, na_position="last").reset_index(drop=True)
    return df

# ============================ SIDEBAR CONTROLS ============================ #
with st.sidebar:
    st.header("⚙️ Controls")
    tickers_input = st.text_area("Tickers (comma‑separated)", value=",".join(DEFAULT_TICKERS)).upper()
    tickers = sorted({t.strip() for t in tickers_input.split(",") if t.strip()})
    target_dte = st.slider("Target DTE", 15, 60, TARGET_DTE)
    ivrv_gate = st.slider("IV/RV threshold", 1.10, 2.50, IVRV_THRESHOLD, 0.05)
    skew_gate = st.slider("Skew Z gate (|z|)", 0.5, 3.5, SKEW_ZSCORE_THRESHOLD, 0.1)
    ivr_credit_guard = st.slider("Debit→Credit Guard (IV Rank < X)", 0.0, 50.0, IVR_LOW_GUARD, 1.0)
    width_cap = st.slider("Max spread width (% of spot)", 0.02, 0.30, SPREAD_WIDTH_LIMIT, 0.01)
    use_nq = st.checkbox("Use NQ/MNQ regime overlay", value=True)
    run_btn = st.button("🔍 Scan Watchlist", use_container_width=True)

# ============================ REGIME STRIP ============================ #
regime_box = st.empty()
if use_nq:
    reg = _nq_regime()
    emoji = "🟢" if reg["status"]=="OK" else ("🟡" if reg["status"]=="Caution" else "🟠")
    regime_box.info(f"{emoji} **Regime**: {reg['status']}  —  Trend: {reg.get('trend','?')}  |  Intraday RV: {reg.get('rv_5m',float('nan')):.3f}  ({reg['label']})")

# ============================ SCAN + DISPLAY ============================ #
if run_btn:
    with st.spinner("Scanning for IV/RV distortion & skew…"):
        df = _scan_watchlist(tickers, target_dte, ivrv_gate, skew_gate, ivr_credit_guard, width_cap)

    if df.empty:
        st.warning("No data. Bad list, network issue, or market conditions outside scan bounds.")
    else:
        st.subheader("📡 Signal Output")
        show_cols = ["ticker","price","expiry","dte","ivrv_ratio","rr25","skew_z","iv_rank","ivr_delta","vega_flip","call_put","dir","time_gate","earnings_trap","play"]
        st.dataframe(df[show_cols], use_container_width=True, height=360)

        st.subheader("🧰 Proposed Spreads")
        for _, row in df.iterrows():
            if isinstance(row["spread"], dict):
                s=row["spread"]; k=row["play"]
                with st.expander(f"{row['ticker']} → {k}  |  {row['expiry']}"):
                    st.write(f"**Width:** {s['width']:.2f}   |   **Net** {'debit' if 'Debit' in k else 'credit'}: {s['net']:.2f}")
                    st.write(f"**Max Gain:** {s['max_gain']:.2f}   |   **Max Loss:** {s['max_loss']:.2f}   |   **Breakeven:** {s['breakeven']:.2f}")
                    st.write(f"**Bias:** {row['call_put']} / {row['dir']}")
                    st.write(f"**Trim/Trigger:** {row['trim_note']}")
                    ll=s['long_leg']; sl=s['short_leg']
                    if ll: st.write(f"LONG  strike **{ll['strike']:.2f}**  Δ {ll['delta']:.2f}  mid {ll['mid']:.2f}")
                    if sl: st.write(f"SHORT strike **{sl['strike']:.2f}**  Δ {sl['delta']:.2f}  mid {sl['mid']:.2f}")

        st.download_button("⬇️ Download scan CSV", df.to_csv(index=False).encode(),
                           file_name="vsrp_scan.csv", mime="text/csv", use_container_width=True)

# ============================ JOURNAL + HEATMAP (SIDE PANEL) ============================ #
with st.expander("🗒️ Log trades → build time‑of‑day edge", expanded=False):
    colA, colB, colC = st.columns([1,1,2])
    t_ticker = colA.text_input("Ticker", value="MSFT", key="log_ticker")
    t_R = colB.number_input("R result", value=0.0, step=0.25, format="%.2f", key="log_R")

    now_utc_naive = datetime.utcnow()
    if hasattr(st, "datetime_input"):
        exec_dt = colC.datetime_input("Execution time (UTC)", value=now_utc_naive, key="log_dt")
    else:
        d = colC.date_input("Execution date (UTC)", value=now_utc_naive.date(), key="log_date")
        t = colC.time_input("Execution time (UTC)", value=now_utc_naive.time(), key="log_time")
        exec_dt = datetime.combine(d, t)

    if exec_dt.tzinfo is None:
        exec_dt = exec_dt.replace(tzinfo=timezone.utc)
    else:
        exec_dt = exec_dt.astimezone(timezone.utc)

    if st.button("Add to journal", key="add_log"):
        try:
            logdf = pd.read_csv(TRADE_LOG_CSV)
        except Exception:
            logdf = pd.DataFrame(columns=["timestamp","ticker","R"])
        logdf = pd.concat([logdf, pd.DataFrame([{"timestamp": exec_dt.isoformat(), "ticker": t_ticker.upper(), "R": t_R}])], ignore_index=True)
        logdf.to_csv(TRADE_LOG_CSV, index=False)
        st.success("Logged.")

    # Heatmap data table
    try:
        log = pd.read_csv(TRADE_LOG_CSV)
        log["timestamp"] = pd.to_datetime(log["timestamp"], utc=True, errors="coerce")
        log = log.dropna(subset=["timestamp"]).copy()
        log["et"] = log["timestamp"].dt.tz_convert(TIMEZONE_ET)
        log["hour"] = log["et"].dt.hour
        log["dow"]  = log["et"].dt.weekday
        agg = (log.groupby(["dow","hour"])
                  .agg(win_rate=("R", lambda x: float(np.mean(x>0)) if len(x) else np.nan),
                       avg_R=("R","mean"),
                       n=("R","count"))
                  .reset_index()
                  .sort_values(["dow","hour"]))
        st.dataframe(agg, use_container_width=True, height=240)
        st.download_button("⬇️ Download journal CSV", log.to_csv(index=False).encode(),
                           file_name="vsrp_trade_log.csv", mime="text/csv")
    except Exception:
        st.info("No log yet. Add a trade to populate.")
        st.markdown("---")
st.subheader("📈 IV/RV Scanner")

ticker = st.text_input("Enter Ticker (e.g. AAPL, TSLA, MSFT)")

if ticker:
    result = get_iv_rv_ratio(ticker.upper())
    if isinstance(result, dict):
        st.markdown("### 🔍 Volatility Snapshot")
        for k, v in result.items():
            st.write(f"**{k}:** {v}")
    else:
        st.error(result)
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
# ---------------------------------------------------------
# 📈 IV/RV Scanner Core — Add this at the bottom of the file
# ---------------------------------------------------------

import yfinance as yf
import numpy as np

st.markdown("---")
st.subheader("📈 IV/RV Scanner")

ticker = st.text_input("Enter Ticker (e.g. AAPL, TSLA, MSFT)")

def get_iv_rv_ratio(ticker):
    try:
        data = yf.Ticker(ticker)
        hist = data.history(period="6mo")
        hist_returns = hist["Close"].pct_change().dropna()
        rv = np.std(hist_returns) * np.sqrt(252)

        options_dates = data.options
        if not options_dates:
            return "No options data available."

        front_month = options_dates[0]
        opt_chain = data.option_chain(front_month)
        last_price = hist["Close"].iloc[-1]
        calls = opt_chain.calls
        atm_call = calls.iloc[(calls['strike'] - last_price).abs().argsort()[:1]]
        iv = atm_call['impliedVolatility'].values[0]
        ratio = iv / rv if rv > 0 else None

        return {
            "Current Price": round(last_price, 2),
            "IV (ATM)": round(iv, 3),
            "RV (6mo)": round(rv, 3),
            "IV/RV Ratio": round(ratio, 2) if ratio else "N/A"
        }

    except Exception as e:
        return f"Error: {e}"

if ticker:
    result = get_iv_rv_ratio(ticker.upper())
    if isinstance(result, dict):
        st.markdown("### 🔍 Volatility Snapshot")
        for k, v in result.items():
            st.write(f"**{k}:** {v}")
    else:
        st.error(result)
