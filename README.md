# optic-prophet-vsrp
AI-powered volatility skew scanner + options spread deployment tool.
# optic_prophet_vsrp.py — VSRP Hooks Edition
# Nick K (Optic Prophet) + ruthless refactor
# Requires: yfinance, numpy, pandas, scipy (stats), python>=3.9 (zoneinfo)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from math import log, sqrt, exp
from typing import Optional, Tuple

from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm, zscore

# ---------------------------- CONFIG ---------------------------- #
LIQUID_TICKERS = ["MSFT", "META", "GOOG", "AMZN", "TSLA", "AAPL"]

TARGET_DTE = 30
MIN_DTE, MAX_DTE = 18, 45            # DTE bounds
OI_MIN = 50                           # quote quality guards
VOL_MIN = 1
RISK_FREE_RATE = 0.045
DIV_YIELD = 0.0

IVRV_THRESHOLD = 1.35                 # rich vol gate
SKEW_ZSCORE_THRESHOLD = 1.8           # extreme skew gate (cross-sectional)
IVR_LOW_GUARD = 20.0                  # <=> no debit spreads
VEGA_FLIP_DELTA = 15.0                # IV Rank jump trigger (pts)

SPREAD_WIDTH_LIMIT = 0.10             # max width as % of spot
DEBIT_CALL_DELTAS = (0.35, 0.20)      # long, short
DEBIT_PUT_DELTAS  = (-0.35, -0.20)
CREDIT_PUT_DELTAS = (-0.25, -0.10)    # short, long
CREDIT_CALL_DELTAS= (0.25, 0.10)

MOMENTUM_WINDOW = 90
MAX_WORKERS = min(8, len(LIQUID_TICKERS))

# Time gating (ET). Deploy Tues–Thu 10:00–11:15. Tweak as you learn.
TIMEZONE_ET = ZoneInfo("America/New_York")
DEPLOY_WINDOWS = {
    1: [("10:00","11:15")],   # Tuesday
    2: [("10:00","11:15")],   # Wednesday
    3: [("10:00","11:15")],   # Thursday
}
# Files
METRICS_CSV = "vsrp_metrics.csv"      # stores daily ATM IV, RR25, IVRV, IV Rank
TRADE_LOG_CSV = "vsrp_trade_log.csv"  # optional: your executions with timestamps
SKew_ZONES_CSV = "vsrp_skew_zones.csv"

# Position sizing ladder (optional) — use your own equity file or set manually
BASE_EQUITY = 20_000.0
BASE_RISK_PCT = 0.005                  # 0.5% per trade at baseline
SCALE_UP_AT   = 0.30                   # +30% since baseline -> 1.5x
FALLBACK_AT   = -0.15                  # -15% drawdown -> 0.5x

# ---------------------------- UTILS ---------------------------- #
def _annualized_rv_from_prices(prices: pd.Series, window: int = 20) -> float:
    if prices is None or len(prices) < window + 2:
        return np.nan
    rets = np.log(prices).diff().dropna()
    return rets.tail(window).std() * sqrt(252)

def _choose_expiry(tkr: str) -> Optional[str]:
    try:
        opts = yf.Ticker(tkr).options
        if not opts:
            return None
        today = datetime.now(timezone.utc).date()
        picks = []
        for d in opts:
            dt = datetime.strptime(d, "%Y-%m-%d").date()
            dte = (dt - today).days
            if dte <= 0:
                continue
            if MIN_DTE <= dte <= 365:   # hard cap
                picks.append((abs(dte - TARGET_DTE), dte, d))
        if not picks:
            return None
        picks.sort(key=lambda x: x[0])
        return picks[0][2]
    except Exception:
        return None

def _bs_delta(S: float, K: float, T: float, r: float, sigma: float, q: float, is_call: bool) -> float:
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return np.nan
    d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))
    if is_call:
        return exp(-q * T) * norm.cdf(d1)
    return -exp(-q * T) * norm.cdf(-d1)

def _mid(row) -> float:
    bid = float(row.get("bid", np.nan))
    ask = float(row.get("ask", np.nan))
    last = float(row.get("lastPrice", np.nan))
    if np.isfinite(bid) and np.isfinite(ask) and ask >= bid and ask > 0:
        return (bid + ask) / 2.0
    return last if np.isfinite(last) else np.nan

def _attach_delta(df: pd.DataFrame, S: float, T: float, r: float, q: float, is_call: bool) -> pd.DataFrame:
    if df is None or df.empty or "impliedVolatility" not in df.columns:
        return pd.DataFrame()
    w = df.copy()
    w = w[(w.get("openInterest", 0) >= OI_MIN) & (w.get("volume", 0) >= VOL_MIN)]
    if w.empty:
        return pd.DataFrame()
    sig = w["impliedVolatility"].astype(float).clip(lower=1e-6, upper=5.0)
    K   = w["strike"].astype(float)
    w["delta"] = [_bs_delta(S, k, T, RISK_FREE_RATE, s, DIV_YIELD, is_call) for k, s in zip(K, sig)]
    w["mid"] = w.apply(_mid, axis=1)
    w = w.dropna(subset=["delta", "impliedVolatility", "mid"])
    return w

def _nearest_by_delta(df: pd.DataFrame, target_delta: float, side: str) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    i = (df["delta"] - target_delta).abs().idxmin()
    row = df.loc[i]
    row = row.copy()
    row["side"] = side
    return row

def _atm_iv(calls: pd.DataFrame, puts: pd.DataFrame, S: float) -> float:
    ivs = []
    for df in (calls, puts):
        if df is None or df.empty:
            continue
        idx = (df["strike"] - S).abs().idxmin()
        iv = float(df.loc[idx, "impliedVolatility"])
        if np.isfinite(iv):
            ivs.append(iv)
    return float(np.nanmean(ivs)) if ivs else np.nan

def _momentum_bias(tkr: str) -> Optional[str]:
    df = yf.download(tkr, period="6mo", progress=False)
    if df.empty:
        return None
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA90"] = df["Close"].rolling(MOMENTUM_WINDOW).mean()
    r = df.iloc[-1]
    if r["Close"] > r["SMA20"] > r["SMA50"]:
        return "bull"
    if r["Close"] < r["SMA20"] < r["SMA50"]:
        return "bear"
    return "neutral"

def _get_next_earnings_date(tkr: str) -> Optional[datetime.date]:
    try:
        edf = yf.Ticker(tkr).get_earnings_dates(limit=1)
        if edf is None or edf.empty:
            return None
        # index holds the timestamp
        return pd.to_datetime(edf.index[0]).date()
    except Exception:
        return None

def _today_et():
    return datetime.now(TIMEZONE_ET)

def _in_deploy_window(now_et: Optional[datetime] = None) -> bool:
    now = now_et or _today_et()
    wd = (now.weekday())  # Monday=0
    if wd not in DEPLOY_WINDOWS:
        return False
    hms = now.strftime("%H:%M")
    for start, end in DEPLOY_WINDOWS[wd]:
        if start <= hms <= end:
            return True
    return False

# ---------------------------- PERSISTENCE / IV RANK ---------------------------- #
def _load_metrics() -> pd.DataFrame:
    try:
        df = pd.read_csv(METRICS_CSV, parse_dates=["date"])
        return df
    except Exception:
        cols = ["date","ticker","price","dte","atm_iv","rv20","ivrv","rr25","iv_rank","iv_rank_delta"]
        return pd.DataFrame(columns=cols)

def _save_metrics(df: pd.DataFrame) -> None:
    df.to_csv(METRICS_CSV, index=False)

def _append_metric_row(row: dict) -> None:
    df = _load_metrics()
    rowdf = pd.DataFrame([row])
    mask = (df["date"] == row["date"]) & (df["ticker"] == row["ticker"])
    if mask.any():
        for k, v in row.items():
            df.loc[mask, k] = v
    else:
        df = pd.concat([df, rowdf], ignore_index=True)
    _save_metrics(df)

def _compute_iv_rank_from_history(ticker: str, today_iv: float, date: pd.Timestamp, window_days: int = 252) -> Tuple[float, float]:
    df = _load_metrics()
    df_t = df[df["ticker"] == ticker].copy()
    if df_t.empty:
        return np.nan, np.nan
    df_t = df_t.sort_values("date")
    # only last N days window
    start_cut = date - pd.Timedelta(days=window_days*1.2)  # a little extra margin
    df_t = df_t[df_t["date"] >= start_cut]
    series = df_t["atm_iv"].astype(float).dropna()
    if len(series) < 10:               # need at least some history
        return np.nan, np.nan
    iv_min, iv_max = float(series.min()), float(series.max())
    if not np.isfinite(iv_min) or not np.isfinite(iv_max) or iv_max <= iv_min:
        return np.nan, np.nan
    rank_today = (today_iv - iv_min) / (iv_max - iv_min) * 100.0
    prev_iv = float(series.iloc[-1]) if len(series) >= 1 else np.nan
    if np.isfinite(prev_iv) and iv_max > iv_min:
        prev_rank = (prev_iv - iv_min) / (iv_max - iv_min) * 100.0
    else:
        prev_rank = np.nan
    delta = rank_today - prev_rank if np.isfinite(prev_rank) else np.nan
    return float(rank_today), float(delta)

# ---------------------------- DATACLASSES ---------------------------- #
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
    kind: str                 # "Bull Call Debit", "Bear Put Debit", "Bull Put Credit", "Bear Call Credit"
    expiry: str
    width: float
    net: float                # debit (>0) or credit (>0)
    max_gain: float
    max_loss: float
    breakeven: Optional[float]
    long_leg: Optional[dict]  # {"strike":..., "delta":..., "mid":...}
    short_leg: Optional[dict]

# ---------------------------- CORE MEASURES ---------------------------- #
def _risk_reversal_25(calls: pd.DataFrame, puts: pd.DataFrame) -> float:
    c = _nearest_by_delta(calls, 0.25, "C")
    p = _nearest_by_delta(puts, -0.25, "P")
    if c is None or p is None:
        return np.nan
    return float(c["impliedVolatility"] - p["impliedVolatility"])

def _collect_snapshot(tkr: str) -> Optional[Tuple[Signal, pd.DataFrame, pd.DataFrame, float, float]]:
    try:
        t = yf.Ticker(tkr)
        spot = float(t.fast_info["last_price"]) if "last_price" in t.fast_info else float(t.history(period="1d")["Close"][-1])

        expiry = _choose_expiry(tkr)
        if expiry is None:
            return None

        # DTE
        today = datetime.now(timezone.utc).date()
        dte = (datetime.strptime(expiry, "%Y-%m-%d").date() - today).days
        T = max(dte, 1) / 365.0

        oc = t.option_chain(expiry)
        calls_raw, puts_raw = oc.calls.copy(), oc.puts.copy()
        calls = _attach_delta(calls_raw, spot, T, RISK_FREE_RATE, DIV_YIELD, True)
        puts  = _attach_delta(puts_raw,  spot, T, RISK_FREE_RATE, DIV_YIELD, False)

        atm_iv = _atm_iv(calls, puts, spot)
        rr25   = _risk_reversal_25(calls, puts)

        # Realized vol (20d)
        px = yf.download(tkr, period="1y", progress=False)["Close"]
        rv20 = _annualized_rv_from_prices(px, window=20)
        ivrv = atm_iv / rv20 if (np.isfinite(atm_iv) and np.isfinite(rv20) and rv20 > 0) else np.nan

        mom = _momentum_bias(tkr)

        # Earnings days
        edate = _get_next_earnings_date(tkr)
        e_days = None
        if edate:
            e_days = (edate - today).days

        sig = Signal(
            ticker=tkr, price=spot, expiry=expiry, dte=dte,
            atm_iv=atm_iv, rv20=rv20, ivrv_ratio=ivrv, rr25=rr25,
            momentum=mom, earnings_days=e_days
        )
        return sig, calls, puts, spot, T
    except Exception:
        return None

# ---------------------------- STRIKES / SPREAD BUILDER ---------------------------- #
def _cap_width(width: float, spot: float) -> float:
    return min(width, spot * SPREAD_WIDTH_LIMIT)

def _select_vertical_spread(calls: pd.DataFrame, puts: pd.DataFrame, S: float, T: float, expiry: str, kind: str) -> Optional[Spread]:
    if kind == "Bull Call Debit":
        long = _nearest_by_delta(calls, DEBIT_CALL_DELTAS[0], "LONG")
        short= _nearest_by_delta(calls, DEBIT_CALL_DELTAS[1], "SHORT")
        if long is None or short is None:
            return None
        if short["strike"] <= long["strike"]:
            # push short further OTM
            cands = calls[calls["strike"] > long["strike"]]
            if cands.empty:
                return None
            short = cands.iloc[(cands["delta"] - DEBIT_CALL_DELTAS[1]).abs().idxmin()]
        width = float(short["strike"] - long["strike"])
        width = _cap_width(width, S)
        net = float(long["mid"] - short["mid"])  # debit
        max_gain = width - net
        max_loss = net
        be = float(long["strike"] + net)
        return Spread(kind, expiry, width, net, max_gain, max_loss, be,
                      {"strike": float(long["strike"]), "delta": float(long["delta"]), "mid": float(long["mid"])},
                      {"strike": float(short["strike"]), "delta": float(short["delta"]), "mid": float(short["mid"])})

    if kind == "Bear Put Debit":
        long = _nearest_by_delta(puts, DEBIT_PUT_DELTAS[0], "LONG")
        short= _nearest_by_delta(puts, DEBIT_PUT_DELTAS[1], "SHORT")
        if long is None or short is None:
            return None
        if short["strike"] >= long["strike"]:
            cands = puts[puts["strike"] < long["strike"]]
            if cands.empty:
                return None
            short = cands.iloc[(cands["delta"] - DEBIT_PUT_DELTAS[1]).abs().idxmin()]
        width = float(long["strike"] - short["strike"])
        width = _cap_width(width, S)
        net = float(long["mid"] - short["mid"])  # debit
        max_gain = width - net
        max_loss = net
        be = float(long["strike"] - net)
        return Spread(kind, expiry, width, net, max_gain, max_loss, be,
                      {"strike": float(long["strike"]), "delta": float(long["delta"]), "mid": float(long["mid"])},
                      {"strike": float(short["strike"]), "delta": float(short["delta"]), "mid": float(short["mid"])})

    if kind == "Bull Put Credit":
        short= _nearest_by_delta(puts, CREDIT_PUT_DELTAS[0], "SHORT")
        long = _nearest_by_delta(puts, CREDIT_PUT_DELTAS[1], "LONG")
        if long is None or short is None:
            return None
        if long["strike"] >= short["strike"]:
            cands = puts[puts["strike"] < short["strike"]]
            if cands.empty:
                return None
            long = cands.iloc[(cands["delta"] - CREDIT_PUT_DELTAS[1]).abs().idxmin()]
        width = float(short["strike"] - long["strike"])
        width = _cap_width(width, S)
        net = float(short["mid"] - long["mid"])  # credit
        max_gain = net
        max_loss = width - net
        be = float(short["strike"] - net)
        return Spread(kind, expiry, width, net, max_gain, max_loss, be,
                      {"strike": float(long["strike"]), "delta": float(long["delta"]), "mid": float(long["mid"])},
                      {"strike": float(short["strike"]), "delta": float(short["delta"]), "mid": float(short["mid"])})

    if kind == "Bear Call Credit":
        short= _nearest_by_delta(calls, CREDIT_CALL_DELTAS[0], "SHORT")
        long = _nearest_by_delta(calls, CREDIT_CALL_DELTAS[1], "LONG")
        if long is None or short is None:
            return None
        if long["strike"] <= short["strike"]:
            cands = calls[calls["strike"] > short["strike"]]
            if cands.empty:
                return None
            long = cands.iloc[(cands["delta"] - CREDIT_CALL_DELTAS[1]).abs().idxmin()]
        width = float(long["strike"] - short["strike"])
        width = _cap_width(width, S)
        net = float(short["mid"] - long["mid"])  # credit
        max_gain = net
        max_loss = width - net
        be = float(short["strike"] + net)
        return Spread(kind, expiry, width, net, max_gain, max_loss, be,
                      {"strike": float(long["strike"]), "delta": float(long["delta"]), "mid": float(long["mid"])},
                      {"strike": float(short["strike"]), "delta": float(short["delta"]), "mid": float(short["mid"])})
    return None

# ---------------------------- SKEW MAGNET ZONES ---------------------------- #
def _skew_magnet_zones(tkr: str, calls: pd.DataFrame, puts: pd.DataFrame, spot: float) -> list[dict]:
    """Heuristic: find OI clusters around spot ±10% and post-gap shelves."""
    zones = []
    # OI clusters
    spread = 0.10 * spot
    mask_calls = (calls["strike"].between(spot - spread, spot + spread))
    mask_puts  = (puts["strike"].between(spot - spread, spot + spread))
    c_grp = calls[mask_calls].groupby("strike")["openInterest"].sum()
    p_grp = puts[mask_puts].groupby("strike")["openInterest"].sum()
    if not c_grp.empty or not p_grp.empty:
        total = (c_grp + p_grp).fillna(0.0)
        if not total.empty:
            peaks = total.sort_values(ascending=False).head(3)
            for k, v in peaks.items():
                zones.append({"ticker": tkr, "level": float(k), "type": "OI_cluster", "weight": int(v)})

    # Gap shelves (post-gap consolidation zones)
    px = yf.download(tkr, period="6mo", progress=False)
    if not px.empty:
        px["prev_close"] = px["Close"].shift(1)
        px["gap"] = (px["Open"] - px["prev_close"]) / px["prev_close"]
        gaps = px[px["gap"].abs() > 0.02].copy()
        for _, row in gaps.tail(5).iterrows():  # last 5 gaps
            day = _
            start = px.loc[day].name
            window = px.loc[start:].head(8)  # look at following week
            if len(window) >= 3:
                shelf = window["Close"].median()
                zones.append({"ticker": tkr, "level": float(shelf), "type": "gap_shelf", "weight": 1})
    return zones

# ---------------------------- STRATEGY DECISION ---------------------------- #
def _decide_play(sig: Signal) -> str:
    """VSRP gates: IV rich + skew extreme + momentum alignment, with IVR guard & time gate."""
    if not sig.time_gate:
        return "KILL (dead time)"
    if not np.isfinite(sig.ivrv_ratio) or sig.ivrv_ratio < IVRV_THRESHOLD:
        return "KILL (IV not rich)"
    if not np.isfinite(sig.skew_z) or abs(sig.skew_z) < SKEW_ZSCORE_THRESHOLD:
        return "KILL (skew not extreme)"
    if sig.momentum == "bull" and sig.rr25 > 0:
        # Switch to credit if IV rank too low (drain guard)
        return "Bull Put Credit" if (np.isfinite(sig.iv_rank) and sig.iv_rank < IVR_LOW_GUARD) else "Bull Call Debit"
    if sig.momentum == "bear" and sig.rr25 < 0:
        return "Bear Call Credit" if (np.isfinite(sig.iv_rank) and sig.iv_rank < IVR_LOW_GUARD) else "Bear Put Debit"
    return "KILL (bias disagree)"

# ---------------------------- RISK / SIZING ---------------------------- #
def _position_risk_unit(equity: float, peak_equity: Optional[float] = None) -> float:
    """Scale risk with performance: +30% => 1.5x; drawdown -15% => 0.5x."""
    if peak_equity is None:
        peak_equity = equity
    dd = (equity - peak_equity) / peak_equity
    mult = 1.0
    if dd >= SCALE_UP_AT:
        mult = 1.5
    elif dd <= FALLBACK_AT:
        mult = 0.5
    return equity * BASE_RISK_PCT * mult

# ---------------------------- MAIN SCAN ---------------------------- #
def _analyze_single(tkr: str):
    snap = _collect_snapshot(tkr)
    if snap is None:
        return None
    sig, calls, puts, spot, T = snap

    # Persist & compute IV Rank
    today = pd.Timestamp.utcnow().normalize()
    iv_rank, iv_rank_delta = _compute_iv_rank_from_history(sig.ticker, sig.atm_iv, today)
    sig.iv_rank = iv_rank
    sig.iv_rank_delta = iv_rank_delta
    sig.vega_flip = bool(np.isfinite(iv_rank_delta) and iv_rank_delta >= VEGA_FLIP_DELTA)

    # Time gate
    sig.time_gate = _in_deploy_window()

    # Fill metric row and persist
    _append_metric_row({
        "date": today, "ticker": sig.ticker, "price": sig.price, "dte": sig.dte,
        "atm_iv": sig.atm_iv, "rv20": sig.rv20, "ivrv": sig.ivrv_ratio, "rr25": sig.rr25,
        "iv_rank": sig.iv_rank, "iv_rank_delta": sig.iv_rank_delta
    })

    return sig, calls, puts, spot, T

def run_vsrp_scan(tickers=LIQUID_TICKERS) -> pd.DataFrame:
    # Parallel collection
    pods = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(_analyze_single, tkr): tkr for tkr in tickers}
        for f in as_completed(futs):
            res = f.result()
            if res is not None:
                pods.append(res)

    if not pods:
        return pd.DataFrame()

    # Cross-sectional skew z-score
    sigs = [p[0] for p in pods]
    rr_vals = np.array([s.rr25 for s in sigs], dtype=float)
    rr_z = zscore(np.where(np.isfinite(rr_vals), rr_vals, np.nan), nan_policy='omit')
    for s, z in zip(sigs, rr_z):
        s.skew_z = float(z) if np.isfinite(z) else np.nan
        s.play = _decide_play(s)

    # Build spreads + skew magnet zones
    rows = []
    all_zones = []
    for (sig, calls, puts, spot, T) in pods:
        # skew magnets
        zones = _skew_magnet_zones(sig.ticker, calls, puts, spot)
        all_zones.extend(zones)

        spread_detail = None
        if "Debit" in sig.play or "Credit" in sig.play:
            spread_detail = _select_vertical_spread(
                calls, puts, spot, T, sig.expiry, kind=sig.play
            )

        # Earnings IV trap flag (if close & IV pumped)
        earnings_flag = False
        if sig.earnings_days is not None and sig.earnings_days <= 10 and np.isfinite(sig.atm_iv) and sig.atm_iv >= 0.80:
            earnings_flag = True

        rows.append({
            "ticker": sig.ticker,
            "price": round(sig.price, 2),
            "expiry": sig.expiry,
            "dte": sig.dte,
            "atm_iv": round(sig.atm_iv, 4) if np.isfinite(sig.atm_iv) else np.nan,
            "rv20": round(sig.rv20, 4) if np.isfinite(sig.rv20) else np.nan,
            "ivrv_ratio": round(sig.ivrv_ratio, 2) if np.isfinite(sig.ivrv_ratio) else np.nan,
            "rr25": round(sig.rr25, 4) if np.isfinite(sig.rr25) else np.nan,
            "skew_z": round(sig.skew_z, 2) if np.isfinite(sig.skew_z) else np.nan,
            "iv_rank": round(sig.iv_rank, 1) if np.isfinite(sig.iv_rank) else np.nan,
            "iv_rank_delta": round(sig.iv_rank_delta, 1) if np.isfinite(sig.iv_rank_delta) else np.nan,
            "vega_flip": sig.vega_flip,
            "momentum": sig.momentum,
            "time_gate": sig.time_gate,
            "earnings_trap": earnings_flag,
            "play": sig.play,
            "spread": spread_detail.__dict__ if spread_detail else None
        })

    # Save zones snapshot (optional dashboard helper)
    if all_zones:
        zdf = pd.DataFrame(all_zones).drop_duplicates()
        try:
            prev = pd.read_csv(SKew_ZONES_CSV)
            zdf = pd.concat([prev, zdf], ignore_index=True).drop_duplicates()
        except Exception:
            pass
        zdf.to_csv(SKew_ZONES_CSV, index=False)

    df = pd.DataFrame(rows).sort_values(by=["ivrv_ratio"], ascending=False, na_position="last").reset_index(drop=True)
    return df

# ---------------------------- HEATMAP FROM TRADE LOG ---------------------------- #
def time_scaling_heatmap(trade_log_csv: str = TRADE_LOG_CSV) -> Optional[pd.DataFrame]:
    """Expect trade log with columns: timestamp (ISO), ticker, R (float)."""
    try:
        log = pd.read_csv(trade_log_csv, parse_dates=["timestamp"])
    except Exception:
        return None
    if log.empty:
        return None
    log["hour"] = log["timestamp"].dt.tz_localize("UTC").dt.tz_convert(TIMEZONE_ET).dt.hour
    log["dow"]  = log["timestamp"].dt.tz_localize("UTC").dt.tz_convert(TIMEZONE_ET).dt.weekday
    # Win = R>0
    agg = log.groupby(["dow","hour"]).agg(
        win_rate=("R", lambda x: np.mean(x>0) if len(x)>0 else np.nan),
        avg_R=("R", "mean"),
        n=("R", "count")
    ).reset_index()
    return agg

# ---------------------------- ENTRY ---------------------------- #
if __name__ == "__main__":
    df = run_vsrp_scan()
    print("\n🔍 VSRP — Hooks Edition 🔍")
    if df.empty:
        print("No data. Check network or ticker list.")
    else:
        cols = ["ticker","price","expiry","dte","ivrv_ratio","rr25","skew_z","iv_rank","iv_rank_delta","vega_flip","momentum","time_gate","earnings_trap","play"]
        print(df[cols].to_string(index=False))

        # Position sizing suggestion (toy ladder)
        equity = BASE_EQUITY
        risk_unit = _position_risk_unit(equity, peak_equity=equity)
        print(f"\nRisk unit (baseline): ${risk_unit:,.2f} per trade.")

        # Spread details
        for _, row in df.iterrows():
            if isinstance(row["spread"], dict):
                s = row["spread"]
                print(f"\n{row['ticker']} → {row['play']} @ {row['expiry']}")
                print(f"  width: {s['width']:.2f} | net {'debit' if 'Debit' in row['play'] else 'credit'}: {s['net']:.2f}")
                print(f"  max_gain: {s['max_gain']:.2f} | max_loss: {s['max_loss']:.2f} | breakeven: {s['breakeven']:.2f}")
                ll = s['long_leg']; sl = s['short_leg']
                if ll: print(f"  LONG  strike {ll['strike']:.2f} Δ{ll['delta']:.2f} mid {ll['mid']:.2f}")
                if sl: print(f"  SHORT strike {sl['strike']:.2f} Δ{sl['delta']:.2f} mid {sl['mid']:.2f}")
