# ---------------------------- JOURNAL / HEATMAP ---------------------------- #
st.divider()
st.header("🧠 Discipline: Log trades → learn your kill zones")

with st.expander("Log a trade (R-based)"):
    from datetime import datetime, timezone

    colA, colB, colC, colD = st.columns(4)
    t_ticker = colA.text_input("Ticker", value="MSFT")
    t_R = colB.number_input("R result", value=0.0, step=0.25, format="%.2f")

    # Robust to Streamlit versions: prefer datetime_input; fallback to date+time inputs
    utc_now = datetime.utcnow()
    if hasattr(st, "datetime_input"):
        exec_dt = colC.datetime_input("Execution time (UTC)", value=utc_now)
    else:
        d = colC.date_input("Execution date (UTC)", value=utc_now.date())
        # time_input requires a time, not a datetime
        t = colD.time_input("Execution time (UTC)", value=utc_now.time())
        exec_dt = datetime.combine(d, t)

    # Make sure it's timezone-aware UTC for clean CSVs
    if exec_dt.tzinfo is None:
        exec_dt = exec_dt.replace(tzinfo=timezone.utc)
    else:
        exec_dt = exec_dt.astimezone(timezone.utc)

    add_btn = st.button("Add to log")
    if add_btn:
        try:
            logdf = pd.read_csv(TRADE_LOG_CSV)
        except Exception:
            logdf = pd.DataFrame(columns=["timestamp","ticker","R"])
        logdf = pd.concat(
            [logdf, pd.DataFrame([{"timestamp": exec_dt.isoformat(), "ticker": t_ticker.upper(), "R": t_R}])],
            ignore_index=True
        )
        logdf.to_csv(TRADE_LOG_CSV, index=False)
        st.success("Logged.")
        st.download_button("Download log CSV", logdf.to_csv(index=False).encode(),
                           file_name="vsrp_trade_log.csv", mime="text/csv")

with st.expander("Show time‑of‑day heatmap data"):
    try:
        log = pd.read_csv(TRADE_LOG_CSV)
        # Parse to aware UTC; don't tz_localize() an already-aware series
        log["timestamp"] = pd.to_datetime(log["timestamp"], utc=True)

        # Convert to ET for grouping
        log["timestamp_et"] = log["timestamp"].dt.tz_convert(TIMEZONE_ET)
        log["hour"] = log["timestamp_et"].dt.hour
        log["dow"]  = log["timestamp_et"].dt.weekday

        agg = (log.groupby(["dow","hour"])
                  .agg(win_rate=("R", lambda x: float(np.mean(x > 0)) if len(x) else np.nan),
                       avg_R=("R","mean"),
                       n=("R","count"))
                  .reset_index()
                  .sort_values(["dow","hour"]))
        st.dataframe(agg, use_container_width=True, height=260)
    except Exception as e:
        st.info("No trade log yet — once you log, this will map your **win‑rate by hour** to hard‑code deploy windows.")
