import io, math, requests
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Track Record", layout="wide")

# 改成你仓库的 raw CSV 链接
DEFAULT_CSV = "https://raw.githubusercontent.com/Zequn-Li/SisypheanStrategyTrackRecord/main/track_record.csv"

@st.cache_data(ttl=600)
def load_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    # 期待4列：date,equity,deposit, withdrawal
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return df

def compute_perf(df: pd.DataFrame):
    df = df.copy()
    prev = df["equity"].shift(1)
    df["ret"] = (df["equity"] - prev - df["deposit"] + df["withdrawal"]) / prev
    df["ret"] = df["ret"].fillna(0.0)
    df["nav"] = (1 + df["ret"]).cumprod()

    ann_factor = 252
    ann_ret = (1 + df["ret"]).prod() ** (ann_factor / max(len(df), 1)) - 1
    ann_vol = df["ret"].std(ddof=0) * math.sqrt(ann_factor)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan

    cum = df["nav"] - 1
    run_max = (cum + 1).cummax()
    drawdown = (cum + 1) / run_max - 1
    max_dd = float(drawdown.min()) if len(drawdown) else 0.0

    return df, ann_ret, ann_vol, sharpe, drawdown, max_dd

st.title("Strategy Track Record")
st.caption("Source: Alpaca account. Cashflows adjusted. For research display only.")

csv_url = st.text_input("CSV URL", value=DEFAULT_CSV)

try:
    raw = load_csv(csv_url)
    df, ann_ret, ann_vol, sharpe, dd, max_dd = compute_perf(raw)

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("NAV (start = 1.00)")
        st.line_chart(df.set_index("date")[["nav"]])

        st.subheader("Drawdown")
        st.line_chart(pd.DataFrame({"drawdown": dd.values}, index=df["date"]))

    with right:
        st.subheader("Summary")
        summary = pd.DataFrame({
            "Metric": ["Period", "Total Return", "Annualized Return", "Annualized Volatility", "Sharpe", "Max Drawdown"],
            "Value": [
                f"{df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()}",
                f"{(df['nav'].iloc[-1]-1)*100:.2f}%",
                f"{ann_ret*100:.2f}%",
                f"{ann_vol*100:.2f}%",
                f"{sharpe:.2f}",
                f"{max_dd*100:.2f}%"
            ],
        })
        st.dataframe(summary, hide_index=True, use_container_width=True)

    st.markdown("Notes: returns use daily equity and cashflow; Sharpe uses 252 trading days.")

except Exception as e:
    st.error(f"Failed to load or parse CSV: {e}")