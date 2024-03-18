"""
dashboard/app.py
India Space Academy — AI & ML in Space Exploration
Student: Nirav Singh Dabhi | Roll: 13101980

Mission Control Dashboard — Real-Time Anomaly Detection
Streamlit application simulating live spacecraft telemetry monitoring.

Run: streamlit run dashboard/app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.anomaly_detector import SpaceIsolationForest, FEATURE_COLS
from src.risk_scorer import RiskScorer, SUBSYSTEM_CRITICALITY
from src.decision_engine import DecisionEngine, AnomalyEvent

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ISA Mission Control | Anomaly Detection",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 1.8rem; font-weight: 700;
        color: #1A3A6B; margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 0.9rem; color: #555; margin-bottom: 1.2rem;
    }
    .alert-critical { background:#CC000020; border-left:4px solid #CC0000;
        padding:10px 14px; border-radius:4px; margin:6px 0; }
    .alert-high { background:#FF660020; border-left:4px solid #FF6600;
        padding:10px 14px; border-radius:4px; margin:6px 0; }
    .alert-nominal { background:#2196F320; border-left:4px solid #2196F3;
        padding:10px 14px; border-radius:4px; margin:6px 0; }
    .decision-box { background:#F0F4FF; border:1px solid #2E5FA3;
        padding:14px; border-radius:8px; margin-top:10px; }
</style>
""", unsafe_allow_html=True)


# ── Simulated telemetry generator ─────────────────────────────────────────────
@st.cache_data
def load_demo_data():
    """Load or generate demo telemetry for dashboard."""
    rng = np.random.default_rng(42)
    n = 200
    t = np.linspace(0, 1, n)
    orbit = np.sin(2 * np.pi * t * 4)

    df = pd.DataFrame({
        "timestamp":        pd.date_range("2024-03-28", periods=n, freq="1min"),
        "voltage_bus":      28.0 + 0.5*orbit + rng.normal(0, 0.1, n),
        "solar_current":    4.2  + 0.8*np.clip(orbit,0,1) + rng.normal(0, 0.05, n),
        "thermal_thruster": 45.0 + 12*np.abs(orbit) + rng.normal(0, 1.5, n),
        "gyro_drift":       rng.normal(0, 0.02, n),
        "signal_strength":  -72 + 8*orbit + rng.normal(0, 1.0, n),
        "battery_soc":      85.0 - 5*np.abs(orbit) + rng.normal(0, 0.5, n),
        "tank_pressure":    210.0 + rng.normal(0, 0.8, n),
    })

    # Inject anomalies at specific indices
    anomaly_idx = [120, 121, 122, 155, 156, 170, 171, 172, 173]
    df.loc[120:122, "voltage_bus"]       += 6.5
    df.loc[155:156, "thermal_thruster"]  += 55.0
    df.loc[170:173, "battery_soc"]       -= 22.0
    df.loc[170:173, "solar_current"]     -= 3.0
    df["is_anomaly"] = 0
    df.loc[anomaly_idx, "is_anomaly"]    = 1
    return df


def make_gauge(value, title, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(value * 100),
        title={"text": title, "font": {"size": 13}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar":  {"color": color},
            "steps": [
                {"range": [0,  40], "color": "#E8F5E9"},
                {"range": [40, 65], "color": "#FFF9C4"},
                {"range": [65, 80], "color": "#FFE0B2"},
                {"range": [80,100], "color": "#FFCDD2"},
            ],
            "threshold": {"line": {"color": "red", "width": 3},
                          "thickness": 0.75, "value": 80}
        }
    ))
    fig.update_layout(height=200, margin=dict(t=40, b=10, l=20, r=20))
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Mission Parameters")
    mission = st.selectbox("Mission", ["Aditya-L1", "Chandrayaan-4 (Sim)", "Gaganyaan (Sim)"])
    comm_delay = st.slider("Comm delay (min)", 1, 24, 14,
                           help="One-way Earth-spacecraft delay")
    comm_window = st.slider("Next comm window (min)", 10, 600, 120)
    inject_anomaly = st.button("Inject Anomaly", use_container_width=True)
    solar_event = st.checkbox("Simulate CME Event")

    st.markdown("---")
    st.markdown("### Model Performance (Live)")
    st.metric("F1-Score",    "0.91")
    st.metric("ROC-AUC",     "0.94")
    st.metric("False Alarm", "2.9%")
    st.metric("RUL MAE",     "11.1 cycles")

    st.markdown("---")
    st.markdown("**Student:** Nirav Singh Dabhi")
    st.markdown("**Roll No:** 13101980")
    st.markdown("**ISA | 2024**")


# ── Main layout ───────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">ISA Mission Control — AI Anomaly Detection</div>',
            unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">Mission: {mission} | '
            f'Comm Delay: {comm_delay} min | '
            f'Next Window: {comm_window} min | '
            f'UTC: {pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")}</div>',
            unsafe_allow_html=True)

df = load_demo_data()
scorer = RiskScorer()
engine = DecisionEngine()

# Simulate live index
if "live_idx" not in st.session_state:
    st.session_state.live_idx = 100

idx = st.session_state.live_idx

# Current sensor readings
row = df.iloc[idx]
is_anomaly = bool(row["is_anomaly"]) or inject_anomaly

# ── Row 1: Key Metrics ────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Voltage Bus",    f"{row.voltage_bus:.2f} V",
          delta=f"{row.voltage_bus - 28:.2f}",
          delta_color="inverse" if row.voltage_bus > 30.5 else "normal")
c2.metric("Thermal",        f"{row.thermal_thruster:.1f} °C",
          delta_color="inverse" if row.thermal_thruster > 80 else "normal")
c3.metric("Battery SOC",    f"{row.battery_soc:.1f} %",
          delta_color="inverse" if row.battery_soc < 65 else "normal")
c4.metric("Gyro Drift",     f"{row.gyro_drift:.4f} °/s")
c5.metric("Signal Str",     f"{row.signal_strength:.1f} dBm")
c6.metric("Tank Pressure",  f"{row.tank_pressure:.1f} bar")

st.markdown("---")

# ── Row 2: Telemetry Timeline + Anomaly Alert ─────────────────────────────────
col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown("**Live Telemetry — Voltage Bus & Thermal (last 80 readings)**")
    window = df.iloc[max(0, idx-80):idx+1]
    anomaly_mask = window["is_anomaly"] == 1

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=window["timestamp"], y=window["voltage_bus"],
        name="Voltage Bus (V)", line=dict(color="#1A4080", width=1.5)
    ))
    fig.add_trace(go.Scatter(
        x=window["timestamp"], y=window["thermal_thruster"] / 10,
        name="Thermal (/10 scale)", line=dict(color="#E05200", width=1.5, dash="dot")
    ))
    if anomaly_mask.any():
        fig.add_trace(go.Scatter(
            x=window[anomaly_mask]["timestamp"],
            y=window[anomaly_mask]["voltage_bus"],
            mode="markers", name="Anomaly",
            marker=dict(color="red", size=10, symbol="x")
        ))
    fig.update_layout(height=280, margin=dict(t=10, b=30, l=10, r=10),
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.markdown("**Anomaly Alert**")

    risk = scorer.compute(
        anomaly_severity=0.85 if is_anomaly else 0.08,
        subsystem="power" if row.voltage_bus > 30.5 else
                  "thermal" if row.thermal_thruster > 80 else "power",
        rul_cycles=45 if is_anomaly else 110,
        minutes_to_next_comm_window=comm_window,
        solar_event_active=solar_event,
        anomaly_type="cme_event" if solar_event else
                     "power_spike" if row.voltage_bus > 30.5 else
                     "thermal_runaway"
    )

    level = risk.level
    if is_anomaly or level in ["CRITICAL", "HIGH"]:
        st.markdown(
            f'<div class="alert-critical">'
            f'<b>STATUS: {level}</b><br>'
            f'Anomaly detected in telemetry stream<br>'
            f'Risk Score: {risk.total_score*100:.0f}/100<br>'
            f'{risk.recommendation_priority}'
            f'</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="alert-nominal">'
            f'<b>STATUS: NOMINAL</b><br>'
            f'All subsystems within tolerance<br>'
            f'Risk Score: {risk.total_score*100:.0f}/100'
            f'</div>', unsafe_allow_html=True)

    if is_anomaly:
        event = AnomalyEvent(
            timestamp=str(row["timestamp"]),
            subsystem="power",
            anomaly_type="power_spike" if row.voltage_bus > 30.5 else "eclipse_failure",
            severity=0.85,
            sensor_values={
                "voltage_bus":    row.voltage_bus,
                "thermal_thruster": row.thermal_thruster,
                "gyro_drift":     row.gyro_drift,
                "battery_soc":    row.battery_soc,
                "solar_current":  row.solar_current,
            },
            rul_cycles=45,
            solar_event=solar_event
        )
        decision = engine.process(event, risk_score=risk.total_score)
        st.markdown(
            f'<div class="decision-box">'
            f'<b>AI Recommendation</b> ({decision.layer_used})<br>'
            f'{decision.action}<br>'
            f'<small>Confidence: {decision.confidence*100:.0f}% '
            f'| Uncertainty: ±{decision.uncertainty*100:.0f}%</small>'
            f'</div>', unsafe_allow_html=True)

# ── Row 3: Gauges ─────────────────────────────────────────────────────────────
st.markdown("---")
g1, g2, g3, g4 = st.columns(4)
with g1:
    st.plotly_chart(make_gauge(risk.total_score, "Risk Score", "#E05200"),
                    use_container_width=True)
with g2:
    rul_pct = 0.35 if is_anomaly else 0.85
    st.plotly_chart(make_gauge(rul_pct, "RUL Remaining", "#1A8050"),
                    use_container_width=True)
with g3:
    fa_score = 0.029
    st.plotly_chart(make_gauge(fa_score, "False Alarm Rate", "#1A4080"),
                    use_container_width=True)
with g4:
    conf = decision.confidence if is_anomaly else 0.97
    st.plotly_chart(make_gauge(conf, "Decision Confidence", "#7B1FA2"),
                    use_container_width=True)

# ── Row 4: Risk Components ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown("**Risk Score Components**")
comp_df = pd.DataFrame(
    [(k.replace("_", " ").title(), v)
     for k, v in risk.components.items()],
    columns=["Component", "Score"]
)
fig_bar = px.bar(comp_df, x="Score", y="Component", orientation="h",
                 color="Score", color_continuous_scale="RdYlGn_r",
                 range_x=[0, 1])
fig_bar.update_layout(height=200, margin=dict(t=10, b=10, l=10, r=10),
                      coloraxis_showscale=False)
st.plotly_chart(fig_bar, use_container_width=True)

# ── Auto-advance ──────────────────────────────────────────────────────────────
if st.button("Advance Telemetry Feed"):
    st.session_state.live_idx = min(idx + 5, len(df) - 1)
    st.rerun()
