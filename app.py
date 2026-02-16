import streamlit as st
import pandas as pd
import numpy as np
from digital_twin import DigitalTwin
from agent_bridge import SmartController

st.set_page_config(layout="wide", page_title="Djerma Smart Grid EMS")

# --- System Initialization ---
if 'sim' not in st.session_state:
    st.session_state.sim = DigitalTwin()
if 'ctrl' not in st.session_state:
    st.session_state.ctrl = SmartController()
if 'logs' not in st.session_state:
    st.session_state.logs = []
if 'co2_saved' not in st.session_state:
    st.session_state.co2_saved = 0.0

# Administrative System Reset
if st.sidebar.button("Initialize System Reset"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

st.title("Helios-Sync")
st.markdown("---")

# --- Primary Engineering Metrics (KPI Bar) ---
if st.session_state.logs:
    last = st.session_state.logs[-1]
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    kpi1.metric("Battery State", f"{last['SoC'] * 100:.1f}%", f"{last['Action']}")
    kpi2.metric("Solar Generation", f"{last['Solar']} kW")
    kpi3.metric("Building Load", f"{last['Load']} kW")
    kpi4.metric("Carbon Offset", f"{st.session_state.co2_saved:.2f} kg")

# --- Main Analytics Interface ---
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("Energy Balance: Generation vs. Demand")
    if st.session_state.logs:
        df = pd.DataFrame(st.session_state.logs)

        # CHANGED: Removed "Forecast" from the visualization list
        # Now it only shows the physical reality (Solar vs Load)
        st.line_chart(df.set_index("Time")[["Solar", "Load"]])

        st.subheader("Storage Unit State of Charge")
        st.area_chart(df.set_index("Time")["SoC"])
    else:
        st.info("System Standby. Awaiting telemetry synchronization.")

with col_right:
    st.subheader("System Control Interface")
    if st.button("ADVANCE OPERATIONAL CYCLE (1 HOUR)", use_container_width=True):
        # 1. Telemetry Capture
        row, soc = st.session_state.sim.step(0)
        feat = [row['actual_solar_gen'], row['actual_load_kw'], row['temperature_2m'],
                row['cloud_cover'], row['hour_sin'], row['hour_cos']]

        # 2. Predictive Computation (LSTM + RL)
        action, forecast = st.session_state.ctrl.get_action(feat, soc)

        # 3. Command Execution
        next_row, final_soc = st.session_state.sim.step(action)

        # Environmental Impact Calculation
        energy_from_battery = next_row['actual_load_kw'] if action == 2 else 0
        st.session_state.co2_saved += (energy_from_battery * 0.45)

        st.session_state.logs.append({
            "Time": str(next_row['time']),
            "Solar": round(next_row['actual_solar_gen'], 2),
            "Load": round(next_row['actual_load_kw'], 2),
            "Forecast": round(forecast[0] * 5.0, 2),  # We keep calculating it for logs, just not plotting it
            "Action": ["Idle", "Charging", "Discharging"][action],
            "SoC": round(final_soc, 3),
            "Temp": round(next_row['temperature_2m'], 1),
            "Cloud": next_row['cloud_cover'],
            "Humidity": next_row['relative_humidity_2m'],
            "Radiation": round(next_row['shortwave_radiation'], 1)
        })

    st.subheader("Real-Time Event Telemetry")
    if st.session_state.logs:
        df = pd.DataFrame(st.session_state.logs)
        st.dataframe(df.tail(12)[["Time", "Action", "SoC", "Solar", "Load"]],
                     use_container_width=True, hide_index=True)

# --- Technical Analytics Footer ---
if st.session_state.logs:
    st.markdown("---")
    st.subheader("Environmental and Sustainability Analytics")
    f1, f2, f3, f4 = st.columns(4)

    with f1:
        st.write("**Weather Physics**")
        st.caption(f"Cloud Cover: {last['Cloud']}%")
        st.caption(f"Relative Humidity: {last['Humidity']}%")
        st.caption(f"Solar Irradiance: {last['Radiation']} W/m²")

    with f2:
        st.write("**Environmental Impact**")
        st.caption(f"Tree Absorption Equivalent: {st.session_state.co2_saved / 21:.2f} trees")
        st.caption(f"Emissions Avoided: {st.session_state.co2_saved:.2f} kg CO2")

    with f3:
        st.write("**Grid Resilience**")
        peak_shave = "Active" if last['Action'] == "Discharging" else "Standby"
        st.caption(f"Peak Shaving Status: {peak_shave}")
        st.caption(f"Grid Stress Index: Nominal")

    with f4:
        st.write("**System Infrastructure**")
        st.caption("Forecaster: LSTM-v2.1")
        st.caption("Policy Protocol: PPO-Recurrent")
        st.caption("Temporal Lookback: 24 Hours")