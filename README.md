#  Helios-Sync: Intelligent Microgrid Energy Management

![Status](https://img.shields.io/badge/Status-Prototype_Ready-success)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Framework](https://img.shields.io/badge/Framework-Streamlit%20|%20TensorFlow-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

##  Executive Summary
**Helios-Sync** is an AI-driven energy management system (EMS) designed to solve the intermittency problem of renewable energy in the Djerma region of Algeria. By synchronizing solar production with consumption patterns, the system minimizes energy waste and maximizes grid stability.

Unlike traditional rule-based controllers, Helios-Sync utilizes a **Hybrid AI Architecture**:
1.  **Forecasting:** Predicting future solar irradiance and building load.
2.  **Optimization:** dynamically controlling battery storage to "Peak Shave" during high-demand periods.
3.  **Maintenance:** Detecting solar panel efficiency loss due to dust/soiling using Computer Vision.

---
## Project Structure
├── app.py                 # Main Streamlit Dashboard Application
├── digital_twin.py        # Physics simulation (Battery & Environment)
├── agent_bridge.py        # Interface for PPO & LSTM Model Inference
├── requirements.txt       # Python dependencies
│── ppo_agent.zip      # RL Agent
│── lstm_forecast.h5   # Solar forecaster
└── data/                  # Weather and Load datasets 
##  System Architecture (Sense-Think-Act)

The system operates on a closed-loop control cycle:

1.  **Sense (Data Layer):** Aggregates telemetry from the Digital Twin (Battery SoC), Smart Meters (Load), and Weather APIs (Djerma historical data).
2.  **Think (Intelligence Layer):**
    * **LSTM:** Forecasts generation for the next 24 hours.
    * **PPO Agent:** Decides optimal battery dispatch (Charge/Discharge/Idle).
    * **CNN:** Analyzes panel imagery for dust accumulation.
3.  **Act (Control Layer):** Executes the dispatch command to the battery inverter and updates the **Mission Control Dashboard**.

---

##  AI Models & Training Pipelines (Kaggle)

The core intelligence of Helios-Sync was trained on Kaggle using accelerated GPUs. Below are the links to the training notebooks and model architectures.

| Module Name | AI Architecture | Functionality | Training Notebook |
| :--- | :--- | :--- | :--- |
| **Helios Brain** | **LSTM + PPO (RL)** | **Energy Optimization.** Combines Time-Series Forecasting (LSTM) with Reinforcement Learning (PPO) to learn the optimal policy for battery charging and discharging. | [🔗 View Kaggle Notebook (RL/LSTM)](https://www.kaggle.com/code/nadjibtitaouine/idea-track) |
| **Helios Vision** | **CNN (ResNet50)** | **Dust Detection.** A Computer Vision model trained on the *Solar Panel Dust Dataset* to detect soiling and trigger maintenance alerts to prevent efficiency loss. | [🔗 View Kaggle Notebook (CNN)](https://www.kaggle.com/code/berghouti/dust-detection) |


---

## Key Features

### 1. Intelligent Dispatch
Uses Reinforcement Learning to decide *when* to store energy. It automatically charges during peak sun hours and discharges during evening peak load, smoothing the curve.

### 2. Predictive Analytics
Instead of reacting to the present, the system looks 24 hours ahead using Long Short-Term Memory (LSTM) networks to predict cloud cover and load spikes.

### 3. Sustainability Tracking
The dashboard translates Kilowatt-hours (kWh) into tangible environmental metrics:
* **Carbon Offset (kg)**
* **Tree Absorption Equivalent**

### 4. Digital Twin Simulation
A physics-based simulation of battery thermodynamics, including State of Charge (SoC) limits, round-trip efficiency losses, and degradation risks.

---

##  Launching 
    streamlit run app.py

### Prerequisites
* Python 3.8 or higher
* Git

### 1. Clone the Repository
```bash
git clone [https://github.com/berghouti/helios-sync.git](https://github.com/berghouti/helios-sync.git)
cd helios-sync
