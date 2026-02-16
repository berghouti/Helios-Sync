import numpy as np
import tensorflow as tf
from stable_baselines3 import PPO


class SmartController:
    def __init__(self, lstm_path='lstm.keras', ppo_path='smart_energy_agent_v1.zip'):
        # 1. Load the Forecaster (LSTM)
        self.forecaster = tf.keras.models.load_model(lstm_path)
        # 2. Load the Decider (RL)
        self.agent = PPO.load(ppo_path)
        # 3. Memory for LSTM (needs 24 hours of history)
        self.history = []

    def get_action(self, raw_features, soc):
        # A. Scale features (0-1) so the LSTM and RL can 'see' them
        scaled_feat = [
            np.clip(raw_features[0] / 5.0, 0, 1),  # Solar
            np.clip(raw_features[1] / 5.0, 0, 1),  # Load
            np.clip((raw_features[2] - 10) / 40, 0, 1),  # Temp
            np.clip(raw_features[3] / 100, 0, 1),  # Cloud
            raw_features[4],  # Hour_Sin
            raw_features[5]  # Hour_Cos
        ]

        # B. LSTM Memory Management
        self.history.append(scaled_feat)
        if len(self.history) > 24: self.history.pop(0)

        # C. Generate Forecast
        if len(self.history) == 24:
            input_seq = np.array(self.history).reshape(1, 24, 6)
            # LSTM predicts the next hour's solar and load
            forecast = self.forecaster.predict(input_seq, verbose=0)[0]
        else:
            # Fallback if we don't have 24 hours yet
            forecast = [scaled_feat[0], scaled_feat[1]]

        # D. RL Decision
        # State: [Solar, Load, Temp, Cloud, Hour_Sin, Hour_Cos, SoC]
        state = np.array([
            scaled_feat[0], scaled_feat[1], scaled_feat[2],
            scaled_feat[3], scaled_feat[4], scaled_feat[5],
            soc
        ], dtype=np.float32).reshape(1, -1)

        action, _ = self.agent.predict(state, deterministic=True)
        return int(action[0]), forecast