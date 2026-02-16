import pandas as pd
import numpy as np


class DigitalTwin:
    def __init__(self, csv_path="engineered_energy_data.csv"):
        self.data = pd.read_csv(csv_path)
        self.current_step = 0
        self.soc = 0.5  # Start at 50%
        self.battery_capacity = 10.0
        self.max_charge_rate = 2.0

    def step(self, action):
        row = self.data.iloc[self.current_step]
        solar = row['actual_solar_gen']
        load = row['actual_load_kw']

        # YOUR PHYSICS LOGIC from the notebook
        if action == 1:  # CHARGE
            charge_amount = min(solar, self.max_charge_rate, (1.0 - self.soc) * self.battery_capacity)
            self.soc += charge_amount / self.battery_capacity
        elif action == 2:  # DISCHARGE
            discharge_amount = min(load, self.max_charge_rate, self.soc * self.battery_capacity)
            self.soc -= discharge_amount / self.battery_capacity

        self.current_step += 1
        return row, self.soc