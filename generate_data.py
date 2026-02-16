import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

# 1. FETCH DATA (Same logic as your notebook)
LAT, LON = 35.6167, 6.2833
end_date = datetime.now().date() - timedelta(days=1)
start_date = end_date - timedelta(days=60)


def fetch_and_process():
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LAT, "longitude": LON,
        "start_date": start_date, "end_date": end_date,
        "hourly": ["temperature_2m", "relative_humidity_2m", "cloud_cover", "shortwave_radiation"],
        "timezone": "UTC"
    }

    print("🛰️ Fetching weather data from Open-Meteo...")
    response = requests.get(url, params=params)
    df = pd.DataFrame(response.json()['hourly'])
    df['time'] = pd.to_datetime(df['time'])

    # 2. FUSE WITH LOAD (Simulating your notebook's logic)
    # Using a synthetic load pattern since the UCI raw link can be slow
    hours = len(df)
    hour_of_day = df['time'].dt.hour
    base_load = 1.0 + 2.0 * np.exp(-0.5 * ((hour_of_day - 20) / 2) ** 2)
    df['actual_load_kw'] = base_load + np.random.normal(0, 0.1, hours)

    # Physics for Solar Gen
    df['actual_solar_gen'] = (df['shortwave_radiation'] * 20 * 0.20) / 1000

    # 3. FEATURE ENGINEERING
    df['hour_sin'] = np.sin(2 * np.pi * hour_of_day / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour_of_day / 24)
    df['day_of_week'] = df['time'].dt.dayofweek
    df['load_lag_24h'] = df['actual_load_kw'].shift(24)
    df['solar_lag_24h'] = df['actual_solar_gen'].shift(24)
    df['target_solar_next_h'] = df['actual_solar_gen'].shift(-1)
    df['target_load_next_h'] = df['actual_load_kw'].shift(-1)

    df = df.dropna().reset_index(drop=True)

    # Save the file the app is looking for
    df.to_csv("engineered_energy_data.csv", index=False)
    print("✅ Success! 'engineered_energy_data.csv' created.")


if __name__ == "__main__":
    fetch_and_process()