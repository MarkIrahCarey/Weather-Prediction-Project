import pandas as pd
import numpy as np
import os

# Set folder containing CSVs
folder_path = "Temperature_Data/"
all_dfs = []

for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path, parse_dates=['datetime'])

        # Reduce the training set
        df = df[
            (df['datetime'].dt.date == pd.to_datetime('2024-05-25').date()) &
            (df['datetime'].dt.hour == 3)
        ]


        # Standardize column names
        df = df.rename(columns={
            'temperature_2m': 'temperature',
            'wind_speed_10m': 'windspeed',
            'rain': 'precipitation'
        })

        all_dfs.append(df)

# Combine all files into one DataFrame
full_df = pd.concat(all_dfs, ignore_index=True)

# Drop rows where temperature or datetime is missing
full_df = full_df.dropna(subset=['temperature', 'datetime'])

# Extract features
full_df['hour'] = full_df['datetime'].dt.hour
full_df['day_of_year'] = full_df['datetime'].dt.dayofyear

# Cyclical encoding for day of year
full_df['day_sin'] = np.sin(2 * np.pi * full_df['day_of_year'] / 365)
full_df['day_cos'] = np.cos(2 * np.pi * full_df['day_of_year'] / 365)

# Fill missing values in weather columns
full_df['precipitation'] = full_df['precipitation'].fillna(0)
full_df['windspeed'] = full_df['windspeed'].fillna(0)

# Save to a CSV with 7 features and the target variable
save_df = full_df[['datetime', 'latitude', 'longitude', 'hour', 'windspeed', 'precipitation', 'day_sin', 'day_cos', 'temperature']]

# Save and download
save_path = "/content/combined_weather_data.csv"
save_df.to_csv(save_path, index=False)