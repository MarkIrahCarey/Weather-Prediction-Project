from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
import joblib

# Load the dataset
df = pd.read_csv("combined_weather_data_reduced.csv")

# Ensure 'datetime' is in pandas datetime format
df['datetime'] = pd.to_datetime(df['datetime'])

# Define features (X) and target (y)
features = ['longitude', 'latitude', 'hour', 'windspeed', 'precipitation', 'day_sin', 'day_cos']
target = 'temperature'

# garb a subset from the dataset
X = df[features]
y = df[target]

# Split data into train and test sets
print("Splitting Data")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training Model...")

# Make the model
model = RandomForestRegressor(
    n_estimators=30,      
    max_depth=10,          
    n_jobs=-1,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)  
print("Model Done Training!")

# Evaluate the model for accuracy
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred), sep=" ")
print("R2:", r2_score(y_test, y_pred), sep=" ")

# Save the model to a file
print("Saving the model...")
joblib.dump(model, 'weather_prediction_v1_model.pkl')
print("Model saved successfully!")
