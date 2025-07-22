import pandas as pd
import numpy as np
import joblib as jb
import warnings
import random as rand

# Suppress all warnings beacuse they are annoying
warnings.filterwarnings('ignore')

def date_to_sin_cos(date):
    date = pd.to_datetime(date)

    day_of_year = date.day_of_year
    custom_period = 365 # This is for 365 days

    sin_date = np.sin(2 * np.pi * day_of_year/ custom_period)
    cos_date = np.cos(2 * np.pi * day_of_year/ custom_period)    

    return sin_date, cos_date

# grab the model
print ("Loading the Model...")
model = jb.load("weather_prediction_v1_model.pkl")
print ("Model Loaded!")

#============INPUT DATE HERE============#
DATE = "2023-01-01"

# Convert the date into day_sin and day_cos
day_sin, day_cos = date_to_sin_cos(DATE)

"""
LIST OF LONGITUDES AND LATITUDES:
Format: Longitude Latitude Area
(13.4443, 144.7937),  # Central Guam
(13.3894, 144.6589),  # Western Guam (Tumon Bay)
(13.4701, 144.7512),  # North (Dededo)
(13.4125, 144.8013),  # Southwest (Agat)
(13.5118, 144.8376),  # Northeast (Yigo)
(13.3387, 144.7345),  # Far Southwest (Umatac)
(13.5762, 144.8551),  # Far North (Ritidian Point)
(13.2310, 144.6923),  # Offshore SW
(13.6325, 144.9124),  # Offshore NE
(13.4443, 144.6500),  # West Coast (Apra Harbor)
(13.4443, 144.9500),  # East Coast (Cocos Island)
(13.3000, 144.7937),  # Southern Offshore
(13.6000, 144.7937),  # Northern Offshore
(13.4443, 144.5000),  # Far West (Philippine Sea)
(13.4443, 145.1000)   # Far East (Pacific Ocean)

GENERAL AREA
Latitude: 13.2°N to 13.7°N
Longitude: 144.5°E to 145.1°E

It may be unstable if we go too far
"""
# long and lat
latitude = 13.6000
longitude = 144.7937

# winspeed and rainfall
windspeed = 20
precipitation = 0
 # in mm

# hour
hour = 14
#======================================#

# Get the temperature prediction
print ("Getting Predicted Temperature...")

# we will do a grid space by using linspace
"""
GENERAL AREA
Latitude: 13.2°N to 13.7°N
Longitude: 144.5°E to 145.1°E
"""
# we will create 15 even squares
list_ = [[13.4443, 144.7937],  # Central Guam
[13.3894, 144.6589],  # Western Guam (Tumon Bay)
[13.4701, 144.7512],  # North (Dededo)
[13.4125, 144.8013],  # Southwest (Agat)
[13.5118, 144.8376],  # Northeast (Yigo)
[13.3387, 144.7345],  # Far Southwest (Umatac)
[13.5762, 144.8551],  # Far North (Ritidian Point)
[13.2310, 144.6923],  # Offshore SW
[13.6325, 144.9124],  # Offshore NE
[13.4443, 144.6500],  # West Coast (Apra Harbor)
[13.4443, 144.9500],  # East Coast (Cocos Island)
[13.3000, 144.7937],  # Southern Offshore
[13.6000, 144.7937],  # Northern Offshore
[13.4443, 144.5000],  # Far West (Philippine Sea)
[13.4443, 145.1000]]  # Far East (Pacific Ocean)


lat, long = 13.4443, 144.7937
import random

# Generate 23 random windspeed values (between 10 and 20)
winspeed = [round(random.uniform(10, 20), 2) for _ in range(24)]

for hour in range(0, 24):
    input_data = [[lat, long, hour, winspeed[hour], 0.0, day_sin, day_cos]]
    predicted_temp = model.predict(input_data)
    print(f"Time: {hour}:00 \t Rainfall:{0.0} \t Windspeed: {winspeed[hour]} \t Temp: {round(predicted_temp[0],2)}˚C")


"""
result = {}
for cord in list_:
    lat = cord[0]
    long = cord[1]
    input_data = [[lat, long, hour, windspeed * rand.random(), precipitation * rand.random(), day_sin, day_cos]]
    predicted_temperature = model.predict(input_data)
    result[(lat, long)] = predicted_temperature[0]

for key in result.keys():
    lat = round(key[0], 3)
    long = round(key[1], 3)
    temp = result.get(key)
    print(f"{lat}˚E, {long}˚N:\t{round(temp, 3)}˚C")
"""
