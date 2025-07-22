# Weather Prediction Project
A machine learning project created for my **Organization of Programming Languages** course.  
It predicts temperatures in **Guam** using a Random Forest model built with `scikit-learn`.

The model performs **interpolation with ~90% accuracy**, but its **extrapolation** ability degrades after about one month, leading to **overfitting** and instability.

---

## Project Overview

**Model**: Random Forest Regressor (scikit-learn)  
**Goal**: Predict temperature (°C) based on environmental features

**Input features:**
- Date
- Wind Speed
- Precipitation
- Longitude
- Latitude

**Output:**
- Temperature in Celsius (°C)

---

## Features

- Source code to **combine** datasets 
- Source code to **train** the Random Forest model
- Python Script to **test and evaluate** the model
- Sample dataset (~59MB) covering 5 years of weather in Guam  
  *(original dataset included 20 years; trimmed for performance and github restrictions)*

---

## Data Source

Weather data collected from **[Open-Meteo](https://open-meteo.com)**.

---
