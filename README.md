# Coastal-Safety-Predictor
🌊 Coastal Safety Predictor | Real-time Coastal Risk Monitoring System
The Coastal Safety Predictor is an intelligent application that uses machine learning and real-time environmental data to predict coastal safety levels. It alerts users to potential coastal risks such as erosion and flooding, helping protect lives and property along vulnerable shorelines.

🚀 Project Overview
Project Name: Coastal Safety Predictor
Tech Stack: Python, Streamlit, scikit-learn, Nexmo API (SMS), NOAA Radar Maps
ML Model: Bernoulli Naive Bayes Classifier
Goal: Predict coastal safety conditions (SAFE / NOT SAFE) based on environmental parameters and issue emergency alerts.

📊 Features
🧠 Machine Learning Model to classify coastal areas as SAFE or NOT SAFE
🌍 Real-time Data Integration with satellite and IoT sensors (e.g., wind speed, wave height, soil moisture)
📈 Visual Dashboard with NOAA radar maps and risk indicators
📱 Emergency SMS Alerts via Nexmo API
📋 Model Performance Metrics including Accuracy, Precision, Recall
🔍 User Input Form to simulate and test custom scenarios

🔍 Parameters Tracked:
Wind Speed – Measured in km/h from local weather sensors or APIs
Precipitation – Rainfall intensity data from satellite feeds
Tidal Patterns – Monitored through oceanographic sensor data
Vegetation Cover – Derived from satellite imagery to assess dune and plant protection
Human Activity Index – Estimated based on location usage data
Erosion Rate – Tracked through time-series coastal imagery and sensor data
Sea Level – Monitored via remote sensing and tide gauge data.

📌 Results & Insights
The model accurately predicts coastal conditions based on environmental data
Useful for disaster preparedness teams and local authorities
Helps in early detection of unsafe coastal areas using minimal data inputs
