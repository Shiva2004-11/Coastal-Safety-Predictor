import pandas as pd
import streamlit as st
import nexmo
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import warnings

warnings.filterwarnings('ignore')

nexmo_api_key = 'b99f9d73'  
nexmo_api_secret = 'K8rYqIxCzeJguDLy'  
nexmo_phone_number = '+919360757708' 
client = nexmo.Client(key=nexmo_api_key, secret=nexmo_api_secret)

dataset_path = "C:/VIT Hackathon/__pycache__/MOCK_DATA.csv"
dataset = pd.read_csv(dataset_path)
dataset['DATE'] = pd.to_datetime(dataset['DATE'], errors='coerce')
dataset.set_index('DATE', inplace=True)

def determine_safety(wave_height, wind_speed, precipitation, sea_level):
    if (wind_speed > 20 or precipitation > 6 or sea_level > 1.1):
        return "NOT SAFE"
    return "SAFE"

def send_emergency_sms(city):
    phone_numbers = [
        '+919489766467',
        '+8428121935'  
    ]
    message = f"Emergency Alert: Flood warning for {city}. Please take necessary precautions."

    for number in phone_numbers:
        try:
            response = client.send_message({
                'from': nexmo_phone_number,
                'to': number,
                'text': message,
            })
            print(f"Message sent to {number}")
        except Exception as e:
            print(f"Failed to send message to {number}: {e}")

class CoastalSafetyApp:
    def __init__(self):
        self.dataset_path = dataset_path
        self.dataset = None
        self.model = None
        self.scaler = None

    def load_dataset(self):
        try:
            self.dataset = pd.read_csv(self.dataset_path)
            st.write("Dataset loaded successfully.")
            st.write("First few rows of the dataset:")
            st.dataframe(self.dataset.head())

            if 'DATE' in self.dataset.columns:
                self.dataset['DATE'] = pd.to_datetime(self.dataset['DATE'], errors='coerce')
                self.dataset.set_index('DATE', inplace=True)
            else:
                raise ValueError("Dataset does not contain 'DATE' column.")

            required_columns = ['WAVE HEIGHT', 'WIND SPEED', 'PRECIPITATION', 'COASTAL VEGETATION',
                                'HUMAN ACTIVITIES', 'EROSION RATE', 'SEA LEVEL', 'TARGET VARIBLE']
            for col in required_columns:
                if col not in self.dataset.columns:
                    raise ValueError(f"Dataset is missing required column: {col}")

            self.train_model()
        except Exception as e:
            st.error(f"Failed to load dataset or train model: {e}")

    def train_model(self):
        if self.dataset is None:
            st.error("Dataset not loaded. Please load the dataset first.")
            return

        X = self.dataset[['WAVE HEIGHT', 'WIND SPEED', 'PRECIPITATION', 'COASTAL VEGETATION',
                          'HUMAN ACTIVITIES', 'EROSION RATE', 'SEA LEVEL']]
        y = self.dataset['TARGET VARIBLE'].apply(lambda x: 1 if x == 'SAFE' else 0)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        self.model = BernoulliNB()
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        support = recall_score(y_test, y_pred, average=None)

        st.write("Model Evaluation Metrics:")
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write(f"Precision: {precision:.2f}")
        st.write(f"Recall: {recall:.2f}")
        st.write(f"F1 Score: {f1:.2f}")
        st.write(f"Support: {support}")

    def predict_safety(self, wave_height, wind_speed, precipitation, coastal_vegetation, human_activities, erosion_rate, sea_level):
        if self.dataset is None or self.scaler is None or self.model is None:
            st.warning("Dataset, scaler, or model is not loaded. Please load the dataset and train the model first.")
            return None

        input_data = pd.DataFrame({
            'WAVE HEIGHT': [wave_height],
            'WIND SPEED': [wind_speed],
            'PRECIPITATION': [precipitation],
            'COASTAL VEGETATION': [coastal_vegetation],
            'HUMAN ACTIVITIES': [human_activities],
            'EROSION RATE': [erosion_rate],
            'SEA LEVEL': [sea_level]
        })

        input_data_scaled = self.scaler.transform(input_data)

        self.dataset_scaled = self.scaler.transform(self.dataset[['WAVE HEIGHT', 'WIND SPEED', 'PRECIPITATION',
                                                                 'COASTAL VEGETATION', 'HUMAN ACTIVITIES',
                                                                 'EROSION RATE', 'SEA LEVEL']])

        distances = np.sqrt(((self.dataset_scaled - input_data_scaled) ** 2).sum(axis=1))
        closest_index = np.argmin(distances)

        closest_row = self.dataset.iloc[closest_index]
        safety = closest_row['TARGET VARIBLE']
        return safety

def main():
    st.sidebar.title("Navigation")
    app = CoastalSafetyApp()
    page = st.sidebar.radio("Go to", ["Home", "Predict Safety", "Model Evaluation"])

    if page == "Home":
        st.title("Welcome to the Coastal Safety App")
        st.image(r"C:\VIT Hackathon\Orange Hi Tech Social Banner Template.png", use_column_width=True)
        st.write("""
            This app predicts the safety of coastal areas based on various conditions 
            such as wave height, wind speed, precipitation, and sea level. 
            Use the sidebar to navigate through the app.
        """)

    elif page == "Predict Safety":
        st.title("Predict Coastal Safety")

        city = st.text_input("CITY", "", key="city_input")
        wave_height = st.number_input("WAVE HEIGHT (M)", key="wave_height_input")
        wind_speed = st.number_input("Wind Speed (km/h)", key="wind_speed_input")
        precipitation = st.number_input("Precipitation (mm)", key="precipitation_input")
        coastal_vegetation = st.number_input("Coastal Vegetation (%)", key="coastal_vegetation_input")
        human_activities = st.number_input("Human Activities (index)", key="human_activities_input")
        erosion_rate = st.number_input("Erosion Rate (cm/year)", key="erosion_rate_input")
        sea_level = st.number_input("Sea Level (m)", key="sea_level_input")

        if st.button("Predict Safety"):
            safety = determine_safety(wave_height, wind_speed, precipitation, sea_level)
            result_text = (f"City: {city}\n"
                           f"Wave Height (M): {wave_height:.2f} m\n"
                           f"Wind Speed: {wind_speed:.2f} km/h\n"
                           f"Precipitation: {precipitation:.2f} mm\n"
                           f"Sea Level: {sea_level:.2f} m\n"
                           f"Safety: {safety}")
            st.write(result_text)

            if safety == "NOT SAFE":
                send_emergency_sms(city)

        if st.button("Open NOAA Radar Map"):
            if city:
                url = f"https://www.ncei.noaa.gov/maps/radar/?location={city.replace(' ', '%20')}"
                st.markdown(f"[NOAA Radar Map for {city}]({url})")
            else:
                st.warning("Please enter a city to view on the NOAA Radar Map.")

    elif page == "Model Evaluation":
        st.title("Model Evaluation Metrics")
        app.load_dataset()

if __name__ == "__main__":
    main()
