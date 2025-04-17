from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load the model and encoders
model = joblib.load('service_prediction_model.pkl')
le_last_service = joblib.load('le_last_service.pkl')
le_service_taken = joblib.load('le_service_taken.pkl')

@app.route('/')
def home():
    return jsonify({'message': 'Welcome to the Vehicle Service Prediction API', 'endpoints': ['/predict_new', '/predict_full']})

@app.route('/predict_new', methods=['POST'])
def predict_new():
    try:
        data = request.json
        current_odo = float(data['current_odo'])
        vehicle_age = float(data['vehicle_age'])
        
        input_data = pd.DataFrame({
            'Current_Odometer_km': [current_odo],
            'Last_Service_Odometer_km': [0],
            'Distance_Since_Service_km': [current_odo],
            'Time_Since_Last_Service_Days': [vehicle_age * 365],
            'Last_Service_Type_Encoded': [le_last_service.transform(['Initial'])[0]],
            'Vehicle_Age_Years': [vehicle_age]
        })
        prediction = model.predict(input_data)
        result = le_service_taken.inverse_transform(prediction)[0]
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict_full', methods=['POST'])
def predict_full():
    try:
        data = request.json
        current_odo = float(data['current_odo_full'])
        last_service_odo = float(data['last_service_odo'])
        distance_since = float(data['distance_since'])
        time_since = float(data['time_since'])
        last_service_type = data['last_service_type']
        vehicle_age = float(data['vehicle_age_full'])
        
        last_service_encoded = le_last_service.transform([last_service_type])[0]
        input_data = pd.DataFrame({
            'Current_Odometer_km': [current_odo],
            'Last_Service_Odometer_km': [last_service_odo],
            'Distance_Since_Service_km': [distance_since],
            'Time_Since_Last_Service_Days': [time_since],
            'Last_Service_Type_Encoded': [last_service_encoded],
            'Vehicle_Age_Years': [vehicle_age]
        })
        prediction = model.predict(input_data)
        result = le_service_taken.inverse_transform(prediction)[0]
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
