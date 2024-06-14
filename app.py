import threading
import time
import os
import pyrebase
import numpy as np
import tflite_runtime.interpreter as tflite
from flask import Flask, render_template, jsonify
import joblib

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET')

firebaseConfig = {
    "apiKey": "AIzaSyBjDArp_CvaEjvELFQWd_S1N7dSJW6Kz0o",
    "authDomain": "data-5647b.firebaseapp.com",
    "databaseURL": "https://data-5647b-default-rtdb.europe-west1.firebasedatabase.app",
    "projectId": "data-5647b",
    "storageBucket": "data-5647b.appspot.com",
    "messagingSenderId": "1068830233307",
    "appId": "1:1068830233307:web:02a0f8d39e0cd6cb4b32fe",
    "measurementId": "G-EJ0HL4XT0R"
}

firebase = pyrebase.initialize_app(firebaseConfig)
db = firebase.database()

# Charger le modèle TensorFlow Lite
interpreter = tflite.Interpreter(model_path="model_GRU_3_ras.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Charger le scaler sauvegardé
scaler = joblib.load('scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data')
def get_data():
    try:
        # Récupérer les données de la base de données Firebase
        data = db.get().val()
        
        # Vérifier et récupérer les valeurs
        courant = data.get('Courant')
        tension = data.get('Tension')
        temperature = data.get('Temperature')
        # Préparer les données d'entrée et les normaliser
        input_data = np.array([[tension, courant, temperature]], dtype=np.float32)
        input_data_normalized = scaler.transform(input_data)
        
        # Exécuter le modèle TFLite pour obtenir la prédiction
        interpreter.set_tensor(input_details[0]['index'], input_data_normalized)
        interpreter.invoke()
        soc_prediction = interpreter.get_tensor(output_details[0]['index'])[0]
        
        # Retourner les données et la prédiction sous forme de réponse JSON
        return jsonify({
            'Courant': courant,
            'Tension': tension,
            'Temperature': temperature,
            'SOC_Prediction': soc_prediction.tolist()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
