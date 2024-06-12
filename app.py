import threading
import time
import os
import pyrebase
import numpy as np
import onnxruntime as ort
from flask import Flask, render_template
from flask_socketio import SocketIO

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET')
app.config['databaseURL]= os.environ.get('')
socketio = SocketIO(app)

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

# Load the ONNX model
ort_session = ort.InferenceSession("model_GRU_3.onnx")

def load_data_and_predict():
    while True:
        try:
            data = db.get().val()
            if isinstance(data, dict):
                courant = data.get('Courant', 0)
                tension = data.get('Tension', 0)
                temperature = data.get('Temperature', 0)
                input_data = np.array([[courant, tension, temperature]], dtype=np.float32)
                
                # Run the ONNX model
                ort_inputs = {ort_session.get_inputs()[0].name: input_data}
                ort_outs = ort_session.run(None, ort_inputs)
                soc = ort_outs[0][0]
                
                # Emit the SocketIO event with the prediction
                socketio.emit('prediction', {'SOC': soc.tolist()})
            else:
                print("Les données récupérées ne sont pas au format attendu :", data)
        except Exception as e:
            print("Erreur lors de la récupération des données ou de la prédiction :", str(e))
        
        time.sleep(5)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    threading.Thread(target=load_data_and_predict).start()
    socketio.run(app, debug=True)

