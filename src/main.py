from flask import Flask, request, jsonify, render_template
try:
    import tensorflow as tf
except Exception:
    tf = None
import numpy as np
import os
import joblib

app = Flask(__name__, static_folder='statics')

# URL of your Flask API for making predictions
api_url = 'http://0.0.0.0:4000/predict'  # Update with the actual URL

# Load the TensorFlow model
model = None
try:
    model = tf.keras.models.load_model('my_model.keras')  # Replace 'my_model.keras' with the actual model file
except Exception:
    model = None

class_labels = ['Setosa', 'Versicolor', 'Virginica']

# Try to load RandomForest model and scaler if present
rf_model = None
scaler = None
if os.path.exists('rf_model.joblib') and os.path.exists('scaler.joblib'):
    try:
        rf_model = joblib.load('rf_model.joblib')
        scaler = joblib.load('scaler.joblib')
    except Exception:
        rf_model = None
        scaler = None


@app.route('/')
def home():
    return "Welcome to the Iris Classifier API!"

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            data = request.form
            sepal_length = float(data['sepal_length'])
            sepal_width = float(data['sepal_width'])
            petal_length = float(data['petal_length'])
            petal_width = float(data['petal_width'])

            # Perform the prediction (TensorFlow model expects scaled inputs if training used scaling)
            input_data = np.array([sepal_length, sepal_width, petal_length, petal_width])[np.newaxis, ]
            if model is None:
                return jsonify({"error": "TensorFlow model not available on server."}), 503
            prediction = model.predict(input_data)
            predicted_class = class_labels[np.argmax(prediction)]

            # Return the predicted class in the response
            # Use jsonify() instead of json.dumps() in Flask
            return jsonify({"predicted_class": predicted_class})
        except Exception as e:
            return jsonify({"error": str(e)})
    elif request.method == 'GET':
        return render_template('predict.html')
    else:
        return "Unsupported HTTP method"


@app.route('/predict_rf', methods=['GET', 'POST'])
def predict_rf():
    """Predict using the RandomForest model (if available).
    Accepts form-encoded or JSON POST with the same four features.
    """
    if rf_model is None or scaler is None:
        return jsonify({"error": "RandomForest model or scaler not available on server."}), 503

    if request.method == 'POST':
        try:
            data = request.get_json() if request.is_json else request.form
            sepal_length = float(data['sepal_length'])
            sepal_width = float(data['sepal_width'])
            petal_length = float(data['petal_length'])
            petal_width = float(data['petal_width'])

            input_data = np.array([sepal_length, sepal_width, petal_length, petal_width])[np.newaxis, ]
            # Apply scaler used during training
            input_data = scaler.transform(input_data)
            prediction = rf_model.predict(input_data)
            predicted_class = class_labels[int(prediction[0])]

            return jsonify({"predicted_class": predicted_class})
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    elif request.method == 'GET':
        return render_template('predict_rf.html')
    else:
        return "Unsupported HTTP method"

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=4000)
