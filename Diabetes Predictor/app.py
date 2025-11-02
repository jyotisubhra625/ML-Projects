from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model and the scaler
model = load_model('diabetes_model.h5')
with open('diabetes.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for making a prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    input_features = [float(x) for x in request.form.values()]
    
    # Reshape and scale the input features
    final_features = np.array(input_features).reshape(1, -1)
    scaled_features = scaler.transform(final_features)

    # Make the prediction and get the raw score
    prediction_score = model.predict(scaled_features)[0][0]

    # NEW LOGIC: Define the output based on the score
    if prediction_score > 0.60:
        output = "Diabetic"
    elif prediction_score < 0.40:
        output = "Not Diabetic"
    else:
        output = "Borderline"

    # Format the score to be a percentage
    prediction_text = f"The person is likely: {output}"
    score_text = f"(Model Confidence Score: {prediction_score*100:.2f}%)"

    # Render the result on the HTML page
    return render_template('index.html', prediction_text=prediction_text, score_text=score_text, result_class=output.lower().replace(" ", "-"))


# Run the application
if __name__ == "__main__":
    app.run(debug=True)