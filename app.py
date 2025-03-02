from flask import Flask, render_template, request
import numpy as np
import pickle

# Load the saved model and scaler
model = pickle.load(open("model/model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Get user input from the form
        input_data = [float(request.form[key]) for key in request.form.keys()]
        
        # Convert input data into a NumPy array and reshape it
        input_data_as_numpy_array = np.array(input_data).reshape(1, -1)
        
        # Standardize input data using the saved scaler
        std_data = scaler.transform(input_data_as_numpy_array)
        
        # Make prediction
        prediction = model.predict(std_data)[0]
        
        # Return result
        result = "Diabetic" if prediction == 1 else "Non-Diabetic"
        
        return render_template("index.html", prediction_text=f"Result: {result}")

if __name__ == "__main__":
    app.run(debug=True)
