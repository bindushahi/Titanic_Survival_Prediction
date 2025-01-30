# Import necessary libraries
from flask import Flask, request, jsonify
import joblib
from pyngrok import ngrok
import os

# Load the trained model
model = joblib.load("logistic_regression_model.pkl")

# Initialize Flask app
app = Flask(__name__)

# HTML template for the form
HTML_FORM = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Titanic Survival Prediction</title>
</head>
<body>
    <h2>Titanic Survival Prediction</h2>
    <form id="predictionForm">
        <label for="pclass">Pclass:</label>
        <input type="text" id="pclass" name="pclass"><br><br>

        <label for="sex">Sex (0 for male, 1 for female):</label>
        <input type="text" id="sex" name="sex"><br><br>

        <label for="age">Age:</label>
        <input type="text" id="age" name="age"><br><br>

        <label for="sibsp">SibSp:</label>
        <input type="text" id="sibsp" name="sibsp"><br><br>

        <label for="parch">Parch:</label>
        <input type="text" id="parch" name="parch"><br><br>

        <label for="fare">Fare:</label>
        <input type="text" id="fare" name="fare"><br><br>

        <label for="embarked">Embarked (0 for S, 1 for C, 2 for Q):</label>
        <input type="text" id="embarked" name="embarked"><br><br>

        <button type="button" onclick="predictSurvival()">Predict</button>
    </form>

    <p id="predictionResult"></p>

    <script>
        function predictSurvival() {
            var xhr = new XMLHttpRequest();
            var url = "/predict";
            var data = new FormData(document.getElementById("predictionForm"));

            xhr.open("POST", url, true);
            xhr.onreadystatechange = function () {
                if (xhr.readyState === 4 && xhr.status === 200) {
                    var response = JSON.parse(xhr.responseText);
                    document.getElementById("predictionResult").innerHTML = "Survival Prediction: " + response.prediction;
                }
            };
            xhr.send(data);
        }
    </script>
</body>
</html>
"""

# Home route to display the form
@app.route("/")
def home():
    return HTML_FORM

# Prediction API
@app.route("/predict", methods=["POST"])
def predict():
    # Extract form data
    pclass = int(request.form["pclass"])
    sex = int(request.form["sex"])
    age = float(request.form["age"])
    sibsp = int(request.form["sibsp"])
    parch = int(request.form["parch"])
    fare = float(request.form["fare"])
    embarked = int(request.form["embarked"])

    # Prepare features for prediction
    features = [[pclass, sex, age, sibsp, parch, fare, embarked]]
    prediction = model.predict(features)[0]  # Model prediction

    return jsonify({"prediction": int(prediction)})

# Function to run Flask and start ngrok tunnel
def run_flask_app():
    # Start Flask app on port 5000
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)

# Run Flask and expose it using ngrok
if __name__ == "__main__":
    # Set your ngrok auth token here
    NGROK_AUTH_TOKEN = "YOUR_NGROK_AUTH_TOKEN"
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)

    # Start ngrok tunnel
    public_url = ngrok.connect(5000).public_url
    print("Public URL:", public_url)

    # Open the URL for quick access
    os.system(f"start {public_url}")  # For Windows
    # os.system(f"xdg-open {public_url}")  # For Linux
    # os.system(f"open {public_url}")  # For macOS

    # Run Flask app
    run_flask_app()
