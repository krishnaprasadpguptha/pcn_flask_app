from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import io
import base64

# Initialize Flask app
app = Flask(__name__)

# Load trained model and preprocessing objects
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))

data_file = os.path.join(os.getcwd(), "data", "Parking_Services_Penalty_Charge_Notices_2019-20_20250205_predicted.csv")
df = pd.read_csv(data_file) if os.path.exists(data_file) else pd.DataFrame()
column_mapping = {
    "Vehicle Type": "VehicleType",
    "Contravention Code": "ContraventionCode",
    "Location": "Location",
    "predicted Apeal": "AppealOutcome"
}
df.rename(columns=column_mapping, inplace=True)
# Home Page
@app.route("/")
def home():
    return render_template("index.html")

# Prediction Page
@app.route("/predict", methods=["POST"])
def predict():
    try:
        vehicleType = request.form["vehicleType"]
        contraventionCode = request.form["contraventionCode"]
        location = request.form["location"]
        
        vehicleType = label_encoders["VehicleType"].transform([vehicleType])[0]
        contraventionCode = label_encoders["ContraventionCode"].transform([contraventionCode])[0]
        location = label_encoders["Location"].transform([location])[0]

        input_data = np.array([[vehicleType, contraventionCode, location]])
        input_data = scaler.transform(input_data)

        prediction = model.predict(input_data)[0]
        result = "Appealed" if prediction == 1 else "No Appeal"
        
        return render_template("result.html", prediction=result)
    except Exception as e:
        return f"Error: {str(e)}"

# View Appeals Page
@app.route("/view_appeals")
def view_appeals():
    if df.empty:
        return "No appeal records found."
    return render_template("view_appeals.html", appeals=df.to_dict(orient='records'))

# Dashboard Page
@app.route("/dashboard")
def dashboard():
    if df.empty:
        return "No data available."
    
    # Generate visualization
    img = io.BytesIO()
    plt.figure(figsize=(8,5))
    print(df.columns)
    df["AppealOutcome"].value_counts().plot(kind="bar", color=["green", "red"])
    plt.title("Appeal Outcomes Distribution")
    plt.xlabel("Outcome")
    plt.ylabel("Count")
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    
    return render_template("dashboard.html", plot_url=plot_url)

if __name__ == "__main__":
    app.run(debug=True)
