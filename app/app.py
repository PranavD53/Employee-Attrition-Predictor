from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("../model/random_forest.pkl", "rb"))

# Create Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Employee Attrition Prediction API Running"

@app.route('/predict', methods=['POST'])
def predict():

    data = request.json

    features = np.array([
        data['Age'],
        data['BusinessTravel'],
        data['DailyRate'],
        data['Department'],
        data['DistanceFromHome'],
        data['Education'],
        data['EducationField'],
        data['EmployeeCount'],
        data['EmployeeNumber'],
        data['EnvironmentSatisfaction'],
        data['Gender'],
        data['HourlyRate'],
        data['JobInvolvement'],
        data['JobLevel'],
        data['JobRole'],
        data['JobSatisfaction'],
        data['MaritalStatus'],
        data['MonthlyIncome'],
        data['MonthlyRate'],
        data['NumCompaniesWorked'],
        data['Over18'],
        data['OverTime'],
        data['PercentSalaryHike'],
        data['PerformanceRating'],
        data['RelationshipSatisfaction'],
        data['StandardHours'],
        data['StockOptionLevel'],
        data['TotalWorkingYears'],
        data['TrainingTimesLastYear'],
        data['WorkLifeBalance'],
        data['YearsAtCompany'],
        data['YearsInCurrentRole'],
        data['YearsSinceLastPromotion'],
        data['YearsWithCurrManager']
    ]).reshape(1, -1)
    print("Feature Shape:", features.shape)
    print("Features:", features)
    prediction = model.predict(features)[0]

    probability = model.predict_proba(features)[0][1]

    risk = "LOW"

    if probability > 0.7:
        risk = "HIGH"
    elif probability > 0.4:
        risk = "MEDIUM"

    return jsonify({
        "prediction": int(prediction),
        "attrition_probability": round(probability * 100, 2),
        "risk_level": risk
    })

if __name__ == '__main__':
    app.run(debug=True)