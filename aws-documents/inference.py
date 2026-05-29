import json
import numpy as np

def predict(data):
    input_array = np.array(data).reshape(1, -1)

    prob = model.predict_proba(input_array)[0][1]

    risk = "LOW"
    if prob > 0.7:
        risk = "HIGH"
    elif prob > 0.4:
        risk = "MEDIUM"

    return {
        "attrition_probability": round(prob * 100, 2),
        "risk_level": risk
    }