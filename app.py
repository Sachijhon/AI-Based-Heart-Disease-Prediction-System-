
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "rf_heart_model.pkl"

# Load either a saved model directly or a dictionary containing the model.
saved_data = joblib.load(MODEL_PATH)

if isinstance(saved_data, dict):
    MODEL = saved_data.get("model", saved_data.get("MODEL"))
    FEATURE_NAMES = saved_data.get(
        "feature_names",
        saved_data.get("FEATURE_NAMES")
    )
else:
    MODEL = saved_data
    FEATURE_NAMES = None

if MODEL is None or not hasattr(MODEL, "predict_proba"):
    raise ValueError(
        "The saved artifact must contain a trained classifier "
        "supporting predict_proba()."
    )

if FEATURE_NAMES is None:
    FEATURE_NAMES = list(MODEL.feature_names_in_)

# The model must have been trained with these feature names and order.
EXPECTED_FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal"
]

if list(FEATURE_NAMES) != EXPECTED_FEATURES:
    raise ValueError(
        "Feature names or order do not match the expected model inputs. "
        f"Saved: {list(FEATURE_NAMES)}"
    )

CLASS_LABELS = list(MODEL.classes_)

if 0 not in CLASS_LABELS or 1 not in CLASS_LABELS:
    raise ValueError(
        "Expected model classes 0 (No Disease) and 1 (Heart Disease)."
    )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read numeric inputs and validate the form.
        values = {
            "age": int(request.form["age"]),
            "sex": int(request.form["sex"]),
            "cp": int(request.form["cp"]),
            "trestbps": int(request.form["trestbps"]),
            "chol": int(request.form["chol"]),
            "fbs": int(request.form["fbs"]),
            "restecg": int(request.form["restecg"]),
            "thalach": int(request.form["thalach"]),
            "exang": int(request.form["exang"]),
            "oldpeak": float(request.form["oldpeak"]),
            "slope": int(request.form["slope"]),
            "ca": int(request.form["ca"]),
            "thal": int(request.form["thal"]),
        }

        if not 1 <= values["age"] <= 120:
            raise ValueError("Age must be between 1 and 120.")
        if not 70 <= values["trestbps"] <= 250:
            raise ValueError("Check the resting blood pressure value.")
        if not 80 <= values["chol"] <= 700:
            raise ValueError("Check the cholesterol value.")
        if not 50 <= values["thalach"] <= 250:
            raise ValueError("Check the maximum heart rate value.")
        if not 0 <= values["oldpeak"] <= 10:
            raise ValueError("Oldpeak must be between 0 and 10.")

        # Keep the exact feature order used during training.
        input_df = pd.DataFrame(
            [[values[name] for name in FEATURE_NAMES]],
            columns=FEATURE_NAMES
        )

        probabilities = MODEL.predict_proba(input_df)[0]
        class_probabilities = dict(zip(CLASS_LABELS, probabilities))

        no_disease_probability = float(class_probabilities[0]) * 100
        heart_disease_probability = float(class_probabilities[1]) * 100

        prediction = int(
            MODEL.predict(input_df)[0]
        )

        return render_template(
            "result.html",
            heart_probability=round(heart_disease_probability, 2),
            no_disease_probability=round(no_disease_probability, 2),
            prediction=prediction,
            inputs=values
        )

    except (ValueError, KeyError, TypeError) as error:
        return render_template(
            "index.html",
            error=f"Please check your inputs: {error}"
        ), 400


if __name__ == "__main__":
    app.run(debug=True)

