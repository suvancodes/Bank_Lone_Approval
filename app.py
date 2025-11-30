from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_cors import CORS
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline
import os
import math

app = Flask(__name__)
CORS(app)   # allow requests from frontend
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret")


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/home", methods=["GET"])
def home():
    return render_template("home.html")


def _to_number(v, cast=int, default=0):
    try:
        return cast(float(v))
    except Exception:
        return default


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)

        # optional bank_name (frontend sends it)
        bank_name = payload.get("Bank_Name", "Unknown")

        data = CustomData(
            Gender=_to_number(payload.get("Gender"), int),
            Married=_to_number(payload.get("Married"), int),
            Education=_to_number(payload.get("Education"), int),
            Self_Employed=_to_number(payload.get("Self_Employed"), int),
            ApplicantIncome=_to_number(payload.get("ApplicantIncome"), float),
            CoapplicantIncome=_to_number(payload.get("CoapplicantIncome"), float),
            LoanAmount=_to_number(payload.get("LoanAmount"), float),
            Loan_Amount_Term=_to_number(payload.get("Loan_Amount_Term"), float),
            Credit_History=_to_number(payload.get("Credit_History"), float),
        )

        df = data.get_data_as_dataframe()
        pipeline = PredictPipeline()

        # get predicted class
        preds = pipeline.predict(df)
        pred = int(preds[0]) if hasattr(preds, "__len__") else int(preds)
        label = "Approved" if pred == 1 else "Rejected"

        # try to get probabilities
        approval_percent = None
        reject_percent = None
        try:
            # some pipelines expose predict_proba
            probas = pipeline.predict_proba(df)
            # ensure shape and values
            if probas is not None and len(probas) > 0:
                # probability for class 1 (approved)
                approval_prob = float(probas[0][1])
                reject_prob = float(probas[0][0])
                approval_percent = round(approval_prob * 100, 2)
                reject_percent = round(reject_prob * 100, 2)
        except Exception:
            # predict_proba not available or failed - fallback below
            approval_percent = None

        # fallback if no probabilities from model
        if approval_percent is None:
            if pred == 1:
                approval_percent = 95.0
                reject_percent = 5.0
            else:
                approval_percent = 4.0
                reject_percent = 96.0

        result = {
            "label": label,
            "prediction": pred,
            "approval_percent": approval_percent,
            "reject_percent": reject_percent,
            "bank": bank_name
        }

        return jsonify({"success": True, "result": result}), 200

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/result")
def result():
    label = session.pop('last_result', None)
    if not label:
        return redirect(url_for("home"))
    return render_template("result.html", result=label)


if __name__ == "__main__":
    print("server runs at http://localhost:5000/")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
