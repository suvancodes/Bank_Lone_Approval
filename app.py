from flask import Flask, render_template, request, redirect, url_for, session
from flask_cors import CORS
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline
import os

app = Flask(__name__)
CORS(app)
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
    except:
        return default


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json()

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

        # -------- PREDICTION -------- #
        pred = int(pipeline.predict(df)[0])
        label = "Approved" if pred == 1 else "Rejected"

        # -------- PROBABILITY -------- #
        try:
            probs = pipeline.model.predict_proba(df)[0]  # [reject, approve]
            reject_prob = float(probs[0])
            approve_prob = float(probs[1])
        except:
            approve_prob = 1.0 if pred == 1 else 0.0
            reject_prob = 1.0 - approve_prob

        return {
            "success": True,
            "result": {
                "label": label,
                "prediction": pred,
                "approve_prob": approve_prob,
                "reject_prob": reject_prob
            }
        }, 200

    except Exception as e:
        return {"success": False, "error": str(e)}, 500


if __name__ == "__main__":
    print("server running on http://localhost:5000")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
