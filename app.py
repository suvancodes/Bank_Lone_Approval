from flask import Flask, render_template, request, redirect, url_for, session
from flask_cors import CORS
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline
import os

app = Flask(__name__)
CORS(app)   # 🔥 FIX: allows requests from Vercel frontend

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
        payload = request.get_json()

        # Even if backend ignores Bank_Name, include it so frontend does not break.
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
        pred = int(pipeline.predict(df)[0])
        label = "Approved" if pred == 1 else "Rejected"

        # JSON response for JS frontend
        return {
            "success": True,
            "result": {"label": label, "prediction": pred}
        }, 200

    except Exception as e:
        return {"success": False, "error": str(e)}, 500


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
