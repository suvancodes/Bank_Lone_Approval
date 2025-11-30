from flask import Flask, render_template, request, redirect, url_for, session
from flask_cors import CORS
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline
import os

app = Flask(__name__)
CORS(app)  # 🔥 THIS ALLOWS REQUESTS FROM VERCEL FRONTEND

app.secret_key = os.environ.get("SECRET_KEY", "dev-secret")  # required for session

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
        payload = request.get_json() if request.is_json else request.form.to_dict()

        data = CustomData(
            Gender=_to_number(payload.get("Gender", 0), int, 0),
            Married=_to_number(payload.get("Married", 0), int, 0),
            Education=_to_number(payload.get("Education", 0), int, 0),
            Self_Employed=_to_number(payload.get("Self_Employed", 0), int, 0),
            ApplicantIncome=_to_number(payload.get("ApplicantIncome", 0), float, 0.0),
            CoapplicantIncome=_to_number(payload.get("CoapplicantIncome", 0), float, 0.0),
            LoanAmount=_to_number(payload.get("LoanAmount", 0), float, 0.0),
            Loan_Amount_Term=_to_number(payload.get("Loan_Amount_Term", 0), float, 0.0),
            Credit_History=_to_number(payload.get("Credit_History", 0), float, 0.0),
        )

        df = data.get_data_as_dataframe()
        pipeline = PredictPipeline()
        preds = pipeline.predict(df)
        pred = int(preds[0]) if hasattr(preds, "__len__") else int(preds)
        label = "Approved" if pred == 1 else "Rejected"

        # ALWAYS RETURN JSON (frontend uses JS fetch)
        return {
            "success": True,
            "result": {"label": label, "prediction": pred}
        }, 200

    except Exception as e:
        return {"success": False, "error": str(e)}, 500


@app.route("/result", methods=["GET"])
def result():
    label = session.pop('last_result', None)
    if not label:
        return redirect(url_for('home'))
    return render_template("result.html", result=label)

if __name__ == "__main__":
    print('server runs in : http://localhost:5000/')
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
