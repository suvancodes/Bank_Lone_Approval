from flask import Flask, render_template, request, redirect, url_for
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

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

@app.route("/predict", methods=["GET", "POST"])
def predict():
    # if someone opens /predict in browser, send them to landing page
    if request.method == "GET":
        return redirect(url_for("index"))

    try:
        # accept form POST (from home.html) or JSON (AJAX)
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

        # return JSON for AJAX, otherwise render home with result
        if request.is_json:
            return {"success": True, "result": {"label": label, "prediction": pred}}, 200
        return render_template("home.html", result=label)

    except Exception as e:
        if request.is_json:
            return {"success": False, "error": str(e)}, 500
        return render_template("home.html", error=str(e))

if __name__ == "__main__":
    print('server runs in : http://localhost:5000/')
    app.run(debug=True)
    