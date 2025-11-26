import os
import sys
import dill
import pandas as pd
from src.exception import CustomException
from src.logger import logging

def save_obj(file_path: str, obj: object) -> str:
    try:
        dir_path = os.path.dirname(file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as f:
            dill.dump(obj, f)
        abs_path = os.path.abspath(file_path)
        logging.info(f"Saved object to {abs_path}")
        return abs_path
    except Exception as e:
        raise CustomException(e, sys)

def load_obj(file_path: str) -> object:
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        with open(file_path, "rb") as f:
            return dill.load(f)
    except Exception as e:
        # raise a CustomException with clear context
        raise CustomException(f"Failed to load object from {file_path}. Error: {e}", sys)
# backward compatibility for other modules that import `lode_obj`
lode_obj = load_obj

class PredictPipeline:
    def __init__(self, model_path: str = "artifact/model.pkl", preprocessor_path: str = "artifact/preprocessor.pkl"):
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path

    def predict(self, feature: pd.DataFrame):
        try:
            model = load_obj(self.model_path)
            preprocessor = load_obj(self.preprocessor_path)
            data_scaled = preprocessor.transform(feature)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(
        self,
        Gender,
        Married,
        Education,
        Self_Employed,
        ApplicantIncome,
        CoapplicantIncome,
        LoanAmount,
        Loan_Amount_Term,
        Credit_History
    ):
        self.Gender = Gender
        self.Married = Married
        self.Education = Education
        self.Self_Employed = Self_Employed
        self.ApplicantIncome = ApplicantIncome
        self.CoapplicantIncome = CoapplicantIncome
        self.LoanAmount = LoanAmount
        self.Loan_Amount_Term = Loan_Amount_Term
        self.Credit_History = Credit_History
        
    def get_data_as_dataframe(self) -> pd.DataFrame:
        try:
            data = {
                "Gender": [self.Gender],
                "Married": [self.Married],
                "Education": [self.Education],
                "Self_Employed": [self.Self_Employed],
                "ApplicantIncome": [self.ApplicantIncome],
                "CoapplicantIncome": [self.CoapplicantIncome],
                "LoanAmount": [self.LoanAmount],
                "Loan_Amount_Term": [self.Loan_Amount_Term],
                "Credit_History": [self.Credit_History],
            }
            return pd.DataFrame(data)
        except Exception as e:
            raise CustomException(e, sys)