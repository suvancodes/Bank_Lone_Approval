import sys
import os
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from src.exception import CustomException
from src.logger import logging
from src.utlit import save_obj

@dataclass
class DataTransformationConfig:
    preprocessing_obj_file_path: str = os.path.join("artifact", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()
    
    def get_data_transformation_obj(self):
        try:
            numeric_features = ['ApplicantIncome','CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']

            numeric_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            logging.info("Numerical feature pipeline created")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", numeric_pipeline, numeric_features)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def start_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data")

            target_column = "Loan_Status"
            numeric_features = ['ApplicantIncome','CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']

            X_train = train_df[numeric_features]
            y_train = train_df[target_column]

            X_test = test_df[numeric_features]
            y_test = test_df[target_column]

            preprocessor = self.get_data_transformation_obj()

            logging.info("Applying preprocessing...")

            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            train_arr = np.c_[X_train_transformed, np.array(y_train)]
            test_arr = np.c_[X_test_transformed, np.array(y_test)]

            save_obj(self.config.preprocessing_obj_file_path, preprocessor)

            return train_arr, test_arr, self.config.preprocessing_obj_file_path
        
        except Exception as e:
            raise CustomException(e, sys)
