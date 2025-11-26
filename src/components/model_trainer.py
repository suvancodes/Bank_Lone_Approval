import os
import sys
from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from src.exception import CustomException
from src.logger import logging
from src.utlit import save_obj, evalute_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifact", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Splitting training and test input data")
            
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )
            
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                "SVM": SVC(kernel='rbf', random_state=42)
            }
            
            model_report = evalute_model(X_train, y_train, X_test, y_test, models)
            
            logging.info(f"Model Report: {model_report}")
            
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]
            
            logging.info(f"Best Model: {best_model_name}, Accuracy: {best_model_score}")
            
            if best_model_score < 0.5:
                logging.warning("Best model accuracy is below 0.6, but continuing anyway...")

            
            logging.info("Saving best model...")
            save_obj(self.model_trainer_config.trained_model_file_path, best_model)
            
            y_pred = best_model.predict(X_test)
            
            logging.info(f"Accuracy: {accuracy_score(y_test, y_pred)}")
            logging.info(f"Precision: {precision_score(y_test, y_pred)}")
            logging.info(f"Recall: {recall_score(y_test, y_pred)}")
            logging.info(f"F1 Score: {f1_score(y_test, y_pred)}")
            logging.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
            logging.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
            
            return self.model_trainer_config.trained_model_file_path
        
        except Exception as e:
            logging.error("Error in model training", exc_info=True)
            raise CustomException(e, sys)
