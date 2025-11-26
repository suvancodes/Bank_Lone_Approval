import pandas as pd
import sys
import os
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging
from src.components.data_tranceformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifact", "train.csv")
    test_data_path: str = os.path.join("artifact", "test.csv")
    raw_data_path: str = os.path.join("artifact", "raw.csv")

class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        try:
            logging.info("Reading dataset")
            df = pd.read_csv("notebook/clean_data.csv")

            os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)
            df.to_csv(self.config.raw_data_path, index=False)

            train, test = train_test_split(df, test_size=0.2, random_state=42)

            train.to_csv(self.config.train_data_path, index=False)
            test.to_csv(self.config.test_data_path, index=False)

            logging.info("Data ingestion completed")

            return self.config.train_data_path, self.config.test_data_path
        
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    ingestion = DataIngestion()
    train_data, test_data = ingestion.initiate_data_ingestion()

    transformer = DataTransformation()
    train_arr, test_arr, _ = transformer.start_data_transformation(train_data, test_data)

    trainer = ModelTrainer()
    trainer.initiate_model_trainer(train_arr, test_arr)

    print("Model training completed!")
