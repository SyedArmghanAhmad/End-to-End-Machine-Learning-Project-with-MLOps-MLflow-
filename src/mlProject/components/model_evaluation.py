import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
from mlProject.entity.config_entity import ModelEvaluationConfig
from mlProject.utils.common import save_json
from pathlib import Path
from mlflow.models.signature import infer_signature
import dagshub
from urllib.parse import urlparse


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        # Initialize DagsHub and MLflow connection
        self._init_mlflow()

    def _init_mlflow(self):
        """Initialize DagsHub and MLflow connection with error handling"""
        try:
            # Initialize DagsHub connection
            dagshub.init(
                repo_owner='SyedArmghanAhmad',
                repo_name='End-to-End-Machine-Learning-Project-with-MLOps-MLflow-',
                mlflow=True
            )
            
            # Set MLflow tracking URI
            mlflow.set_tracking_uri(self.config.mlflow_uri)
            mlflow.set_registry_uri(self.config.mlflow_uri)
            
            print(f"Successfully connected to DagsHub MLflow tracking at: {self.config.mlflow_uri}")
        except Exception as e:
            print(f"Error connecting to DagsHub MLflow: {str(e)}")
            # Fallback to local MLflow tracking
            mlflow.set_tracking_uri("file:///./mlruns")
            mlflow.set_registry_uri(None)
            print("Falling back to local MLflow tracking")

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def log_into_mlflow(self):
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]

        try:
            # Create model signature and example
            predictions = model.predict(test_x)
            signature = infer_signature(test_x, predictions)
            input_example = test_x.iloc[0:1]

            with mlflow.start_run():
                # Calculate and log metrics
                (rmse, mae, r2) = self.eval_metrics(test_y, predictions)
                scores = {"rmse": rmse, "mae": mae, "r2": r2}
                save_json(path=Path(self.config.metric_file_name), data=scores)

                mlflow.log_params(self.config.all_params)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2", r2)
                mlflow.log_metric("mae", mae)

                # Determine if we can register the model
                tracking_uri = mlflow.get_tracking_uri()
                can_register = urlparse(tracking_uri).scheme in ['http', 'https']

                # Log model with conditional registration
                if can_register:
                    try:
                        mlflow.sklearn.log_model(
                            sk_model=model,
                            artifact_path="model",
                            registered_model_name="ElasticnetModel",
                            signature=signature,
                            input_example=input_example
                        )
                        print("Model successfully registered in MLflow Model Registry")
                    except Exception as e:
                        print(f"Model registration failed: {str(e)}")
                        print("Logging model without registration")
                        mlflow.sklearn.log_model(
                            sk_model=model,
                            artifact_path="model",
                            signature=signature,
                            input_example=input_example
                        )
                else:
                    mlflow.sklearn.log_model(
                        sk_model=model,
                        artifact_path="model",
                        signature=signature,
                        input_example=input_example
                    )
                    print("Model logged locally (remote registry not available)")

        except Exception as e:
            print(f"MLflow operation failed: {str(e)}")
            # Fallback: Save metrics locally
            predictions = model.predict(test_x)
            (rmse, mae, r2) = self.eval_metrics(test_y, predictions)
            scores = {"rmse": rmse, "mae": mae, "r2": r2}
            save_json(path=Path(self.config.metric_file_name), data=scores)
            print("Metrics saved locally due to MLflow failure")