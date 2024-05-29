import os
import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse

from mlproject.entity.config_entity import EvaluationConfig
from mlproject.utils.common import save_json


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    
    def _valid_generator(self):

        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.30
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )


    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

                

    def log_into_mlflow(self):
        try:
            
            os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/prabhav131/MLOps.mlflow"
            os.environ["MLFLOW_TRACKING_USERNAME"] = "prabhav131"
            os.environ["MLFLOW_TRACKING_PASSWORD"] = "7a57c6a16d8bdede4119a82de1fd4e2fb08645e6"
            
            # Set tracking URIs
            mlflow.set_tracking_uri(self.config.mlflow_uri)
            
            # # Set the experiment
            # experiment_name = "MLOps experiement"
            # mlflow.set_experiment(experiment_name)
            
            experiment_name = "VGG16Experiment"

            # Check if the experiment exists
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
            else:
                experiment_id = experiment.experiment_id

            mlflow.set_experiment(experiment_name)
            
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            
            # Start a new MLflow run
            with mlflow.start_run():
                # Log parameters and metrics
                if isinstance(self.config.all_params, dict):
                    mlflow.log_params(self.config.all_params)
                else:
                    raise ValueError("all_params should be a dictionary")
                    
                if len(self.score) == 2:
                    mlflow.log_metrics({"loss": self.score[0], "accuracy": self.score[1]})
                else:
                    raise ValueError("score should contain exactly 2 elements: loss and accuracy")
                
                

        except mlflow.exceptions.MlflowException as e:
            print(f"MLflowException: {e}")
            raise
        except Exception as e:
            print(f"General Exception: {e}")
            raise
                