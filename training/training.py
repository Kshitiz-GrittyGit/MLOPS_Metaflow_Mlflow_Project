import pandas as pd
from pathlib import Path
import pandas as pd
from metaflow import FlowSpec,step, IncludeFile, Parameter,current
from common import DatasetMixin, build_feature_transformer, build_target_transformer, build_model
from sklearn.model_selection import KFold
import mlflow
import os
import logging

class Training(FlowSpec, DatasetMixin):

    mlflow_tracking_uri = Parameter(
        'mlflow_tracking_uri',
        help = 'mlflow runing server',
        default = os.getenv('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000')
    )

    accuracy_threshold = Parameter(
        'accuracy_threshold',
        help = 'The threshold above which we will allow the model to register',
        default = 0.9
    )


    @step
    def start(self):
        print('The Pipeline is Starting')

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        run = mlflow.start_run(run_name=current.run_id)
        self.mlflow_run_id = run.info.run_id

        self.data = self.load_dataset()
        print('Data Has been loaded')
        self.next(self.cross_validation, self.transform)

    @step
    def cross_validation(self):

        kfold = KFold(n_splits = 5, shuffle=True)
        self.folds = list(enumerate(kfold.split(self.data)))
        
        self.next(self.transform_folds, foreach = 'folds')

    @step
    def transform_folds(self):
        self.fold_number, (self.train_indices, self.test_indices) = self.input

        self.train_data = self.data.iloc[self.train_indices]
        self.test_data = self.data.iloc[self.test_indices]
        self.feature_transformer = build_feature_transformer()
        self.x_train = self.feature_transformer.fit_transform(self.train_data)
        self.x_test = self.feature_transformer.transform(self.test_data)

        """Now let's transform target data"""

        self.target_transformer = build_target_transformer()
        self.y_train = self.target_transformer.fit_transform(self.train_data)
        self.y_test = self.target_transformer.transform(self.test_data)

        self.next(self.train_fold)

    @step
    def train_fold(self):
        import mlflow
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        with (mlflow.start_run(run_id = self.mlflow_run_id),
              mlflow.start_run(run_name = f'cross validation {self.fold_number}',
              nested = True
              ) as run,
        ):

            mlflow.autolog(log_models=False)
            
            self.mlflow_fold_run_id = run.info.run_id
            self.model = build_model()
            self.model.fit(self.x_train,self.y_train)
            
        self.next(self.evaluate_fold)

    @step
    def evaluate_fold(self):

        import mlflow

        from sklearn.metrics import accuracy_score

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        self.pred = self.model.predict(self.x_test)

        self.accuracy = accuracy_score(self.y_test, self.pred)

        mlflow.log_metrics(
            {'fold_accuracy' : self.accuracy},
            run_id=self.mlflow_fold_run_id
        )
        self.next(self.average_scores)

    @step
    def average_scores(self,inputs):
        import mlflow
        import numpy as np
        """Average score of each individual model"""
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        self.merge_artifacts(inputs, include=['mlflow_run_id'])

        metrics = [i.accuracy for i in inputs]
        self.mean_accuracy = np.mean(metrics)
        self.std_accuracy = np.std(metrics)

        mlflow.log_metrics(
            {
                'mean_accuracy' : self.mean_accuracy,
                'std_accuracy' : self.std_accuracy
            }, run_id= self.mlflow_run_id
        ) 

        self.next(self.register_model)


    @step
    def transform(self):

        self.target_transformer = build_target_transformer()
        self.y = self.target_transformer.fit_transform(self.data[['species']])

        self.feature_transformer = build_feature_transformer()
        self.x = self.feature_transformer.fit_transform(self.data)
        self.next(self.train_model)

    @step
    def train_model(self):

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        with mlflow.start_run(run_id = self.mlflow_run_id):

            mlflow.autolog(log_models = False)
            
            self.model = build_model()
            self.model.fit(self.x,self.y)
            
        self.next(self.register_model)

    @step
    def register_model(self, inputs):
        import mlflow
        import tempfile
        import joblib
        from pathlib import Path
        
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        self.merge_artifacts(inputs)
        
        if self.mean_accuracy > self.accuracy_threshold:
            self.registered = True
            logging.info('Registering model....')
        
            with mlflow.start_run(run_id=self.mlflow_run_id):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_dir = Path(tmp_dir)
        
                    # Save model and transformers
                    model_path = tmp_dir / "model.xgb"
                    features_path = tmp_dir / "features.joblib"
                    target_path = tmp_dir / "target.joblib"
        
                    joblib.dump(self.model, model_path)
                    joblib.dump(self.feature_transformer, features_path)
                    joblib.dump(self.target_transformer, target_path)
        
                    # Log XGBoost model (main one)
                    mlflow.xgboost.log_model(
                        xgb_model=self.model,
                        artifact_path="model",
                        registered_model_name="penguins"
                    )
        
                    # Log artifacts if needed
                    mlflow.log_artifact(features_path.as_posix(), artifact_path="preprocessing")
                    mlflow.log_artifact(target_path.as_posix(), artifact_path="preprocessing")
        
        else:
            self.registered = False
            logging.info(
                "The accuracy of the model (%.2f) is lower than the threshold (%.2f). Skipping registration.",
                self.mean_accuracy,
                self.accuracy_threshold,
            )
        
        self.next(self.end)

    
    @step
    def end(self):
        print('This is the end of the Pipeline')


if __name__ == "__main__":
    Training()
