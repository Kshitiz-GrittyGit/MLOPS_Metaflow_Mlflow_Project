from pathlib import Path
import pandas as pd
from metaflow import FlowSpec,step, IncludeFile, Parameter,current
from common import DatasetMixin, build_feature_transformer, build_target_transformer, build_model
from sklearn.model_selection import KFold
import mlflow
import os

class Training(FlowSpec, DatasetMixin):

    mlflow_tracking_uri = Parameter(
        'mlflow_tracking_uri',
        help = 'mlflow runing server',
        default = os.getenv('MLFLOW_TRACKING_URI', 'https://127.0.0.1:5001')
    )


    @step
    def start(self):
        print('The Pipeline is Starting')

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        run = mlflow.start_run(run_name=current.run_id)
        self.mlflow_run_id = run.info.run_id

        self.data = self.load_dataset()
        self.next(self.cross_validation)

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
        feature_transformer = build_feature_transformer()
        self.x_train = feature_transformer.fit_transform(self.train_data)
        self.x_test = feature_transformer.transform(self.test_data)

        """Now let's transform target data"""

        target_transformer = build_target_transformer()
        self.y_train = target_transformer.fit_transform(self.train_data)
        self.y_test = target_transformer.transform(self.test_data)

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
            
            self.mlflow_fold_run_id = run.info.run_id
            self.model = build_model(self.x_train, self.x_test, self.y_train)
            
        self.next(self.evaluate_fold)

    @step
    def evaluate_fold(self):

        import mlflow

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        from sklearn.metrics import accuracy_score

        self.accuracy = accuracy_score(self.y_test, self.model)

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

        self.next(self.end)
        
    @step
    def end(self):
        print('This is the end of the Pipeline')

if __name__ == "__main__":
    Training()