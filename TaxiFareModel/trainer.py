from google.cloud import storage
import joblib
from mlflow import sklearn
from mlflow.sklearn import save_model
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR

from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data, clean_data

import mlflow
from mlflow.tracking import MlflowClient
from memoized_property import memoized_property
import importlib
import TaxiFareModel.gcparams as gcparams




class Trainer():

    MLFLOW_URI = "https://mlflow.lewagon.co/"
    #experiment_name = "[DE] [Munich] [eloisahernandez] TaxFareModel"

    def __init__(self, X, y, **kwargs):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.__dict__.update(**kwargs)

    def set_pipeline(self, **kwargs):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([('dist_trans', DistanceTransformer()),
                              ('stdscaler', StandardScaler())])
        time_pipe = Pipeline([('time_enc',
                               TimeFeaturesEncoder('pickup_datetime')),
                              ('ohe', OneHotEncoder(handle_unknown='ignore'))])
        preproc_pipe = ColumnTransformer([('distance', dist_pipe, [
            "pickup_latitude", "pickup_longitude", 'dropoff_latitude',
            'dropoff_longitude'
        ]), ('time', time_pipe, ['pickup_datetime'])],
                                         remainder="drop")
        model_module = importlib.import_module(kwargs['model_module'])
        model_class_ = getattr(model_module, kwargs['estimator'])
        model_instance = model_class_()
        self.pipeline = Pipeline([('preproc', preproc_pipe),
                                  (kwargs['estimator'], model_instance)])
        self.mlflow_log_param('model', kwargs['estimator'])


    def run(self, **kwargs):
        """set and train the pipeline"""
        self.set_pipeline(**kwargs)
        self.pipeline.fit(self.X, self.y)


    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        self.mlflow_log_metric('rmse', rmse)
        return rmse

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(self.MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(
                self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


    def upload_model_to_gcp(self):
        client = storage.Client()
        bucket = client.bucket(gcparams.BUCKET_NAME)
        blob = bucket.blob(gcparams.STORAGE_LOCATION)
        blob.upload_from_filename('model.joblib')

    def save_model(self):
        """method that saves the model into a .joblib file and uploads it on Google Storage /models folder"""
        joblib.dump(self.pipeline, 'model.joblib')
        print("saved model.joblib locally")

        self.upload_model_to_gcp()
        print(
        f"uploaded model.joblib to gcp cloud storage under \n => {gcparams.STORAGE_LOCATION}"
        )



if __name__ == "__main__":

    params = dict(
        nrows=100_000,  # number of samples
        local=False,  # get data from AWS
        optimize=True,
        estimator="LinearSVR",
        model_module = 'sklearn.svm',
        mlflow=True,  # set to True to log params to mlflow
        experiment_name="[DE] [Munich] [eloisahernandez] TaxFareModel",
        pipeline_memory=None,
        distance_type="manhattan",
        feateng=[
            "distance_to_center", "direction", "distance", "time_features",
            "geohash"
        ])
    # get data
    df = get_data(**params)
    # clean data
    df = clean_data(df)
    # set X and y
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)
    # hold out
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)
    # train
    trainer = Trainer(X_train, y_train, **params)
    trainer.run(**params)
    # evaluate
    print(trainer.evaluate(X_val, y_val))
    trainer.save_model()
