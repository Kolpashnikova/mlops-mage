import mlflow 

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

from sklearn.metrics import root_mean_squared_error

import pickle

mlflow.set_tracking_uri("sqlite:///mlflow/mlflow.db")

mlflow.set_experiment("mage-experiment")
mlflow.sklearn.autolog()

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data(l):
    X_train = l[0]
    y_train = l[1]
    vectorizer = l[2]
    with mlflow.start_run():

        lr = LinearRegression()
        lr.fit(X_train, y_train)

        y_pred = lr.predict(X_train)

        rmse = root_mean_squared_error(y_train, y_pred)

        mlflow.log_metric("rmse", rmse)

        mlflow.sklearn.log_model(lr, "models_mlflow")

        #with open("preprocessor.b", "wb") as f_out:
        #    pickle.dump(vectorizer, f_out)

        #mlflow.log_artifact("preprocessor.b", artifact_path="preprocessor")

