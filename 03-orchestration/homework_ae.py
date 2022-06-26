from urllib.parse import non_hierarchical
import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from prefect import flow, task, get_run_logger
from prefect.task_runners import SequentialTaskRunner

from datetime import date, datetime
from dateutil.relativedelta import relativedelta

import pickle

@task
def read_data(path):
    logger = get_run_logger()
    
    df = pd.read_parquet(path)
    
    return df

@task
def prepare_features(df, categorical, train=True):
    logger = get_run_logger()
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical):
    logger = get_run_logger()
    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    get_run_logger(f"The MSE of training is: {mse}")
    return lr, dv

@task
def run_model(df, categorical, dv, lr):
    logger = get_run_logger()
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")
    return

def get_paths(input_date=None):
    logger = get_run_logger()
    if input_date == None or input_date == '':
        input_date = date.today()
    else:
        input_date =  datetime.strptime(input_date, '%Y-%m-%d')
    
    train_date = input_date - relativedelta(months=2)
    val_date = input_date - relativedelta(months=1)
    
    train_path = f'./data/fhv_tripdata_{train_date.strftime("%Y-%m")}.parquet'
    val_path = f'./data/fhv_tripdata_{val_date.strftime("%Y-%m")}.parquet'
    logger.info(train_path, val_path)
    
    return train_path,val_path

@flow(task_runner=SequentialTaskRunner)
def main(date="2021-08-15"):
    logger = get_run_logger()

    train_path, val_path = get_paths(date)
    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical).result()

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False).result()

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    run_model(df_val_processed, categorical, dv, lr)

    with open(f"/home/ubuntu/Github/mlops-zoomcamp/03-orchestration/model/model-{date}.bin", 'wb') as m_out:
        pickle.dump((dv, lr), m_out)
    
    with open(f"/home/ubuntu/Github/mlops-zoomcamp/03-orchestration/model/dv-{date}.b", 'wb') as d_out:
        pickle.dump((dv), d_out)

main()
