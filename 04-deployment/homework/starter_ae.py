import pickle
import pandas as pd
import numpy as np
import argparse

def load_model():
    with open('model.bin', 'rb') as f_in:
        (dv, model) = pickle.load(f_in)
        return  (dv, model)

def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df

def predict(year, month):
    df = read_data(f'https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{year:04d}-{month:02d}.parquet')

    (dv, model) = load_model()

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    df_result = pd.DataFrame()
    df_result['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
   
    df_result.to_parquet(
        f'df_results{year:04d}-{month:02d}.parquet',
        engine='pyarrow',
        compression=None,
        index=False
    )
    return y_pred

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--year",
        default= 2021
    )
    parser.add_argument(
        "--month",
        default= 2
    )

    args = parser.parse_args()
    categorical = ['PUlocationID', 'DOlocationID']

    mean_pred = np.mean(predict(int(args.year), int(args.month)))
    print(f"Mean predicted duration for {args.year}-{args.month} is :",mean_pred)


