# %%
!pip freeze | grep scikit-learn

# %%
import pickle
import pandas as pd

# %%
with open('model.bin', 'rb') as f_in:
    dv, lr = pickle.load(f_in)

# %%
categorical = ['PUlocationID', 'DOlocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

# %%
df = read_data('https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_2021-02.parquet')

# %%
dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = lr.predict(X_val)

# %%
import numpy as np
np.mean(y_pred)

# %%
df.head()

# %%

df['ride_id'] = f'{2021:04d}/{2:02d}_' + df.index.astype('str')
df['ride_id']

# %%
df_result = pd.DataFrame()
df_result['ride_id'] = df[['ride_id']]
df_result['pred'] = y_pred

df_result.to_parquet(
    'df_results.parquet',
    engine='pyarrow',
    compression=None,
    index=False
)


