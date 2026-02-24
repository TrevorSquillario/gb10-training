import time
import pandas as pd
import glob
import os

start = time.perf_counter()
csv_pattern = os.path.expanduser('~/gb10/nyc-data/extracted/*.csv')
df = pd.concat((pd.read_csv(f) for f in sorted(glob.glob(csv_pattern))), ignore_index=True)
print('Total Rows:', len(df))
df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
df = df[df['passenger_count'] > 0]
agg = df.groupby('PULocationID').agg({'trip_distance': 'mean', 'total_amount': 'sum'})
print(agg.head())
print('Elapsed (CPU):', time.perf_counter() - start)