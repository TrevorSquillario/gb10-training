import time
import cudf
import glob
import os

start = time.perf_counter()
csv_pattern = os.path.expanduser('~/gb10/nyc-data/extracted/*.csv')
df = cudf.concat([cudf.read_csv(f) for f in sorted(glob.glob(csv_pattern))])
print('Total Rows:', len(df))
# simple example transform + aggregation
df['tpep_pickup_datetime'] = cudf.to_datetime(df['tpep_pickup_datetime'])
df = df[df['passenger_count'] > 0]
agg = df.groupby('PULocationID').agg({'trip_distance': 'mean', 'total_amount': 'sum'})
print(agg.head())
print('Elapsed (GPU):', time.perf_counter() - start)