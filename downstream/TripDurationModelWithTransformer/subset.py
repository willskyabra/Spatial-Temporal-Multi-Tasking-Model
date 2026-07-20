import pandas as pd

# 2milrows datetimes
# 2013-02-01 00:02:00
# 2013-02-28 23:59:09

csv_file = ''

dtype_dict = {
    'medallion': 'str',
    'hack_license': 'str',
    'vendor_id': 'str',
    'rate_code': 'int',
    'store_and_fwd_flag': 'str',
    'passenger_count': 'int',
    'trip_time_in_secs': 'int',
    'trip_distance': 'float',
    'pickup_longitude': 'float',
    'pickup_latitude': 'float',
    'dropoff_longitude': 'float',
    'dropoff_latitude': 'float',
    'pickup_zone': 'str',
    'dropoff_zone': 'str'
}

date_cols = ['pickup_datetime', 'dropoff_datetime']

df = pd.read_csv(csv_file, index_col=None, header=0, dtype=dtype_dict, parse_dates=date_cols)

train_date1 = '2013-02-01'
train_date2 = '2013-02-02'
train_date3 = '2013-02-03'
test_date = '2013-02-04'

train_df = df[
    ((df['pickup_datetime'].dt.date.astype(str) == train_date1) &
    ((df['pickup_datetime'].dt.hour >= 9) & (df['pickup_datetime'].dt.hour < 10) &
     (df['dropoff_datetime'].dt.hour >= 9) & (df['dropoff_datetime'].dt.hour < 10))) |
    ((df['pickup_datetime'].dt.date.astype(str) == train_date2) &
    ((df['pickup_datetime'].dt.hour >= 9) & (df['pickup_datetime'].dt.hour < 10) &
     (df['dropoff_datetime'].dt.hour >= 9) & (df['dropoff_datetime'].dt.hour < 10))) |
    ((df['pickup_datetime'].dt.date.astype(str) == train_date3) &
    ((df['pickup_datetime'].dt.hour >= 7) & (df['pickup_datetime'].dt.hour < 9) &
     (df['dropoff_datetime'].dt.hour >= 7) & (df['dropoff_datetime'].dt.hour < 9)))
]

test_df = df[
    (df['pickup_datetime'].dt.date.astype(str) == test_date) &
    (df['pickup_datetime'].dt.hour >= 9) & (df['pickup_datetime'].dt.hour < 10) &
    ((df['dropoff_datetime'].dt.hour >= 9) & (df['dropoff_datetime'].dt.hour < 10))
]

train_df.to_csv('', index=False)
test_df.to_csv('', index=False)