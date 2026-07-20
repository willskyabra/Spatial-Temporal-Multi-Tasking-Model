import pandas as pd

csv_file = ''

df = pd.read_csv(csv_file, dtype={4: 'str'})

df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'])

train_date1 = '2013-08-25'
train_date2 = '2013-08-26'
train_date3 = '2013-08-27'
test_date = '2013-08-27'

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
