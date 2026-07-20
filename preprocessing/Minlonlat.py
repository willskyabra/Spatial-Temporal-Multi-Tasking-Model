import string
import pandas as pd


global_min_lat = 1e5
global_min_lon = 1e5

for i in range(1, 13):
    df = pd.read_csv('' + str(i) + '.csv', dtype={4: 'str'})

    df.columns = df.columns.str.strip()

    # Removing rows with incorrect latitude or longitude values
    df = df[
        (df['pickup_latitude'] >= 30) & (df['pickup_latitude'] <= 50) &
        (df['pickup_longitude'] <= -65) & (df['pickup_longitude'] >= -85) &
        (df['dropoff_latitude'] >= 30) & (df['dropoff_latitude'] <= 50) &
        (df['dropoff_longitude'] <= -65) & (df['dropoff_longitude'] >= -85)
    ]

    min_latitude = min(df['pickup_latitude'].min(), df['dropoff_latitude'].min())
    min_longitude = min(df['pickup_longitude'].min(), df['dropoff_longitude'].min())

    if min_latitude < global_min_lat:
        global_min_lat = min_latitude
    if min_longitude < global_min_lon:
        global_min_lon = min_longitude

print('Global min latitude: ' + str(global_min_lat))
print('Global min longitude: ' + str(global_min_lon))

data = {
    'Global min latitude': [global_min_lat],
    'Global min longitude': [global_min_lon]
}

min_df = pd.DataFrame(data)

min_df.to_csv('',index=False)
