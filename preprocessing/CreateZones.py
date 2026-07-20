import string
import pandas as pd


global_min_lon_lat = pd.read_csv('')
min_lat = global_min_lon_lat.loc[0, 'Global min latitude']
min_lon = global_min_lon_lat.loc[0, 'Global min longitude']

# Adding zone columns
df = pd.read_csv('')
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

def calculate_zone(lat, lon, min_lat, min_lon, interval=0.01):
    lat_index = int((lat - min_lat) // interval)
    lon_index = int((lon - min_lon) // interval)
    
    lat_letter = ""
    while lat_index >= 0:
        lat_letter = string.ascii_uppercase[lat_index % 26] + lat_letter
        lat_index = (lat_index // 26) - 1
    lon_number = lon_index + 1
    
    return f"{lat_letter}{lon_number}"


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

    df['pickup_zone'] = df.apply(
        lambda row: calculate_zone(row['pickup_latitude'], row['pickup_longitude'], min_latitude, min_longitude),
        axis=1
    )

    df['dropoff_zone'] = df.apply(
        lambda row: calculate_zone(row['dropoff_latitude'], row['dropoff_longitude'], min_latitude, min_longitude),
        axis=1
    )

    df.to_csv('' + str(i) + '_zones.csv', index=False)

print('Complete')
