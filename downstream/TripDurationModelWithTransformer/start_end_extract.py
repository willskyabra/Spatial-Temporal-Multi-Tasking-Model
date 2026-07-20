import pandas as pd
import glob
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_data(file_pattern):
    """
    Load and process CSV files matching the given file pattern, one at a time.
    """
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

    all_files = glob.glob(file_pattern)
    processed_data = []
    target_values = []
    
    for filename in all_files:
        print("Processing file: ", filename)
        df = pd.read_csv(filename, index_col=None, header=0, dtype=dtype_dict, parse_dates=date_cols)
        X_numerical, X_categorical, y = preprocess_data(df)
        if X_numerical.size > 0:  # Check if there's data to add
            processed_data.append((X_numerical, X_categorical))
            target_values.append(y)

    # Combine data from all files
    X_numerical_combined = np.concatenate([data[0] for data in processed_data], axis=0) if processed_data else np.array([])
    X_categorical_combined = np.concatenate([data[1] for data in processed_data], axis=0) if processed_data else np.array([])
    y_combined = np.concatenate(target_values, axis=0) if target_values else np.array([])

    return X_numerical_combined, X_categorical_combined, y_combined

def preprocess_data(frame):
    """
    Perform data preprocessing: feature extraction, scaling, and reshaping.
    """
    frame = frame.sort_values(by=['medallion', 'hack_license', 'pickup_datetime'])
    
    frame['pickup_day'] = frame['pickup_datetime'].dt.day
    frame['pickup_month'] = frame['pickup_datetime'].dt.month
    frame['pickup_year'] = frame['pickup_datetime'].dt.year
    frame['pickup_time'] = frame['pickup_datetime']
    frame["pickup_time_in_hour"] = frame["pickup_datetime"].dt.hour
    frame["pickup_time_in_minute"] = frame["pickup_datetime"].dt.minute
    frame["pickup_time_in_seconds"] = frame["pickup_datetime"].dt.second

    frame['dropoff_day'] = frame['dropoff_datetime'].dt.day
    frame['dropoff_month'] = frame['dropoff_datetime'].dt.month
    frame['dropoff_year'] = frame['dropoff_datetime'].dt.year
    frame['dropoff_time'] = frame['dropoff_datetime']
    frame["dropoff_time_in_hour"] = frame["dropoff_datetime"].dt.hour
    frame["dropoff_time_in_minute"] = frame["dropoff_datetime"].dt.minute
    frame["dropoff_time_in_seconds"] = frame["dropoff_datetime"].dt.second

    categorical_cols = ['medallion', 'hack_license', 'vendor_id', 'rate_code', 'store_and_fwd_flag']

    le = {}
    for col in categorical_cols:
        le[col] = LabelEncoder()
        frame[col] = le[col].fit_transform(frame[col])

    # Numerical features
    numerical_cols = ['pickup_day', 'pickup_month', 'pickup_year', 'pickup_time_in_hour', 
                      'pickup_time_in_minute', 'pickup_time_in_seconds', 'dropoff_day', 
                      'dropoff_month', 'dropoff_year', 'dropoff_time_in_hour', 
                      'dropoff_time_in_minute', 'dropoff_time_in_seconds', 'passenger_count', 
                      'trip_time_in_secs', 'trip_distance', 'pickup_longitude', 
                      'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']

    scaler = StandardScaler()
    frame[numerical_cols] = scaler.fit_transform(frame[numerical_cols])

    y = frame['trip_time_in_secs'].values

    frame['dropoff_day'] = float('nan')
    frame['dropoff_month'] = float('nan')
    frame['dropoff_year'] = float('nan')
    frame['dropoff_time'] = float('nan')
    frame["dropoff_time_in_hour"] = float('nan')
    frame["dropoff_time_in_minute"] = float('nan')
    frame["dropoff_time_in_seconds"] = float('nan')
    frame["trip_time_in_secs"] = float('nan')
    frame["dropoff_longitude"] = float('nan')
    frame["dropoff_latitude"] = float('nan')

    X_numerical = frame[numerical_cols].values
    X_categorical = frame[categorical_cols].values

    return X_numerical, X_categorical, y
