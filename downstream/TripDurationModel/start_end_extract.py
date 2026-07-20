import pandas as pd
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
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
        'pickup_zone': 'str'
    }

    date_cols = ['pickup_datetime']

    all_files = glob.glob(file_pattern)
    processed_data = []
    target_values = []
    
    for filename in all_files:
        print("Processing file: ", filename)
        df = pd.read_csv(filename, index_col=None, header=0, dtype=dtype_dict, parse_dates=date_cols)
        X_numerical, y = preprocess_data(df)
        if X_numerical.size > 0:
            processed_data.append(X_numerical)
            target_values.append(y)

    X_numerical_combined = np.concatenate(processed_data, axis=0) if processed_data else np.array([])
    y_combined = np.concatenate(target_values, axis=0) if target_values else np.array([])

    return X_numerical_combined, y_combined

def preprocess_data(frame):
    """
    Perform data preprocessing: feature extraction, scaling, and reshaping.
    """
    frame = frame.sort_values(by=['medallion', 'hack_license', 'pickup_datetime'])
    
    frame['pickup_day'] = frame['pickup_datetime'].dt.day
    frame['pickup_month'] = frame['pickup_datetime'].dt.month
    frame['pickup_year'] = frame['pickup_datetime'].dt.year
    frame['pickup_hour'] = frame['pickup_datetime'].dt.hour
    frame['pickup_minute'] = frame['pickup_datetime'].dt.minute
    frame['pickup_second'] = frame['pickup_datetime'].dt.second

    # Categorical features
    categorical_cols = ['vendor_id', 'store_and_fwd_flag', 'pickup_zone']
    # Apply LabelEncoder to each categorical column and store encoders for later use
    le = {}
    for col in categorical_cols:
        le[col] = LabelEncoder()
        frame[col] = le[col].fit_transform(frame[col])

    # Numerical features
    numerical_cols = ['pickup_day', 'pickup_month', 'pickup_year', 'pickup_hour',
                      'pickup_minute', 'pickup_second', 'passenger_count', 'trip_distance', 'pickup_longitude', 'pickup_latitude', 'vendor_id', 'rate_code', 'store_and_fwd_flag', 'pickup_zone']

    scaler = StandardScaler()
    frame[numerical_cols] = scaler.fit_transform(frame[numerical_cols])

    # Target variable (trip_time_in_secs)
    y = frame['trip_time_in_secs'].values

    return frame[numerical_cols].values, y
