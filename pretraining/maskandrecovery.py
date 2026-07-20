import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from model import TaxiDriverClassifier, EncoderOnlyTransformer, TaxiDriverClassifierSelfAttention
from sklearn.model_selection import train_test_split
from extract_feature import load_data
from torch.autograd import Variable
import math

class TaxiMaskRecovery(nn.Module):
    def __init__(self, nyc_taxi_encoder, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=0.1):
        super(TaxiMaskRecovery, self).__init__()
        self.nyc_taxi_encoder = nyc_taxi_encoder
        
        self.positional_encoding = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        self.output_projection = nn.Linear(d_model, 19)

    def forward(self, X_numerical, X_categorical, X_numerical_masked):
        encoded_features = self.nyc_taxi_encoder(X_numerical, X_categorical, X_numerical_masked)

        encoded_features = encoded_features.unsqueeze(1)
        encoded_features = encoded_features.transpose(0, 1)

        src = self.positional_encoding(encoded_features)

        encoder_output = self.transformer_encoder(src)
        encoder_output = encoder_output.transpose(0, 1)
        
        predictions = self.output_projection(encoder_output.squeeze(1))

        return predictions
    
class NYCFeatureEncoder(nn.Module):
    def __init__(self, medallion_vocab_size, hack_license_vocab_size, vendor_vocab_size, rate_code_vocab_size,
                 store_and_fwd_flag_vocab_size, pickup_zone_vocab_size, dropoff_zone_vocab_size, d_model, numerical_dim=19):
        super(NYCFeatureEncoder, self).__init__()

        self.medallion_embed = nn.Embedding(medallion_vocab_size, d_model // 4)
        self.hack_license_embed = nn.Embedding(hack_license_vocab_size, d_model // 4)
        self.vendor_embed = nn.Embedding(vendor_vocab_size, d_model // 4)
        self.rate_code_embed = nn.Embedding(rate_code_vocab_size, d_model // 4)
        self.store_and_fwd_flag_embed = nn.Embedding(store_and_fwd_flag_vocab_size, d_model // 4)
        self.pickup_zone_embed = nn.Embedding(pickup_zone_vocab_size, d_model // 4)
        self.dropoff_zone_embed = nn.Embedding(dropoff_zone_vocab_size, d_model // 4)

        self.numerical_projection = nn.Linear(numerical_dim, numerical_dim)
        self.numerical_mask_token = nn.Parameter(torch.randn(16, numerical_dim))

        self.output_projection = nn.Linear(243, d_model)

    def forward(self, X_numerical, X_categorical, X_numerical_masked):
        # Extract categorical features from X_categorical
        medallion_id = X_categorical[:, 0]
        hack_license = X_categorical[:, 1]
        vendor_id = X_categorical[:, 2]
        rate_code = X_categorical[:, 3]
        store_and_fwd_flag = X_categorical[:, 4]
        pickup_zone = X_categorical[:, 5]
        dropoff_zone = X_categorical[:, 6]

        # Apply embeddings for categorical features
        medallion_emb = self.medallion_embed(medallion_id)
        hack_license_emb = self.hack_license_embed(hack_license)
        vendor_emb = self.vendor_embed(vendor_id)
        rate_code_emb = self.rate_code_embed(rate_code)
        store_and_fwd_flag_emb = self.store_and_fwd_flag_embed(store_and_fwd_flag)
        pickup_zone_emb = self.pickup_zone_embed(pickup_zone)
        dropoff_zone_emb = self.dropoff_zone_embed(dropoff_zone)
        
        categorical_emb = torch.cat([medallion_emb, hack_license_emb, vendor_emb, rate_code_emb, store_and_fwd_flag_emb, pickup_zone_emb, dropoff_zone_emb], dim=-1)
        
        mask = ~torch.isnan(X_numerical_masked)

        numerical_emb = torch.where(
            mask,
            self.numerical_projection(X_numerical),
            self.numerical_mask_token
        )
        
        combined_features = torch.cat([categorical_emb, numerical_emb], dim=-1)
        
        return self.output_projection(combined_features)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Create a long enough position encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # Shape: [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add position encoding to input
        x = x + self.pe[:x.size(0), :]
        return x


def mask_coordinates(X_numerical):
    pickup_day_idx = 0
    pickup_month_idx = 1
    pickup_year_idx = 2
    pickup_time_hour_idx = 3
    pickup_time_min_idx = 4
    pickup_time_secs_idx = 5
    drop_day_idx = 6
    drop_month_idx = 7
    drop_year_idx = 8
    drop_time_hour_idx = 9
    drop_time_min_idx = 10
    drop_time_secs_idx = 11
    passenger_count_idx = 12
    trip_time_secs_idx = 13
    trip_dist_idx = 14
    pickup_lat_idx = 15  
    pickup_long_idx = 16 
    dropoff_lat_idx = 17
    dropoff_long_idx = 18 

    X_for_mask = X_numerical.clone()
    batch_size = X_numerical.shape[0]

    mask_choices = torch.randint(0,4, (batch_size,))

    # Mask coords
    X_for_mask[mask_choices == 0, pickup_lat_idx] = float('nan')
    X_for_mask[mask_choices == 0, pickup_long_idx] = float('nan')
    X_for_mask[mask_choices == 1, dropoff_lat_idx] = float('nan')
    X_for_mask[mask_choices == 1, dropoff_long_idx] = float('nan')

    # Mask trip info stuff
    X_for_mask[mask_choices == 2, passenger_count_idx] = float('nan')
    X_for_mask[mask_choices == 3, trip_time_secs_idx] = float('nan')
    X_for_mask[mask_choices == 4, trip_dist_idx] = float('nan')

    # Mask pickup time stuff
    X_for_mask[mask_choices == 5, pickup_day_idx] = float('nan')
    X_for_mask[mask_choices == 6, pickup_month_idx] = float('nan')
    X_for_mask[mask_choices == 7, pickup_year_idx] = float('nan')
    X_for_mask[mask_choices == 3, pickup_time_hour_idx] = float('nan')
    X_for_mask[mask_choices == 3, pickup_time_min_idx] = float('nan')
    X_for_mask[mask_choices == 3, pickup_time_secs_idx] = float('nan')

    # Mask dropoff time stuff
    X_for_mask[mask_choices == 11, drop_day_idx] = float('nan')
    X_for_mask[mask_choices == 12, drop_month_idx] = float('nan')
    X_for_mask[mask_choices == 13, drop_year_idx] = float('nan')
    X_for_mask[mask_choices == 3, drop_time_hour_idx] = float('nan')
    X_for_mask[mask_choices == 3, drop_time_min_idx] = float('nan')
    X_for_mask[mask_choices == 3, drop_time_secs_idx] = float('nan')

    return X_for_mask, mask_choices




class TaxiMaskRecoveryNoZone(nn.Module):
    def __init__(self, nyc_taxi_encoder, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=0.1):
        super(TaxiMaskRecoveryNoZone, self).__init__()
        self.nyc_taxi_encoder = nyc_taxi_encoder
        
        self.positional_encoding = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        self.output_projection = nn.Linear(d_model, 19)

    def forward(self, X_numerical, X_categorical, X_numerical_masked):
        encoded_features = self.nyc_taxi_encoder(X_numerical, X_categorical, X_numerical_masked)

        encoded_features = encoded_features.unsqueeze(1)
        encoded_features = encoded_features.transpose(0, 1)

        src = self.positional_encoding(encoded_features)

        encoder_output = self.transformer_encoder(src)
        encoder_output = encoder_output.transpose(0, 1)
        
        predictions = self.output_projection(encoder_output.squeeze(1))

        return predictions
    
class NYCFeatureEncoderNoZone(nn.Module):
    def __init__(self, medallion_vocab_size, hack_license_vocab_size, vendor_vocab_size, rate_code_vocab_size,
                 store_and_fwd_flag_vocab_size, pickup_zone_vocab_size, dropoff_zone_vocab_size, d_model, numerical_dim=19):
        super(NYCFeatureEncoderNoZone, self).__init__()

        self.medallion_embed = nn.Embedding(medallion_vocab_size, d_model // 4)
        self.hack_license_embed = nn.Embedding(hack_license_vocab_size, d_model // 4)
        self.vendor_embed = nn.Embedding(vendor_vocab_size, d_model // 4)
        self.rate_code_embed = nn.Embedding(rate_code_vocab_size, d_model // 4)
        self.store_and_fwd_flag_embed = nn.Embedding(store_and_fwd_flag_vocab_size, d_model // 4)
        self.pickup_zone_embed = nn.Embedding(pickup_zone_vocab_size, d_model // 4)
        self.dropoff_zone_embed = nn.Embedding(dropoff_zone_vocab_size, d_model // 4)

        self.numerical_projection = nn.Linear(numerical_dim, numerical_dim)
        self.numerical_mask_token = nn.Parameter(torch.randn(16, numerical_dim))

        self.output_projection = nn.Linear(179, d_model)

    def forward(self, X_numerical, X_categorical, X_numerical_masked):
        # Extract categorical features from X_categorical
        medallion_id = X_categorical[:, 0]
        hack_license = X_categorical[:, 1]
        vendor_id = X_categorical[:, 2]
        rate_code = X_categorical[:, 3]
        store_and_fwd_flag = X_categorical[:, 4]

        # Apply embeddings for categorical features
        medallion_emb = self.medallion_embed(medallion_id)
        hack_license_emb = self.hack_license_embed(hack_license)
        vendor_emb = self.vendor_embed(vendor_id)
        rate_code_emb = self.rate_code_embed(rate_code)
        store_and_fwd_flag_emb = self.store_and_fwd_flag_embed(store_and_fwd_flag)
        
        categorical_emb = torch.cat([medallion_emb, hack_license_emb, vendor_emb, rate_code_emb, store_and_fwd_flag_emb], dim=-1)
        
        mask = ~torch.isnan(X_numerical_masked)

        numerical_emb = torch.where(
            mask,
            self.numerical_projection(X_numerical),
            self.numerical_mask_token
        )
        
        combined_features = torch.cat([categorical_emb, numerical_emb], dim=-1)
        
        return self.output_projection(combined_features)

