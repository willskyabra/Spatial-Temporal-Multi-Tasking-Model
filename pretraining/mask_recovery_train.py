import torch
import torch.nn as nn
import torch.optim as optim
import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
from start_end_extract_zones import load_data
from maskandrecovery import NYCFeatureEncoder, TaxiMaskRecovery, mask_coordinates
from sklearn.preprocessing import StandardScaler

TEST_SIZE = 0.1
VAL_SIZE = 0.111111  # for splitting the 90% of remaining data into, 80% an 10% respectively
BATCH_SIZE = 16
SEED = 21

LEARNING_RATE = 0.00001

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

# medallion vocab size:13415
# hack vocab size:32062
# vendor vocab size:2
# rate vocab size:12
# store vocab size:2
# pickup zone vocab size:4763
# dropoff zone vocab size:5749

if torch.cuda.is_available():
  device = torch.device("cuda:0")
  print("GPU")
else:
  device = torch.device("cpu")
  print("CPU")

class TaxiDriverDataset(torch.utils.data.Dataset):
    def __init__(self, X_numerical, X_categorical, device):
        self.X_numerical = torch.as_tensor(X_numerical, dtype=torch.float32, device=device)
        self.X_categorical = torch.as_tensor(X_categorical, dtype=torch.long, device=device)

    def __len__(self):
        return len(self.X_numerical)

    def __getitem__(self, idx):
        return self.X_numerical[idx], self.X_categorical[idx]
    
def load_data_split():
    X_numerical, X_categorical, scaler = load_data("")
    dataset = TaxiDriverDataset(X_numerical, X_categorical, device)

    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    next_split_indices, test_indices = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=SEED
    )

    train_indices, val_indices = train_test_split(
        next_split_indices,
        test_size=VAL_SIZE,
        random_state=SEED
    )

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    test_subset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE)

    return train_loader, val_loader, test_loader, scaler

def torch_stack_labels(inps_unmasked):
    labels = torch.stack([
        inps_unmasked[:, pickup_lat_idx],
        inps_unmasked[:, pickup_long_idx],
        inps_unmasked[:, dropoff_lat_idx],
        inps_unmasked[:, dropoff_long_idx],
        inps_unmasked[:, passenger_count_idx],
        inps_unmasked[:, trip_time_secs_idx],
        inps_unmasked[:, trip_dist_idx],
        inps_unmasked[:, pickup_day_idx],
        inps_unmasked[:, pickup_month_idx],
        inps_unmasked[:, pickup_year_idx],
        inps_unmasked[:, pickup_time_hour_idx],
        inps_unmasked[:, pickup_time_min_idx],
        inps_unmasked[:, pickup_time_secs_idx],
        inps_unmasked[:, drop_day_idx],
        inps_unmasked[:, drop_month_idx],
        inps_unmasked[:, drop_year_idx],
        inps_unmasked[:, drop_time_hour_idx],
        inps_unmasked[:, drop_time_min_idx],
        inps_unmasked[:, drop_time_secs_idx]
    ], dim=1)

    return labels

def get_masks(mask_choices):
    pickup_mask = mask_choices == 0
    dropoff_mask = mask_choices == 1
    pass_mask = mask_choices == 2
    total_time_mask = mask_choices == 3
    trip_dist_mask = mask_choices == 4
    pickup_day_mask = mask_choices == 5
    pickup_month_mask = mask_choices == 6
    pickup_year_mask = mask_choices == 7
    pt_hour_mask = mask_choices == 8
    pt_min_mask = mask_choices == 9
    pt_secs_mask = mask_choices ==  10
    drop_day_mask = mask_choices == 11
    drop_month_mask = mask_choices == 12
    drop_year_mask = mask_choices == 13
    dt_hour_mask = mask_choices == 14
    dt_min_mask = mask_choices == 15
    dt_secs_mask = mask_choices == 16

    masks = [pickup_mask, dropoff_mask, pass_mask, total_time_mask, trip_dist_mask, pickup_day_mask, pickup_month_mask, pickup_year_mask, pt_hour_mask, pt_min_mask, pt_secs_mask, drop_day_mask, drop_month_mask, drop_year_mask, dt_hour_mask, dt_min_mask, dt_secs_mask]

    return masks

def train(model, optimizer, loss_fn, train_loader, device, scaler):
    model.train()
    feature_counts = []
    feature_correct_counts = []
    feature_total_counts = torch.zeros(17, device=device)
    feature_total_correct_counts = torch.zeros(17, device=device)
    train_coord_acc = 0.0
    train_count = 0
    train_loss = 0
    tolerance = 0.01
    k = 0

    for X_numerical, X_categorical in tqdm(train_loader):
        # k += 1
        feature_counts = []
        feature_correct_counts = []
        numerical_inps = X_numerical.to(device)
        categorical_inps = X_categorical.to(device)

        inps_unmasked = numerical_inps.clone()

        masked_numerical_inps, mask_choices = mask_coordinates(numerical_inps)

        labels = torch_stack_labels(inps_unmasked)

        coord_pred = model(inps_unmasked, categorical_inps, masked_numerical_inps)

        masked_pred = torch.zeros((mask_choices.size(0), 2), device=device)
        masked_labels = torch.zeros((mask_choices.size(0), 2), device=device)

        masks = get_masks(mask_choices)

        # if k % 1000 == 0:
        #     print('inputs unmasked', inps_unmasked[:,12])
        #     inps = inps_unmasked.cpu()
        #     orig_values = scaler.inverse_transform(inps)
        #     print('orig values', orig_values[:,12])
        #     print('predicted scaled', coord_pred[:,12])
        #     inps2 = coord_pred.cpu()
        #     pred_transform = scaler.inverse_transform(inps2.detach().numpy())
        #     print('prediction real values', pred_transform[:,12])

        for i, mask in enumerate(masks):
            if i == 0:
                masked_pred[mask, 0] = coord_pred[mask, 0]
                masked_pred[mask, 1] = coord_pred[mask, 1]
                masked_labels[mask, 0] = labels[mask, 0]
                masked_labels[mask, 1] = labels[mask, 1]
            elif i == 1:
                masked_pred[mask, 0] = coord_pred[mask, 2]
                masked_pred[mask, 1] = coord_pred[mask, 3]
                masked_labels[mask, 0] = labels[mask, 2]
                masked_labels[mask, 1] = labels[mask, 3]
            else:
                index = i + 2
                masked_pred[mask, 0] = coord_pred[mask, index]
                masked_labels[mask, 0] = labels[mask, index]

        loss = loss_fn(masked_pred, masked_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for i, mask in enumerate(masks):
            if i == 0 or i == 1:
                feature_counts.append(mask.sum().item() * 2)
            else:
                feature_counts.append(mask.sum().item())

        # Combine all counts
        total_count = sum(feature_counts)
        train_count += total_count

        for i, mask in enumerate(masks):
            if i == 0 or i == 1:
                feature_correct_counts.append(torch.count_nonzero(torch.abs(masked_pred[mask] - masked_labels[mask]) < tolerance).item())
            else:
                feature_correct_counts.append(torch.count_nonzero(torch.abs(masked_pred[mask] - masked_labels[mask])[:, 0] < 0.1).item())

        for i in range(17):
            feature_total_counts[i] += feature_counts[i]
            feature_total_correct_counts[i] += feature_correct_counts[i]

        # Sum all correct predictions
        total_correct = sum(feature_correct_counts)
        train_coord_acc += total_correct

        train_loss += loss.item() * len(X_numerical)

    train_coord_acc /= train_count
    train_loss /= train_count
    feature_accs = feature_total_correct_counts / feature_total_counts

    return train_loss, train_coord_acc, feature_accs

# Define the testing function
def evaluate(model, loss_fn, val_loader, device):
    model.eval()
    feature_counts = []
    feature_correct_counts = []
    feature_total_counts = torch.zeros(17, device=device)
    feature_total_correct_counts = torch.zeros(17, device=device)
    val_acc = 0.0
    val_loss = 0.0
    val_count = 0
    tolerance = 0.01
    i = 0

    with torch.no_grad():
        for X_numerical, X_categorical in val_loader:  
            feature_counts = []
            feature_correct_counts = []
            numerical_inps = X_numerical.to(device)
            categorical_inps = X_categorical.to(device)

            inps_unmasked = numerical_inps.clone()

            masked_numerical_inps, mask_choices = mask_coordinates(numerical_inps)

            labels = torch_stack_labels(inps_unmasked)

            coord_pred = model(inps_unmasked, categorical_inps, masked_numerical_inps)

            masked_pred = torch.zeros((mask_choices.size(0), 2), device=device)
            masked_labels = torch.zeros((mask_choices.size(0), 2), device=device)

            masks = get_masks(mask_choices)

            for i, mask in enumerate(masks):
                if i == 0:
                    masked_pred[mask, 0] = coord_pred[mask, 0]
                    masked_pred[mask, 1] = coord_pred[mask, 1]
                    masked_labels[mask, 0] = labels[mask, 0]
                    masked_labels[mask, 1] = labels[mask, 1]
                elif i == 1:
                    masked_pred[mask, 0] = coord_pred[mask, 2]
                    masked_pred[mask, 1] = coord_pred[mask, 3]
                    masked_labels[mask, 0] = labels[mask, 2]
                    masked_labels[mask, 1] = labels[mask, 3]
                else:
                    index = i + 2
                    masked_pred[mask, 0] = coord_pred[mask, index]
                    masked_labels[mask, 0] = labels[mask, index]

            loss = loss_fn(masked_pred, masked_labels)

            # Compute individual counts for each feature
            for i, mask in enumerate(masks):
                if i == 0 or i == 1:
                    feature_counts.append(mask.sum().item() * 2)
                else:
                    feature_counts.append(mask.sum().item())

            # Combine all counts
            total_count = sum(feature_counts)
            val_count += total_count

            # Compute correct predictions for each feature
            for i, mask in enumerate(masks):
                if i == 0 or i == 1:
                    feature_correct_counts.append(torch.count_nonzero(torch.abs(masked_pred[mask] - masked_labels[mask]) < tolerance).item())
                else:
                    feature_correct_counts.append(torch.count_nonzero(torch.abs(masked_pred[mask] - masked_labels[mask])[:, 0] < 0.1).item())

            for i in range(17):
                feature_total_counts[i] += feature_counts[i]
                feature_total_correct_counts[i] += feature_correct_counts[i]

            # Sum all correct predictions
            total_correct = sum(feature_correct_counts)
            val_acc += total_correct

            val_loss += loss.item() * len(X_numerical)

    val_acc /= val_count
    val_loss /= val_count
    feature_accs = feature_total_correct_counts / feature_total_counts

    return val_loss, val_acc, feature_accs


def train_model():
    """
    Main function to initiate the model training process.
    Includes loading data, setting up the model, optimizer, and criterion,
    and executing the training and validation loops.
    """

    # medallion vocab size:13415
    # hack vocab size:32062
    # vendor vocab size:2
    # rate vocab size:12
    # store vocab size:2
    # pickup zone vocab size:4763
    # dropoff zone vocab size:5749

    feature_names = ['pickup', 'dropoff', 'pass_count', 'total_time', 'trip_dist', 'pickup_day', 'pickup_month', 'pickup_year', 'pt_hour', 'pt_min', 'pt_secs', 'drop_day', 'drop_month', 'drop_year', 'dt_hour', 'dt_min', 'dt_secs']
    train_loader, val_loader, test_loader, scaler = load_data_split()
    DIMENSION = 128
    nyc_encoder = NYCFeatureEncoder(13415, 32062, 2, 12, 3, 4763, 5749, DIMENSION).to(device)
    model = TaxiMaskRecovery(nyc_encoder, DIMENSION, 4, 3, DIMENSION).to(device)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # model setup for single coordinate recovery
    # nyc_encoder = NYCFeatureEncoder(13415, 32062, 2, 12, 3, 128).to(device)
    # model = TaxiMaskRecovery(nyc_encoder, 128, 4, 3, 128).to(device)

    # loss_fn = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.000275)

    print("Learning rate: ", LEARNING_RATE)
    print("Model dimension: ", DIMENSION)
    print("Batch size: ", BATCH_SIZE)

    n_epochs = 10
    count_bad_acc = 0
    max_val_acc = 80
    max_train_acc = 80
    model_count = 0
    best_model_path = ''

    for epoch in range(n_epochs):
        train_loss, train_coord_acc, feature_acc_train = train(model, optimizer, loss_fn, train_loader, device, scaler)
        val_loss, val_coord_acc, feature_acc_val = evaluate(model, loss_fn, val_loader, device)

        print("\n" + str(datetime.datetime.now()))
        print(f"\nEpoch {epoch+1}: Train accuracy - Prediction All Numerical Random Masked: {train_coord_acc*100:.2f}%, Train loss: {train_loss:.4f}")
        print(f"Epoch {epoch+1}: Validation accuracy - Prediction All Numerical Random Masked: {val_coord_acc*100:.2f}%, Validation loss: {val_loss:.4f}\n")

        for i in range(17):
            print("Train accuracy - " + feature_names[i] + f": {feature_acc_train[i]*100:.2f}%")
            print("Validation accuracy - " + feature_names[i] + f": {feature_acc_val[i]*100:.2f}%")

        # for early stopping if I want
        if val_coord_acc * 100 < 30:
            count_bad_acc += 1

        if val_coord_acc * 100 > max_val_acc and train_coord_acc * 100 > max_train_acc:
            max_val_acc = val_coord_acc * 100
            max_train_acc = train_coord_acc * 100

            if model_count < 5 and max_val_acc > 80:
                path = ""
                model_name = "TimeFeatures-Epoch" + str(epoch+1) + "TrainAccuracy" + str(max_train_acc) + "ValAccuracy" + str(max_val_acc) + ".pt"
                path = path + model_name
                best_model_path = path
                torch.save(model.state_dict(),path)
                model_count = model_count + 1 

        # if count_bad_acc > 5:
        #     break

    # Load the model with the best weights 
    model = TaxiMaskRecovery(nyc_encoder, 128, 4, 3, 128).to(device)
    model.load_state_dict(torch.load(best_model_path))

    # Evaluate the model on the test set
    test_loss, test_coord_acc, feature_acc_test = evaluate(model, loss_fn, test_loader, device)

    print(f"\nTest Results: Test accuracy - Prediction All Numerical Random Masked: {test_coord_acc*100:.2f}%, Test loss: {test_loss:.4f}")

    for i in range(17):
        print("Train accuracy - " + feature_names[i] + f": {feature_acc_test[i]*100:.2f}%")
        print("Validation accuracy - " + feature_names[i] + f": {feature_acc_test[i]*100:.2f}%")