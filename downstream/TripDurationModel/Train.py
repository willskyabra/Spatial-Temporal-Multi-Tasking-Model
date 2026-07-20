import sklearn
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn.functional as F
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import itertools
from Model import TaxiTripTimeRegressor
from start_end_extract import load_data

if torch.cuda.is_available():
  device = torch.device("cuda:0")
  print("GPU")
else:
  device = torch.device("cpu")
  print("CPU")

def create_dataloader(x, y, batch_size=64, shuffle=True):
    tensor_x = torch.tensor(x, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def train(model, optimizer, criterion, train_loader):
    model.train()
    train_loss = 0.0

    for data in train_loader:
        inputs, targets = data
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    return train_loss / len(train_loader)

def evaluate(model, criterion, test_loader):
    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for data in test_loader:
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)

            test_loss += loss.item()

    return test_loss / len(test_loader)

def tune_hyperparams(x_train_val, y_train_val, epochs, k_folds):
    learning_rates = [0.05, 0.01, 0.005, 0.001]
    batch_sizes = [16, 32, 64]
    w_decays = [0.05, 0.1, 0.5, 0]
    param_combinations = list(itertools.product(learning_rates, batch_sizes, w_decays))

    best_params = None
    best_val_loss = float("inf")

    for lr, batch_size, w_decay in param_combinations:
        print(f"\nTesting Hyperparameters: LR={lr}, Batch Size={batch_size}, Weight Decay={w_decay}")

        kf = KFold(n_splits=k_folds, shuffle=True, random_state=400)
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(x_train_val)):
            x_train_fold, x_val_fold = x_train_val[train_idx], x_train_val[val_idx]
            y_train_fold, y_val_fold = y_train_val[train_idx], y_train_val[val_idx]

            train_loader = create_dataloader(x_train_fold, y_train_fold, batch_size=batch_size, shuffle=True)
            val_loader = create_dataloader(x_val_fold, y_val_fold, batch_size=batch_size, shuffle=False)

            model = TaxiTripTimeRegressor(x_train_val.shape[1]).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=w_decay)
            criterion = torch.nn.MSELoss()

            best_fold_loss = float("inf")

            for epoch in range(epochs):
                train_loss = train(model, optimizer, criterion, train_loader)
                val_loss = evaluate(model, criterion, val_loader)

                if val_loss < best_fold_loss:
                    best_fold_loss = val_loss

            fold_results.append(best_fold_loss)

        avg_val_loss = np.mean(fold_results)
        print(f"Avg Validation Loss for LR={lr}, Batch Size={batch_size}, Weight Decay={w_decay}: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_params = (lr, batch_size, w_decay)

    print(f"\nBest Hyperparameters: LR={best_params[0]}, Batch Size={best_params[1]}, Weight Decay={best_params[2]}, Validation Loss={best_val_loss:.4f}")
    return best_params

def train_model():
    """
    Main function to train and validate the regression model.
    """
    x_train, y_train = load_data("")    
    x_test, y_test = load_data("")

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    # Hyperparamer tuning
    best_lr, best_batch_size, w_decay = tune_hyperparams(x_train, y_train, epochs = 10, k_folds = 5)

    # Early stopping and regular training
    train_loader = create_dataloader(x_train, y_train, batch_size=best_batch_size, shuffle=True)
    val_loader = create_dataloader(x_val, y_val, batch_size=best_batch_size, shuffle=True)

    model = TaxiTripTimeRegressor(x_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_lr, weight_decay=w_decay)
    criterion = torch.nn.MSELoss()

    # Early stopping implementation
    patience = 3
    min_delta = 0.001
    best_val_loss = float("inf")
    counter = 0

    epochs = 10
    for epoch in range(epochs):
        train_loss = train(model, optimizer, criterion, train_loader)
        val_loss = evaluate(model, criterion, val_loader)
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss 
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print(f"Stopped early at epoch {epoch+1}")
            break

    test_loader = create_dataloader(x_test, y_test, batch_size=best_batch_size, shuffle=False)
    test_loss = evaluate(model, criterion, test_loader)
    print(f"Test Loss: {test_loss:.4f}")

    torch.save(model, "trip_time_regressor.pt")
    torch.save(model, "trip_time_regressor.pth")
