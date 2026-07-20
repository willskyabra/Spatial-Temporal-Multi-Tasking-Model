import sys
sys.path.append('/home/neishigami/CS586-DS504-Spring2024/project2')

import sklearn
from sklearn import model_selection
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn.functional as F
from tqdm import tqdm
from Model import TaxiTripTimeRegressor
from start_end_extract import load_data
from maskandrecovery import TaxiMaskRecoveryNoZone, NYCFeatureEncoderNoZone

if torch.cuda.is_available():
  device = torch.device("cuda:0")
  print("GPU")
else:
  device = torch.device("cpu")
  print("CPU")

BATCH_SIZE = 64

class TaxiTripDataset(Dataset):
    """
    Custom dataset class for Taxi Trip Time Regression.
    Handles loading and preparing data for the model.
    """
    def __init__(self, x_train_num, x_train_cat, y_train, x_test_num, x_test_cat, y_test):
        X_train_num, X_val_num, X_train_cat, X_val_cat, Y_train, Y_val = model_selection.train_test_split(x_train_num, x_train_cat, y_train, test_size=0.25, random_state=400)

        self.train_set = DataLoader(TensorDataset(torch.tensor(X_train_num, dtype=torch.float32), torch.tensor(X_train_cat, dtype=torch.long), torch.tensor(Y_train, dtype=torch.float32)), batch_size=16, shuffle=True)
        self.test_set = DataLoader(TensorDataset(torch.tensor(x_test_num, dtype=torch.float32), torch.tensor(x_test_cat, dtype=torch.long), torch.tensor(y_test, dtype=torch.float32)), batch_size=16, shuffle=False)
        self.validate_set = DataLoader(TensorDataset(torch.tensor(X_val_num, dtype=torch.float32), torch.tensor(X_val_cat, dtype=torch.long), torch.tensor(Y_val, dtype=torch.float32)), batch_size=16, shuffle=False)

def train(model, optimizer, criterion, train_loader):
    """
    Function to train the regression model.
    """
    model.train()
    train_loss = 0.0
    count = 0
    count_correct = 0
    tolerance = 0.1
    i = 0

    for data in tqdm(train_loader):
        X_num, X_cat, targets = data
        X_num, X_cat, targets = X_num.to(device), X_cat.to(device), targets.to(device)

        if X_num.shape[0] == 16:
            optimizer.zero_grad()
            outputs = model(X_num, X_cat)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()

            if i == 0:
                print('targets: ', targets)
                print('outputs: ', outputs.squeeze())
                i =1

            count_correct += torch.count_nonzero(torch.abs(targets-outputs.squeeze()) < tolerance).item()
            count += BATCH_SIZE

            train_loss += loss.item()

    return train_loss / len(train_loader), count_correct / count

def evaluate(model, criterion, test_loader):
    """
    Function to evaluate the regression model.
    """
    model.eval()
    test_loss = 0.0
    count = 0
    count_correct = 0
    tolerance = 0.1

    with torch.no_grad():
        for data in test_loader:
            X_num, X_cat, targets = data
            X_num, X_cat, targets = X_num.to(device), X_cat.to(device), targets.to(device)

            if X_num.shape[0] == 16:
                outputs = model(X_num, X_cat)
                loss = criterion(outputs.squeeze(), targets)

                count_correct += torch.count_nonzero(torch.abs(targets-outputs.squeeze()) < tolerance).item()
                count += BATCH_SIZE

                test_loss += loss.item()

    return test_loss / len(test_loader), count_correct / count

def train_model():
    """
    Main function to train and validate the regression model.
    """
    x_numerical_train, x_categorical_train, y_train = load_data("")
    x_numerical_test, x_categorical_test, y_test = load_data("")

    dataset = TaxiTripDataset(x_numerical_train, x_categorical_train, y_train, x_numerical_test, x_categorical_test, y_test)

    nyc_encoder = NYCFeatureEncoderNoZone(13415, 32062, 2, 12, 3, 4763, 5749, 128).to(device)
    transformer = TaxiMaskRecoveryNoZone(nyc_encoder, 128, 4, 3, 128).to(device)
    # pretrained_state_dict = torch.load('', map_location=device)
    # transformer.load_state_dict(pretrained_state_dict, strict=False)

    # for param in transformer.parameters():
    #     param.requires_grad = False

    model = TaxiTripTimeRegressor(x_numerical_train.shape[1], transformer).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    criterion = torch.nn.MSELoss()

    epochs = 10
    for i in range(epochs):
        train_loss, train_acc = train(model, optimizer, criterion, dataset.train_set)
        print(f"Epoch {i+1} | Training Acc: {train_acc*100:.4f}%")
        print(f"Epoch {i+1} | Training Loss: {train_loss:.4f}")

        val_loss, val_acc = evaluate(model, criterion, dataset.validate_set)
        print(f"Epoch {i+1} | Validation Acc: {val_acc*100:.4f}%")
        print(f"Epoch {i+1} | Validation Loss: {val_loss:.4f}")


    test_loss, test_acc = evaluate(model, criterion, dataset.test_set)
    print(f"Test Acc: {test_acc*100:.4f}%")
    print(f"Test Loss: {test_loss:.4f}")

    # torch.save(model, "trip_time_regressor.pt")
    # torch.save(model, "trip_time_regressor.pth")
