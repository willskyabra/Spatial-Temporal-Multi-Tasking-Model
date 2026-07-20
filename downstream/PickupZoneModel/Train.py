import sklearn
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn.functional as F
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tqdm import tqdm
import itertools
from Model import PickupZoneClassifier
from start_end_extract import load_data

if torch.cuda.is_available():
  device = torch.device("cuda:0")
  print("GPU")
else:
  device = torch.device("cpu")
  print("CPU")

def plot_heatmap(class_accuracies, filename):
    grid_data = {}
    lat_labels = set()
    lon_labels = set()
    
    for zone, acc in class_accuracies.items():
        lat_part = "".join(filter(str.isalpha, zone))
        lon_part = "".join(filter(str.isdigit, zone))
        
        if lat_part and lon_part:
            lat_idx = sum((ord(char) - ord('A') + 1) * (26**i) for i, char in enumerate(reversed(lat_part))) - 1
            lon_idx = int(lon_part) - 1
            grid_data[(lat_idx, lon_idx)] = acc
            lat_labels.add((lat_idx, lat_part))
            lon_labels.add((lon_idx, lon_part))
    
    min_lat, max_lat = (min(lat_labels)[0], max(lat_labels)[0]) if lat_labels else (0, 0)
    min_lon, max_lon = (min(lon_labels)[0], max(lon_labels)[0]) if lon_labels else (0, 0)

    lat_range = list(range(min_lat, max_lat + 1))
    lon_range = list(range(min_lon, max_lon + 1))

    lat_labels = sorted(lat_labels, key=lambda x: x[1], reverse=True)
    lon_labels = sorted(lon_labels, key=lambda x: x[1])

    all_lat_labels = []
    for i in lat_range:
        label = ""
        quotient = i
        while quotient >= 0:
            label = chr((quotient % 26) + ord('A')) + label
            quotient = quotient // 26 - 1
        all_lat_labels.append(label)

    lon_labels_dict = dict(lon_labels)

    lat_tick_labels = all_lat_labels
    lon_tick_labels = [lon_labels_dict.get(i, str(i + 1)) for i in lon_range]

    heatmap_matrix = np.full((len(lat_range), len(lon_range)), np.nan)

    for (lat_idx, lon_idx), acc in grid_data.items():
        heatmap_matrix[lat_range.index(lat_idx), lon_range.index(lon_idx)] = acc

    # Plotting heatmap
    plt.figure(figsize=(12, 8))

    heatmap_matrix = heatmap_matrix[::-1]

    ax = sns.heatmap(heatmap_matrix, fmt='.2f', cmap="coolwarm_r", linewidths=0.5, cbar_kws={'label': 'Accuracy'})

    plt.title("Pickup Zone Accuracy Heatmap")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    # Fixing tickmarks
    ax.set_xticks(np.arange(len(lon_range)))
    ax.set_xticklabels(lon_tick_labels, rotation=45)

    ax.set_yticks(np.arange(len(lat_range)))
    ax.set_yticklabels(lat_tick_labels[::-1], rotation=0)

    plt.tight_layout()

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def create_dataloader(x, y, batch_size=64, shuffle=True):
    tensor_x = torch.tensor(x, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def train(model, optimizer, criterion, train_loader):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for data in train_loader:
        inputs, targets = data
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

    accuracy = correct / total
    return train_loss / len(train_loader), accuracy

def evaluate(model, criterion, test_loader, label_encoder=None):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data in test_loader:
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    accuracy = correct / total

    if label_encoder is None:
        return test_loss / len(test_loader), accuracy

    # Decoding and computing accuracy for each class
    actual_labels = label_encoder.inverse_transform(all_targets)
    predicted_labels = label_encoder.inverse_transform(all_preds)

    unique_labels = np.unique(all_targets)
    class_accuracies = {
        label_encoder.inverse_transform([label])[0]: 
        (np.array(all_preds)[all_targets == label] == label).sum() / (all_targets == label).sum() 
        for label in unique_labels}

    return test_loss / len(test_loader), accuracy, class_accuracies, actual_labels, predicted_labels

def tune_hyperparams(x_train_val, y_train_val, epochs, k_folds, num_classes):
    learning_rates = [0.05, 0.01, 0.005, 0.001]
    batch_sizes = [16, 32, 64]
    w_decays = [0.05, 0.1, 0.5, 0]
    param_combinations = list(itertools.product(learning_rates, batch_sizes, w_decays))

    best_params = None
    best_val_acc = 0

    for lr, batch_size, w_decay in param_combinations:
        print(f"\nTesting Hyperparameters: LR={lr}, Batch Size={batch_size}, Weight Decay={w_decay}")

        kf = KFold(n_splits=k_folds, shuffle=True, random_state=400)
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(x_train_val)):
            x_train_fold, x_val_fold = x_train_val[train_idx], x_train_val[val_idx]
            y_train_fold, y_val_fold = y_train_val[train_idx], y_train_val[val_idx]

            train_loader = create_dataloader(x_train_fold, y_train_fold, batch_size=batch_size, shuffle=True)
            val_loader = create_dataloader(x_val_fold, y_val_fold, batch_size=batch_size, shuffle=False)

            model = PickupZoneClassifier(x_train_val.shape[1], num_classes).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=w_decay)
            criterion = torch.nn.CrossEntropyLoss()

            best_fold_acc = 0

            for epoch in range(epochs):
                train_loss, train_acc = train(model, optimizer, criterion, train_loader)
                val_loss, val_acc = evaluate(model, criterion, val_loader)

                if val_acc > best_fold_acc:
                    best_fold_acc = val_acc

            fold_results.append(best_fold_acc)

        avg_val_acc = np.mean(fold_results)
        print(f"Avg Validation Accuracy for LR={lr}, Batch Size={batch_size}, Weight Decay={w_decay}: {avg_val_acc:.4f}")

        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            best_params = (lr, batch_size, w_decay)

    print(f"\nBest Hyperparameters: LR={best_params[0]}, Batch Size={best_params[1]}, Weight Decay={best_params[2]}, Validation Accuracy={best_val_acc:.4f}")
    return best_params

def train_model():
    """
    Main function to train and validate the classification model.
    """
    label_encoder = LabelEncoder()

    x_train, y_train = load_data("", label_encoder=None)
    label_encoder.fit(y_train)
    x_train, y_train = load_data("", label_encoder=label_encoder)
    x_test, y_test = load_data("", label_encoder=label_encoder)

    num_classes = len(set(y_train))

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    # Hyperparamer tuning
    best_lr, best_batch_size, w_decay = tune_hyperparams(x_train, y_train, epochs = 10, k_folds = 5, num_classes=num_classes)

    # Early stopping and regular training
    train_loader = create_dataloader(x_train, y_train, batch_size=best_batch_size, shuffle=True)
    val_loader = create_dataloader(x_val, y_val, batch_size=best_batch_size, shuffle=True)

    model = PickupZoneClassifier(x_train.shape[1], num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_lr, weight_decay=w_decay)
    criterion = torch.nn.CrossEntropyLoss()

    # Early stopping implementation
    patience = 3
    best_val_acc = 0
    counter = 0

    epochs = 10
    for epoch in range(epochs):
        train_loss, train_acc = train(model, optimizer, criterion, train_loader)
        val_loss, val_acc = evaluate(model, criterion, val_loader)
        print(f"Epoch {epoch+1} | Train Accuracy: {train_acc:.4f} | Val Accuracy: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print(f"Stopped early at epoch {epoch+1}")
            break

    test_loader = create_dataloader(x_test, y_test, batch_size=best_batch_size, shuffle=False)
    test_loss, test_acc, class_accuracies, actual_labels, predicted_labels = evaluate(model, criterion, test_loader, label_encoder)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
    print("\nPer-Class Accuracy:")
    for class_name, acc in class_accuracies.items():
        print(f"{class_name}: {acc:.4f}")

    plot_heatmap(class_accuracies, 'pickup_zone_classifier_heatmap.png')

    torch.save(model, "pickup_zone_classifier.pt")
    torch.save(model, "pickup_zone_classifier.pth")
