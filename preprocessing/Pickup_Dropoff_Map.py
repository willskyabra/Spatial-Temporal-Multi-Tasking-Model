import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

file_path = ""
data = pd.read_csv(file_path)

pickup_counts = data["pickup_zone"].value_counts()
dropoff_counts = data["dropoff_zone"].value_counts()

print("Pickup Zone Counts:")
print(pickup_counts.sort_index().to_string())
print("\nDropoff Zone Counts:")
print(dropoff_counts.sort_index().to_string())

def process_zone_counts(zone_counts):
    grid_data = {}
    lat_labels = set()
    lon_labels = set()
    
    for zone, count in zone_counts.items():
        lat_part = "".join(filter(str.isalpha, zone))
        lon_part = "".join(filter(str.isdigit, zone))
        
        if lat_part and lon_part:
            lat_idx = sum((ord(char) - ord('A') + 1) * (26**i) for i, char in enumerate(reversed(lat_part))) - 1
            lon_idx = int(lon_part) - 1
            grid_data[(lat_idx, lon_idx)] = count
            lat_labels.add((lat_idx, lat_part))
            lon_labels.add((lon_idx, lon_part))
    
    # Ensure continuous lat/lon ranges
    min_lat, max_lat = (min(lat_labels)[0], max(lat_labels)[0]) if lat_labels else (0, 0)
    min_lon, max_lon = (min(lon_labels)[0], max(lon_labels)[0]) if lon_labels else (0, 0)

    lat_range = list(range(min_lat, max_lat + 1))
    lon_range = list(range(min_lon, max_lon + 1))

    # Sort labels for latitude in reverse order
    lat_labels = sorted(lat_labels, key=lambda x: x[1], reverse=True)  # Reverse sorting for latitudes
    lon_labels = sorted(lon_labels, key=lambda x: x[1])  # Standard sorting for longitude

    # Fill missing latitudes between the minimum and maximum values
    all_lat_labels = []
    for i in lat_range:
        label = ""
        quotient = i
        while quotient >= 0:
            label = chr((quotient % 26) + ord('A')) + label
            quotient = quotient // 26 - 1
        all_lat_labels.append(label)
    
    # Create full labels ensuring all possible values are included
    lon_labels_dict = dict(lon_labels)

    # Create the tick labels for both latitudes and longitudes
    lat_tick_labels = all_lat_labels
    lon_tick_labels = [lon_labels_dict.get(i, str(i + 1)) for i in lon_range]  # Default lon labels to numbers

    # Create heatmap matrix with NaNs
    heatmap_matrix = np.full((len(lat_range), len(lon_range)), np.nan)

    for (lat_idx, lon_idx), count in grid_data.items():
        heatmap_matrix[lat_range.index(lat_idx), lon_range.index(lon_idx)] = count

    return heatmap_matrix, lon_range, lon_tick_labels, lat_range, lat_tick_labels



def plot_heatmap(heatmap_matrix, lon_indices, lon_tick_labels, lat_indices, lat_tick_labels, title, filename):
    plt.figure(figsize=(12, 8))

    # Reverse the rows in the heatmap matrix to match the reversed latitudes
    heatmap_matrix = heatmap_matrix[::-1]

    ax = sns.heatmap(heatmap_matrix, fmt='.0f', cmap="coolwarm_r", linewidths=0.5, cbar_kws={'label': 'Total Count'})

    plt.title(title)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    # Set correct labels on the tick marks
    ax.set_xticks(np.arange(len(lon_indices)))
    ax.set_xticklabels(lon_tick_labels, rotation=45)

    ax.set_yticks(np.arange(len(lat_indices)))
    ax.set_yticklabels(lat_tick_labels[::-1], rotation=0)  # Reverse the order of y-tick labels

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


pickup_heatmap, lon_idx_p, lon_labels_p, lat_idx_p, lat_labels_p = process_zone_counts(pickup_counts)
dropoff_heatmap, lon_idx_d, lon_labels_d, lat_idx_d, lat_labels_d = process_zone_counts(dropoff_counts)

plot_heatmap(pickup_heatmap, lon_idx_p, lon_labels_p, lat_idx_p, lat_labels_p, "Pickup Zone Count Heatmap", "pickup_heatmap.png")
plot_heatmap(dropoff_heatmap, lon_idx_d, lon_labels_d, lat_idx_d, lat_labels_d, "Dropoff Zone Count Heatmap", "dropoff_heatmap.png")

