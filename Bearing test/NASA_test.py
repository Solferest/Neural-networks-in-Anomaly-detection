import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime

# 1. Set dataset path (modify this to your actual path)
BASE_PATH = r"C:/Users/LENOVO/PycharmProjects/Bearing test/IMS-Rexnord Bearing Data"
#BASE_PATH = r"D:/Документы/мага/сем2/NNAD/Neural-networks-in-Anomaly-detection/Bearing test/IMS-Rexnord Bearing Data"
TEST_FOLDERS = ["1st_test", "2nd_test", "3rd_test"]

# 2. Function to process a single vibration data file (unchanged)
def load_bearing_data(file_path):
    try:
        data = pd.read_csv(file_path, header=None, names=['vibration'])

        features = {
            'mean': np.mean(data['vibration']),
            'std': np.std(data['vibration']),
            'skewness': stats.skew(data['vibration']),
            'kurtosis': stats.kurtosis(data['vibration']),
            'rms': np.sqrt(np.mean(data['vibration'] ** 2)),
            'peak_to_peak': np.ptp(data['vibration']),
            'crest_factor': np.max(np.abs(data['vibration'])) / np.sqrt(np.mean(data['vibration'] ** 2)),
            'shape_factor': np.sqrt(np.mean(data['vibration'] ** 2)) / np.mean(np.abs(data['vibration']))
        }
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

# 3. Parse timestamp from filename (unchanged)
def parse_timestamp(filename):
    try:
        # Extract timestamp parts from filename (format: YEAR.MONTH.DAY.HOUR.MINUTE.SECOND.csv)
        parts = filename.split('.')[:6]  # Ignore the .csv extension
        dt = datetime(
            year=int(parts[0]),
            month=int(parts[1]),
            day=int(parts[2]),
            hour=int(parts[3]),
            minute=int(parts[4]),
            second=int(parts[5])
        )
        return dt
    except Exception as e:
        print(f"Error parsing timestamp from {filename}: {str(e)}")
        return None

# 4. Build the full dataset from all bearing files
def create_dataset(base_path, test_folders):
    features_list = []
    labels = []

    for test_folder in test_folders:
        test_path = os.path.join(base_path, test_folder, test_folder)  # e.g., .../1st test/1st test
        
        if not os.path.exists(test_path):
            print(f"Warning: Test path not found - {test_path}")
            continue

        try:
            # Get all data files and sort by timestamp
            files = [f for f in os.listdir(test_path)]
            files_with_times = []

            for f in files:
                dt = parse_timestamp(f)
                if dt:
                    files_with_times.append((dt, f))

            # Sort files chronologically
            files_with_times.sort()
            sorted_files = [f[1] for f in files_with_times]

            if not sorted_files:
                print(f"No valid CSV files found in {test_path}")
                continue

            # Determine failure point (last 10% of files considered faulty)
            failure_point = int(len(sorted_files) * 0.9)

            for i, file in enumerate(sorted_files):
                file_path = os.path.join(test_path, file)
                features = load_bearing_data(file_path)
                if features is None:
                    continue

                features['test_folder'] = test_folder  # Track which test folder this came from
                features['timestamp'] = files_with_times[i][0]
                label = 0 if i < failure_point else 1  # 0=normal, 1=anomaly
                labels.append(label)
                features_list.append(features)

        except Exception as e:
            print(f"Error processing test folder {test_folder}: {str(e)}")

    if not features_list:
        print("Error: No bearing data was processed")
        return None

    df = pd.DataFrame(features_list)
    df['label'] = labels
    return df

# 5. Train and evaluate the model (unchanged)
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    print("\nTraining Accuracy:", accuracy_score(y_train, train_pred))
    print("Test Accuracy:", accuracy_score(y_test, test_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, test_pred))

    return model, scaler

# 6. Plot feature importance (unchanged)
def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance")
    plt.bar(range(len(feature_names)), importances[indices], align="center")
    plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.show()

# 7. Main function to run the full pipeline
def main():
    print("=== NASA Bearing Anomaly Detection ===")

    # Step 1: Load and process data
    print("\nLoading and processing vibration data...")
    dataset = create_dataset(BASE_PATH, TEST_FOLDERS)

    if dataset is None:
        print("Failed to load dataset. Please check the path and files.")
        return

    print("\nFirst 5 samples:")
    print(dataset.head())
    print("\nNormal (0) vs Anomaly (1) counts:")
    print(dataset['label'].value_counts())

    # Step 2: Prepare features and labels
    feature_cols = ['mean', 'std', 'skewness', 'kurtosis', 'rms', 'peak_to_peak', 'crest_factor', 'shape_factor']
    X = dataset[feature_cols]
    y = dataset['label']

    # Step 3: Train model
    print("\nTraining model...")
    model, scaler = train_model(X, y)

    # Step 4: Show important features
    print("\nPlotting feature importance...")
    plot_feature_importance(model, feature_cols)

    print("\n=== System Ready for Anomaly Detection ===")


if __name__ == "__main__":
    main()
