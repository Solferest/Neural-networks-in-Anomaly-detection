import pandas as pd
import numpy as np
from scipy import stats, fft
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set random seed for reproducibility
np.random.seed(42)

# Define paths
DATA_ROOT = Path("XJTU-SY_Bearing_Datasets/XJTU-SY_Bearing_Datasets")  # Adjusted for the nested structure
PLOT_DIR = Path("plots/")
PLOT_DIR.mkdir(exist_ok=True)

# Operating conditions and their corresponding bearings
OPERATING_CONDITIONS = {
    "35Hz12kN": [f"Bearing1_{i}" for i in range(1, 6)],
    "37.5Hz11kN": [f"Bearing2_{i}" for i in range(1, 6)],
    "40Hz10kN": [f"Bearing3_{i}" for i in range(1, 6)],
}

# Number of CSV files per bearing (from PDF)
BEARING_FILE_COUNTS = {
    "Bearing1_1": 123, "Bearing1_2": 161, "Bearing1_3": 158, "Bearing1_4": 122, "Bearing1_5": 52,
    "Bearing2_1": 491, "Bearing2_2": 161, "Bearing2_3": 533, "Bearing2_4": 42, "Bearing2_5": 339,
    "Bearing3_1": 253, "Bearing3_2": 2496, "Bearing3_3": 371, "Bearing3_4": 1515, "Bearing3_5": 114
}


# Function to extract advanced features from a vibration signal
def extract_features(signal, sampling_rate=25600):
    signal = np.asarray(signal, dtype=np.float64)
    if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

    features = {
        'mean': np.mean(signal),
        'std': np.std(signal),
        'rms': np.sqrt(np.mean(signal ** 2)),
        'peak': np.max(np.abs(signal)),
        'peak_to_peak': np.ptp(signal),
        'kurtosis': stats.kurtosis(signal) if len(signal) > 3 else 0,
        'skewness': stats.skew(signal) if len(signal) > 2 else 0,
        'crest_factor': np.max(np.abs(signal)) / np.sqrt(np.mean(signal ** 2)) if np.mean(signal ** 2) > 0 else 0,
        'shape_factor': np.sqrt(np.mean(signal ** 2)) / np.mean(np.abs(signal)) if np.mean(np.abs(signal)) > 0 else 0
    }

    n = len(signal)
    if n > 0:
        freq_signal = fft.fft(signal)
        freq_amplitude = np.abs(freq_signal)[:n // 2] / n
        freqs = fft.fftfreq(n, 1 / sampling_rate)[:n // 2]

        dominant_idx = np.argmax(freq_amplitude)
        features['dominant_freq'] = freqs[dominant_idx]
        features['dominant_amplitude'] = freq_amplitude[dominant_idx]
        features['spectral_energy'] = np.sum(freq_amplitude ** 2)
        features['spectral_centroid'] = np.sum(freqs * freq_amplitude) / np.sum(freq_amplitude) if np.sum(
            freq_amplitude) > 0 else 0
        features['spectral_kurtosis'] = stats.kurtosis(freq_amplitude) if len(freq_amplitude) > 3 else 0
    else:
        for key in ['dominant_freq', 'dominant_amplitude', 'spectral_energy', 'spectral_centroid', 'spectral_kurtosis']:
            features[key] = 0

    return features


# Load and preprocess data from the extracted directory
def load_bearing_data(bearing_dir, bearing_name):
    feature_list = []
    timestamps = []
    file_indices = []
    num_files = BEARING_FILE_COUNTS[bearing_name]

    for file_idx in range(1, num_files + 1):
        csv_file = bearing_dir / f"{file_idx}.csv"
        if not csv_file.exists():
            print(f"Warning: {csv_file} not found. Skipping...")
            continue

        try:
            # Read CSV with possible header
            df = pd.read_csv(csv_file, header=None,
                             names=['Horizontal_vibration_signals', 'Vertical_vibration_signals'], skiprows=0)

            # Check if the first row looks like a header by trying to convert to numeric
            first_row = df.iloc[0]
            if not pd.to_numeric(first_row, errors='coerce').notna().all():
                print(f"Detected header in {csv_file}. Skipping first row...")
                df = pd.read_csv(csv_file, header=0,
                                 names=['Horizontal_vibration_signals', 'Vertical_vibration_signals'])

            # Validate and convert columns to numeric
            for col in ['Horizontal_vibration_signals', 'Vertical_vibration_signals']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].isna().all():
                    print(f"Warning: All values in {col} of {csv_file} are non-numeric or NaN. Skipping file...")
                    continue
                df[col] = df[col].fillna(0)

            # Check row count
            expected_length = 32768
            actual_length = len(df)
            if actual_length != expected_length:
                print(
                    f"Warning: {csv_file} has {actual_length} rows instead of expected {expected_length}. Proceeding with available data...")

            h_features = extract_features(df['Horizontal_vibration_signals'].values)
            v_features = extract_features(df['Vertical_vibration_signals'].values)

            combined_features = {f"h_{k}": v for k, v in h_features.items()}
            combined_features.update({f"v_{k}": v for k, v in v_features.items()})

            feature_list.append(combined_features)
            timestamps.append(file_idx * 60)
            file_indices.append(file_idx)
            print(f"Processed file {file_idx} for {bearing_name} successfully.")
        except pd.errors.ParserError as e:
            print(f"Error parsing {csv_file}: {e}. Skipping...")
            continue
        except Exception as e:
            print(f"Unexpected error processing {csv_file}: {e}. Skipping...")
            continue

    if not feature_list:
        print(f"No features extracted for {bearing_name}. Returning empty DataFrame.")
        return pd.DataFrame()

    features_df = pd.DataFrame(feature_list)
    features_df['timestamp'] = timestamps
    features_df['file_idx'] = file_indices
    features_df['bearing'] = bearing_name
    print(f"Features extracted for {bearing_name}: {len(features_df)} files processed.")
    return features_df


# Anomaly detection
def detect_anomalies(features_df, contamination=0.1):
    feature_cols = [col for col in features_df.columns if col not in ['timestamp', 'file_idx', 'bearing']]
    X = features_df[feature_cols].values

    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    anomaly_labels = iso_forest.fit_predict(X)

    h_rms_mean, h_rms_std = features_df['h_rms'].mean(), features_df['h_rms'].std()
    v_rms_mean, v_rms_std = features_df['v_rms'].mean(), features_df['v_rms'].std()
    statistical_anomaly = (features_df['h_rms'] > h_rms_mean + 3 * h_rms_std) | \
                          (features_df['v_rms'] > v_rms_mean + 3 * v_rms_std)

    features_df['anomaly'] = anomaly_labels
    features_df['anomaly'] = features_df['anomaly'].map({1: 'Normal', -1: 'Anomaly'})
    features_df['statistical_anomaly'] = statistical_anomaly.map({True: 'Anomaly', False: 'Normal'})
    features_df['final_anomaly'] = np.where(
        (features_df['anomaly'] == 'Anomaly') | (features_df['statistical_anomaly'] == 'Anomaly'),
        'Anomaly', 'Normal'
    )
    print(f"Anomaly detection completed for {features_df['bearing'].iloc[0]}: {len(features_df)} samples.")
    return features_df


# Plotting functions
def plot_vibration_signals(bearing_dir, bearing_name, features_df, max_files=3):
    if features_df.empty:
        print(f"Cannot plot vibration signals for {bearing_name}: No data available.")
        return

    plt.figure(figsize=(15, 10))
    selected_files = features_df['file_idx'].iloc[:max_files]

    for i, file_idx in enumerate(selected_files):
        csv_file = bearing_dir / f"{file_idx}.csv"
        df = pd.read_csv(csv_file, header=None, names=['Horizontal_vibration_signals', 'Vertical_vibration_signals'])
        # Check for header in plotting as well
        first_row = df.iloc[0]
        if not pd.to_numeric(first_row, errors='coerce').notna().all():
            df = pd.read_csv(csv_file, header=0, names=['Horizontal_vibration_signals', 'Vertical_vibration_signals'])

        anomaly_label = features_df[features_df['file_idx'] == file_idx]['final_anomaly'].iloc[0]

        plt.subplot(max_files, 2, 2 * i + 1)
        plt.plot(df['Horizontal_vibration_signals'], label='Horizontal', color='blue')
        plt.title(f"Horizontal Vibration - {bearing_name} File {file_idx} ({anomaly_label})")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.legend()

        plt.subplot(max_files, 2, 2 * i + 2)
        plt.plot(df['Vertical_vibration_signals'], label='Vertical', color='orange')
        plt.title(f"Vertical Vibration - {bearing_name} File {file_idx} ({anomaly_label})")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.legend()

    plt.tight_layout()
    plot_path = PLOT_DIR / f"vibration_signals_{bearing_name}.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved vibration signals plot: {plot_path}")


def plot_anomaly_timeline(features_df, bearing_name):
    if features_df.empty:
        print(f"Cannot plot anomaly timeline for {bearing_name}: No data available.")
        return

    plt.figure(figsize=(15, 6))
    sns.scatterplot(x='timestamp', y='h_rms', hue='final_anomaly', style='final_anomaly',
                    palette={'Normal': 'blue', 'Anomaly': 'red'}, s=100, data=features_df)
    plt.title(f"Anomaly Detection Timeline (Horizontal RMS) - {bearing_name}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Horizontal RMS")
    plt.legend(title="Status")
    plot_path = PLOT_DIR / f"anomaly_timeline_{bearing_name}.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved anomaly timeline plot: {plot_path}")


def plot_feature_distributions(features_df, bearing_name):
    if features_df.empty:
        print(f"Cannot plot feature distributions for {bearing_name}: No data available.")
        return

    key_features = ['h_rms', 'h_kurtosis', 'v_rms', 'v_kurtosis']
    plt.figure(figsize=(15, 10))

    for i, feature in enumerate(key_features, 1):
        plt.subplot(2, 2, i)
        sns.histplot(data=features_df, x=feature, hue='final_anomaly', multiple='stack',
                     palette={'Normal': 'blue', 'Anomaly': 'red'})
        plt.title(f"Distribution of {feature} - {bearing_name}")
        plt.xlabel(feature)
        plt.ylabel("Count")

    plt.tight_layout()
    plot_path = PLOT_DIR / f"feature_distributions_{bearing_name}.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved feature distributions plot: {plot_path}")


def plot_spectral_analysis(bearing_dir, bearing_name, features_df, max_files=3):
    if features_df.empty:
        print(f"Cannot plot spectral analysis for {bearing_name}: No data available.")
        return

    plt.figure(figsize=(15, 10))
    selected_files = features_df['file_idx'].iloc[:max_files]
    sampling_rate = 25600

    for i, file_idx in enumerate(selected_files):
        csv_file = bearing_dir / f"{file_idx}.csv"
        df = pd.read_csv(csv_file, header=None, names=['Horizontal_vibration_signals', 'Vertical_vibration_signals'])
        # Check for header in plotting as well
        first_row = df.iloc[0]
        if not pd.to_numeric(first_row, errors='coerce').notna().all():
            df = pd.read_csv(csv_file, header=0, names=['Horizontal_vibration_signals', 'Vertical_vibration_signals'])

        anomaly_label = features_df[features_df['file_idx'] == file_idx]['final_anomaly'].iloc[0]

        signal = df['Horizontal_vibration_signals'].values
        n = len(signal)
        freq_signal = fft.fft(signal)
        freq_amplitude = np.abs(freq_signal)[:n // 2] / n
        freqs = fft.fftfreq(n, 1 / sampling_rate)[:n // 2]

        plt.subplot(max_files, 1, i + 1)
        plt.plot(freqs, freq_amplitude, color='purple')
        plt.title(f"Frequency Spectrum - {bearing_name} File {file_idx} ({anomaly_label})")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.xlim(0, 5000)

    plt.tight_layout()
    plot_path = PLOT_DIR / f"spectral_analysis_{bearing_name}.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved spectral analysis plot: {plot_path}")


def generate_summary(all_features_df):
    if all_features_df.empty:
        print("No data available for summary.")
        return

    summary = all_features_df.groupby(['bearing', 'final_anomaly']).size().unstack(fill_value=0)
    summary['Total'] = summary['Normal'] + summary['Anomaly']
    summary['Anomaly_Rate'] = summary['Anomaly'] / summary['Total']
    summary['Operating_Condition'] = summary.index.map(
        lambda x: '35Hz12kN' if x.startswith('Bearing1') else '37.5Hz11kN' if x.startswith('Bearing2') else '40Hz10kN'
    )
    print("\nAnomaly Detection Summary:")
    print(summary)
    summary.to_csv(PLOT_DIR / "anomaly_summary.csv")
    print(f"Saved summary to {PLOT_DIR / 'anomaly_summary.csv'}")


def main():
    all_features_df = pd.DataFrame()

    # Check if data root exists
    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            f"Directory {DATA_ROOT} not found. Please ensure the extracted folder structure is correct.")

    # Iterate over operating conditions and bearings
    for condition, bearings in OPERATING_CONDITIONS.items():
        condition_dir = DATA_ROOT / condition
        if not condition_dir.exists():
            print(f"Warning: {condition_dir} not found. Skipping...")
            continue
        print(f"\nProcessing operating condition: {condition}")

        for bearing in bearings:
            bearing_dir = condition_dir / bearing
            if not bearing_dir.exists():
                print(f"Warning: {bearing_dir} not found. Skipping...")
                continue
            print(f"Processing bearing: {bearing}")

            features_df = load_bearing_data(bearing_dir, bearing)
            if features_df.empty:
                print(f"No data found for {bearing}. Skipping...")
                continue

            features_df = detect_anomalies(features_df)
            all_features_df = pd.concat([all_features_df, features_df], ignore_index=True)

            # Generate plots
            plot_vibration_signals(bearing_dir, bearing, features_df)
            plot_anomaly_timeline(features_df, bearing)
            plot_feature_distributions(features_df, bearing)
            plot_spectral_analysis(bearing_dir, bearing, features_df)
            print(f"Plots for {bearing} should be saved in {PLOT_DIR}")

    if not all_features_df.empty:
        generate_summary(all_features_df)
    else:
        print("No data processed. Check directory structure and file availability.")


if __name__ == "__main__":
    main()