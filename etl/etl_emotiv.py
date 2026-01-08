import os
import glob
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import welch, butter, filtfilt
import config
from db_utils import get_db_connection, create_table_if_not_exists

# --- CONFIGURATION (Loaded from config.py) ---
FS = config.FS
EEG_COL_START = 3
EEG_COL_END = 16 # 14 channels
EPOCH_SEC = config.EPOCH_SEC
FOLDER_PATH = "EMOTIV_Data"  # Updated folder
DATASET_NAME = "EMOTIV"

BANDS = config.BANDS

# --- PROCESS FUNCTIONS ---

def bandpass(x, low=0.5, high=40, order=4):
    nyq = FS / 2
    b, a = butter(order, [low/nyq, high/nyq], btype="band")
    if len(x) <= 256: # Avoid filtering very short signals
        return x
    return filtfilt(b, a, x)

def band_power(sig, band):
    fmin, fmax = band
    freqs, psd = welch(sig, FS, nperseg=min(256, len(sig)))
    idx = (freqs >= fmin) & (freqs <= fmax)
    return np.trapz(psd[idx], freqs[idx])

def load_eeg(path):
    mat = loadmat(path)
    data = mat["o"]["data"][0][0]
    eeg = data[:, EEG_COL_START:EEG_COL_END+1] # 14 channels
    return eeg.T  # (channels, samples)

def slice_with_labels(eeg):
    samples = eeg.shape[1]
    s1 = min(10 * 60 * FS, samples)
    s2 = min(20 * 60 * FS, samples)

    segments = [
        (eeg[:, :s1], 0),      # Focused
        (eeg[:, s1:s2], 1),    # Unfocused
        (eeg[:, s2:], 2)       # Drowsy
    ]
    return segments

def extract_features_from_epoch(epoch, ch_count):
    feat = {}
    for ch in range(ch_count):
        sig = bandpass(epoch[ch])
        for name, band in BANDS.items():
            feat[f"ch{ch+1}_{name}"] = float(band_power(sig, band)) # float for JSON/SQL compatibility
    return feat

def get_feature_names(n_channels=14):
    names = []
    for ch in range(n_channels):
        for band in BANDS.keys():
            names.append(f"ch{ch+1}_{band}")
    return names

def run_etl():
    # 1. Setup Database
    print("Setting up database...")
    feature_names = get_feature_names()
    create_table_if_not_exists(feature_names)

    # 2. Process Files
    files = sorted(glob.glob("*.mat"))
    print(f"Found {len(files)} .mat files.")

    conn = get_db_connection()
    cursor = conn.cursor()

    total_inserted = 0

    for f in files:
        fname = os.path.basename(f)
        print(f"Processing {fname}...")
        
        try:
            eeg = load_eeg(f)
            n_channels = eeg.shape[0]
            
            # Sanity check channel count
            if n_channels != 14:
                print(f"Skipping {fname}: Expected 14 channels, got {n_channels}")
                continue

            segments = slice_with_labels(eeg)
            
            win = EPOCH_SEC * FS
            batch_data = []

            print(f"DEBUG: EEG shape: {eeg.shape}")
            print(f"DEBUG: Segments count: {len(segments)}")
            
            for seg_data, label in segments:
                n_samples_seg = seg_data.shape[1]
                print(f"DEBUG: Seg label {label}, samples {n_samples_seg}")

                n_samples = seg_data.shape[1]
                # Epoching
                for start in range(0, n_samples - win, win):
                    epoch = seg_data[:, start:start+win]
                    feats = extract_features_from_epoch(epoch, n_channels)
                    
                    # Prepare row for SQL
                    row = {
                        "dataset_name": DATASET_NAME,
                        "subject_id": fname,
                        "label": label,
                        **feats # unpack feature columns
                    }
                    batch_data.append(row)
            
            if not batch_data:
                continue

            # 3. Bulk Insert
            # Construct INSERT statement dynamically
            cols = ["dataset_name", "subject_id", "label"] + feature_names
            placeholders = ", ".join(["%s"] * len(cols))
            columns_str = ", ".join([f"`{c}`" for c in cols])
            
            sql = f"INSERT INTO eeg_features ({columns_str}) VALUES ({placeholders})"
            
            val_list = []
            for row in batch_data:
                # Ensure order matches 'cols'
                val_list.append(tuple(row[c] for c in cols))

            cursor.executemany(sql, val_list)
            conn.commit()
            print(f"  -> Inserted {len(batch_data)} epochs.")
            total_inserted += len(batch_data)

        except Exception as e:
            print(f"Error processing {fname}: {e}")

    cursor.close()
    conn.close()
    print(f"ETL Complete. Total epochs stored: {total_inserted}")

if __name__ == "__main__":
    print("Script started.")
    run_etl()
