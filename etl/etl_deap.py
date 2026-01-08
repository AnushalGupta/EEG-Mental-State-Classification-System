import os
import glob
import numpy as np
import pickle 
from scipy.io import loadmat
from scipy.signal import welch, butter, filtfilt
import config
from db_utils import get_db_connection, create_table_if_not_exists

# --- CONFIGURATION (Frozen) ---
FS = config.FS
EPOCH_SEC = config.EPOCH_SEC
BANDS = config.BANDS
DATASET_NAME = "DEAP"
FOLDER_PATH = "DEAP_Data" # Expecting a folder

# Strict Feature vector dimensionality: 14 channels
CHANNELS_TO_USE = 14 

# --- FUNCTIONS ---

def bandpass(x, low=0.5, high=40, order=4):
    nyq = FS / 2
    b, a = butter(order, [low/nyq, high/nyq], btype="band")
    if len(x) <= 256: 
        return x
    return filtfilt(b, a, x)

def band_power(sig, band):
    fmin, fmax = band
    freqs, psd = welch(sig, FS, nperseg=min(256, len(sig)))
    idx = (freqs >= fmin) & (freqs <= fmax)
    return np.trapz(psd[idx], freqs[idx])

def map_label(arousal_score):
    """
    Maps DEAP Arousal (1-9) to Project Labels.
    Rule from Phase 2 Prompt:
      0 -> Focused (High Arousal / alert-engaged)
      1 -> Unfocused (Mid Arousal / transitional)
      2 -> Drowsy (Low Arousal / passive)
    """
    # Thresholds (Subject to tuning, but fixing for now)
    # High > 5, Low < 3, Mid 3-5
    if arousal_score > 5:
        return 0 # Focused
    elif arousal_score < 3:
        return 2 # Drowsy
    else:
        return 1 # Unfocused

def extract_features_from_epoch(epoch, ch_count):
    feat = {}
    for ch in range(ch_count):
        sig = bandpass(epoch[ch])
        for name, band in BANDS.items():
            feat[f"ch{ch+1}_{name}"] = float(band_power(sig, band))
    return feat

def get_feature_names(n_channels=14):
    names = []
    for ch in range(n_channels):
        for band in BANDS.keys():
            names.append(f"ch{ch+1}_{band}")
    return names

def process_file(f, feature_names):
    fname = os.path.basename(f)
    print(f"Processing {fname}...")
    
    try:
        if f.endswith('.dat'):
            with open(f, 'rb') as file:
                # DEAP .dat (python) files are usually dictionaries
                content = pickle.load(file, encoding='latin1')
                data = content['data']
                labels = content['labels']
        else:
            # Fallback to .mat if needed
            mat = loadmat(f)
            if 'data' not in mat or 'labels' not in mat:
                 print(f"Skipping {fname}: Invalid DEAP .mat format.")
                 return []
            data = mat['data']
            labels = mat['labels']
        
        # data shape: trials x channels x samples
        n_trials, n_channels, n_samples_total = data.shape
        
        batch_data = []
        win = config.EPOCH_SAMPLES
        
        for trial_idx in range(n_trials):
            # 1. Get Labels
            # labels: [valence, arousal, dominance, liking]
            arousal = labels[trial_idx, 1] 
            mapped_label = map_label(arousal)
            
            # 2. Get Data & Channel Selection
            trial_data = data[trial_idx, :CHANNELS_TO_USE, :] 
            
            # 3. Epoching
            # DEAP (python) is 40 x 40 x 8064 (63s * 128Hz)
            # Preprocessed data usually has baseline removed (3s). 
            # If 8064, it includes 3s baseline. We might want to skip it?
            # For now, processing uniformly.
            for start in range(0, n_samples_total - win, win):
                epoch = trial_data[:, start:start+win]
                
                # 4. Feature Extraction
                feats = extract_features_from_epoch(epoch, CHANNELS_TO_USE)
                
                row = {
                    "dataset_name": DATASET_NAME,
                    "subject_id": f"{fname}_t{trial_idx}", 
                    "label": mapped_label,
                    **feats
                }
                batch_data.append(row)
                
        return batch_data

    except Exception as e:
        print(f"Error processing {fname}: {e}")
        return []

def run_etl():
    print(f"--- Starting DEAP ETL ---")
    print(f"Looking for files in: {FOLDER_PATH}")
    
    feature_names = get_feature_names(CHANNELS_TO_USE)
    create_table_if_not_exists(feature_names)
    
    # Look for .dat and .mat
    files = sorted(glob.glob(os.path.join(FOLDER_PATH, "*.dat")) + glob.glob(os.path.join(FOLDER_PATH, "*.mat")))
    print(f"Found {len(files)} files.")
    
    if not files:
        print("No files found. Please create 'DEAP_Data' folder and add .mat files.")
        return

    conn = get_db_connection()
    cursor = conn.cursor()
    total_inserted = 0

    for f in files:
        batch_data = process_file(f, feature_names)
        
        if not batch_data:
            continue
            
        # Bulk Insert
        cols = ["dataset_name", "subject_id", "label"] + feature_names
        placeholders = ", ".join(["%s"] * len(cols))
        columns_str = ", ".join([f"`{c}`" for c in cols])
        sql = f"INSERT INTO eeg_features ({columns_str}) VALUES ({placeholders})"
        
        val_list = []
        for row in batch_data:
            val_list.append(tuple(row[c] for c in cols))
            
        cursor.executemany(sql, val_list)
        conn.commit()
        print(f"  -> Inserted {len(batch_data)} epochs.")
        total_inserted += len(batch_data)

    cursor.close()
    conn.close()
    print(f"DEAP ETL Complete. Total records: {total_inserted}")

if __name__ == "__main__":
    run_etl()
