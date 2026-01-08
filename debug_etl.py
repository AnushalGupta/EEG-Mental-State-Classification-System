import os
import glob
from scipy.io import loadmat
import mysql.connector

# DB Config
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "326513",
    "database": "eeg_ml"
}

def check_db():
    print("Checking DB...")
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SHOW TABLES LIKE 'eeg_features'")
        result = cursor.fetchone()
        if result:
            print("Table 'eeg_features' exists.")
            cursor.execute("DESCRIBE eeg_features")
            cols = cursor.fetchall()
            print(f"Table has {len(cols)} columns.")
        else:
            print("Table 'eeg_features' DOES NOT exist.")
        conn.close()
    except Exception as e:
        print(f"DB Error: {e}")

def check_file():
    print("\nChecking File Processing...")
    files = glob.glob("*.mat")
    if not files:
        print("No .mat files found!")
        return
    
    f = files[0]
    print(f"Loading {f}...")
    try:
        mat = loadmat(f)
        data = mat["o"]["data"][0][0]
        print(f"Raw data shape: {data.shape}")
        
        EEG_COL_START = 3
        EEG_COL_END = 17
        eeg = data[:, EEG_COL_START:EEG_COL_END+1].T
        print(f"EEG shape: {eeg.shape}")
        
        FS = 128
        EPOCH_SEC = 5
        win = EPOCH_SEC * FS
        n_samples = eeg.shape[1]
        
        print(f"Window size: {win}")
        print(f"Samples: {n_samples}")
        
        loops = list(range(0, n_samples - win, win))
        print(f"Expected epochs in full file: {len(loops)}")
        
    except Exception as e:
        print(f"File Error: {e}")

if __name__ == "__main__":
    check_db()
    check_file()
