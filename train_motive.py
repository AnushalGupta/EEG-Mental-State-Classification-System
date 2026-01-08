import os
import glob
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import welch, butter, filtfilt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, classification_report

FS = 128  # sampling frequency
EEG_COL_START = 3
EEG_COL_END = 17
EPOCH_SEC = 5

BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30)
}

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

def load_eeg(path):
    mat = loadmat(path)
    data = mat["o"]["data"][0][0]
    eeg = data[:, EEG_COL_START:EEG_COL_END+1]
    return eeg.T  # (channels, samples)

def slice_with_labels(eeg):
    samples = eeg.shape[1]

    # slice points
    s1 = 10 * 60 * FS
    s2 = 20 * 60 * FS

    # enforce bounds
    s1 = min(s1, samples)
    s2 = min(s2, samples)

    segments = [
        (eeg[:, :s1], 0),      # focused
        (eeg[:, s1:s2], 1),    # unfocused
        (eeg[:, s2:], 2)       # drowsy
    ]
    return segments

def extract_features(eeg):
    n_channels, n_samples = eeg.shape
    win = EPOCH_SEC * FS

    feats = []
    for start in range(0, n_samples - win, win):
        epoch = eeg[:, start:start+win]
        feat = {}

        for ch in range(n_channels):
            sig = bandpass(epoch[ch])
            for name, band in BANDS.items():
                feat[f"ch{ch+1}_{name}"] = band_power(sig, band)

        feats.append(feat)

    return feats

def process_folder(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.mat")))

    X_list = []
    y_list = []
    groups = []

    for f in files:
        fname = os.path.basename(f)
        eeg = load_eeg(f)

        segments = slice_with_labels(eeg)

        for seg, label in segments:
            features = extract_features(seg)
            for feat in features:
                X_list.append(feat)
                y_list.append(label)
                groups.append(fname)

        print(fname, "processed.")

    X = pd.DataFrame(X_list).fillna(0)
    y = np.array(y_list)
    groups = np.array(groups)

    print("Final dataset:", X.shape, "Labels:", np.unique(y, return_counts=True))
    return X, y, groups

def train(X, y, groups):
    gkf = GroupKFold(n_splits=5)
    fold = 1

    for tr, te in gkf.split(X, y, groups):
        clf = RandomForestClassifier(n_estimators=300)
        clf.fit(X.iloc[tr], y[tr])
        pred = clf.predict(X.iloc[te])
        print(f"\nFOLD {fold} Accuracy = {accuracy_score(y[te], pred):.4f}")
        print(classification_report(y[te], pred))
        fold += 1

if __name__ == "__main__":
    X, y, groups = process_folder(".")
    train(X, y, groups)
