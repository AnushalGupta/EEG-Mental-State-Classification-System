# Central Configuration for EEG Project

# 1. Feature Extraction Parameters (FROZEN)
FS = 128                  # Sampling Frequency (Hz)
EPOCH_SEC = 5             # Window length in seconds
EPOCH_SAMPLES = FS * EPOCH_SEC

# Frequency Bands (Hz)
BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30)
}

# 2. Label Semantics (Mapping)
# Standardized Labels across all datasets
# 0: Focused (High Attention)
# 1: Unfocused (Distracted)
# 2: Drowsy (Fatigued/Low Arousal)
LABEL_MAP = {
    0: "Focused",
    1: "Unfocused",
    2: "Drowsy"
}

# 3. Reproducibility
RANDOM_SEED = 42

# 4. Database
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "326513",
    "database": "eeg_ml"
}

# 5. Channel Configuration
# Enforce 14 channels for consistency
EXPECTED_CHANNELS = 14
