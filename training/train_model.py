import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from db_utils import get_db_connection
import config

def load_data_from_db(datasets=["EMOTIV"]):
    print(f"Loading data for {datasets} from Database...")
    conn = get_db_connection()
    
    # Construct comma-separated string for SQL IN clause
    datasets_str = "', '".join(datasets)
    query = f"SELECT * FROM eeg_features WHERE dataset_name IN ('{datasets_str}')"
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    if df.empty:
        print(f"Warning: No data found for {datasets}")
        return df

    print(f"Loaded {len(df)} samples.")
    print(f"Class distribution: {df['label'].value_counts().to_dict()}")
    return df

def prepare_data(df):
    # Identify feature columns (starting with 'ch')
    feature_cols = [c for c in df.columns if c.startswith("ch")]
    
    X = df[feature_cols]
    y = df["label"].astype(int)
    groups = df["subject_id"]
    
    return X, y, groups

def train_model(X, y, groups):
    gkf = GroupKFold(n_splits=5)
    
    print(f"\n--- Starting Training (GroupKFold) ---")
    print(f"Random Seed: {config.RANDOM_SEED}")
    print(f"Class Weights: Balanced")
    
    fold = 1
    accuracies = []
    f1_scores = []
    
    # Check if we have enough groups
    n_groups = groups.nunique()
    if n_groups < 5:
        print(f"Warning: Only {n_groups} subjects found. Reducing n_splits.")
        gkf = GroupKFold(n_splits=n_groups)

    for tr, te in gkf.split(X, y, groups):
        X_tr, y_tr = X.iloc[tr], y.iloc[tr]
        X_te, y_te = X.iloc[te], y.iloc[te]
        
        # Fixed Random Seed & Class Balancing
        clf = RandomForestClassifier(
            n_estimators=100, 
            n_jobs=-1, 
            random_state=config.RANDOM_SEED,
            class_weight="balanced"
        )
        clf.fit(X_tr, y_tr)
        
        pred = clf.predict(X_te)
        
        # Metrics
        acc = accuracy_score(y_te, pred)
        f1 = f1_score(y_te, pred, average="macro")
        cm = confusion_matrix(y_te, pred)
        
        accuracies.append(acc)
        f1_scores.append(f1)
        
        print(f"\nFOLD {fold}: Accuracy = {acc:.4f} | Macro F1 = {f1:.4f}")
        print("Confusion Matrix:\n", cm)
        print(classification_report(y_te, pred))
        fold += 1
        
    print(f"\nResults Summary:")
    print(f"Mean Accuracy: {np.mean(accuracies):.4f}")
    print(f"Mean Macro F1: {np.mean(f1_scores):.4f}")

def run_experiment(datasets, exp_name):
    print(f"\n==========================================")
    print(f"EXPERIMENT: {exp_name}")
    print(f"Datasets: {datasets}")
    print(f"==========================================")
    
    df = load_data_from_db(datasets)
    if df.empty:
        print("Skipping experiment (No Data).")
        return

    X, y, groups = prepare_data(df)
    train_model(X, y, groups)

if __name__ == "__main__":
    try:
        # 1. EMOTIV Only
        run_experiment(["EMOTIV"], "Phase 3a: Baseline (EMOTIV)")
        
        # 2. DEAP Only (Phase 2 Verification)
        run_experiment(["DEAP"], "Phase 3b: Validation (DEAP)")
        
        # 3. Combined
        run_experiment(["EMOTIV", "DEAP"], "Phase 3c: Generalized Model (Combined)")
        
    except Exception as e:
        print(f"Error: {e}")
