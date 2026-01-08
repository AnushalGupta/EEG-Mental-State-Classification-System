# 🧠 EEG Mental State Classification System

A research-grade machine learning pipeline to classify mental states (Focused, Unfocused, Drowsy/Fatigued) using EEG signals. This project integrates multiple heterogeneous datasets (**EMOTIV** and **DEAP**) into a unified SQL-backed architecture for robust and reproducible analysis.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Machine Learning](https://img.shields.io/badge/ML-RandomForest-green)
![Database](https://img.shields.io/badge/Database-MySQL-orange)

## 📌 Project Overview
Monitoring cognitive load and fatigue is critical for aviation safety, driving, and human-computer interaction. This system:
1.  **Extracts Features** from raw EEG signals (Band Power: Delta, Theta, Alpha, Beta).
2.  **Standardizes Feature** representations across datasets recorded with different hardware/contexts (EMOTIV Epoc vs. BioSemi) using a rigorous mapping methodology.
3.  **Stores Features** in a centralized MySQL database to enable offline querying and reproducible experiments.
4.  **Classifies States** using subject-wise cross-validation to ensure models generalize to new users.

## 🏗️ Architecture
The system follows a **"Store-Once, Train-Anytime"** pipeline:

```mermaid
graph LR
    A[Raw EEG Data\n(.mat / .dat)] --> B(ETL Pipelines)
    B --> C[(MySQL Database\nFeature Store)]
    C --> D(ML Training\nRandom Forest)
    D --> E[Evaluation Report]
```

## 📂 Repository Structure
```
eeg-mental-state-classification/
│
├── etl/                    # Data Extraction Scripts
│   ├── etl_emotiv.py       # Processes EMOTIV (Task-based) data
│   └── etl_deap.py         # Processes DEAP (Affective) data
│
├── training/               # Machine Learning
│   └── train_model.py      # Random Forest Trainer with GroupKFold
│
├── sql/                    # Database
│   └── schema.sql          # Table definitions
│
├── docs/                   # Research Documentation
│   ├── methodology.md      # Rules for Label Mapping & Feature Extraction
│   └── results.md          # Summary of Experiments
│
├── config.py               # Frozen Configuration (FS, Bands, Seeds)
├── db_utils.py             # Database Connectivity
└── requirements.txt        # Python Dependencies
```

## 🚀 Key Features
- **Multi-Dataset Integration**: Implements a documented proxy mapping from affective (arousal) labels to cognitive state categories.
- **Subject-Independent Validation**: Uses `GroupKFold` to prevent data leakage (Train on Subject A, Test on Subject B).
- **Frozen Configuration**: strict configuration management ensures all experiments are 100% reproducible.
- **Balanced Training**: Handles severe class imbalance (e.g., Drowsy >>> Unfocused) using calculated class weights.

## 📊 Results Summary
| Experiment | Dataset | Accuracy | Macro F1 | Insight |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | EMOTIV | **66.6%** | 0.49 | Strong detection of Drowsiness. |
| **Generalized** | Combined | **63.0%** | **0.53** | Improved macro-F1 indicates better handling of minority classes under class imbalance. |

*(Detailed confusion matrices available in `docs/results.md`)*

## 🛠️ Setup & Usage
1.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Configure Database**
    Update `config.py` with your MySQL credentials.
3.  **Run ETL (Data Ingestion)**
    Run scripts as modules from the root directory to ensure imports work correctly:
    ```bash
    python -m etl.etl_emotiv
    python -m etl.etl_deap
    ```
4.  **Train Model**
    ```bash
    python -m training.train_model
    ```

## 📜 Dataset Acknowledgements
- **EMOTIV**: Mental Attention State Detection (Kaggle).
- **DEAP**: Koelstra et al., 2012 (A Database for Emotion Analysis using Physiological Signals).
