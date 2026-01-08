# Research Methodology: EEG Mental State Classification

## 1. Dataset Strategy
To ensure robustness and generalizability, this study utilizes two distinct datasets:

### Primary Dataset: EMOTIV
- **Source**: Kaggle (Mental Attention State Detection)
- **Role**: Main training and validation source.
- **Nature**: Task-based recordings with explicit time-segmented labels.
- **Labels**:
    - **Focused**: First 10 minutes (Task engagement)
    - **Unfocused**: Next 10 minutes (Task distraction)
    - **Drowsy**: Remaining time (Fatigue onset)

### Proxy Validation Dataset: DEAP
- **Source**: Koelstra et al., 2012
- **Role**: Secondary validation to test generalization of "mental state" concepts.
- **Nature**: Emotion/Affective trials.
- **Mapping Strategy**:
    - High Arousal (> 5) + High Valence -> **Focused** (Approximation)
    - Low Arousal (< 3) -> **Drowsy/Passive** (Approximation)
    - *Note: Exact mapping thresholds to be tuned.*

## 2. Label Semantics (Frozen)
The system classifies into three discrete mental states. These definitions are frozen for all downstream tasks.

| Label ID | State | EMOTIV Definition | DEAP Proxy Definition |
| :--- | :--- | :--- | :--- |
| **0** | **Focused** | Active task engagement | High Arousal |
| **1** | **Unfocused** | Distracted / Mind-wandering | Mid Arousal / Irrelevant |
| **2** | **Drowsy** | Fatigued / Sleepy | Low Arousal |

## 3. Feature Extraction Protocol (Frozen)
To ensure comparability, all signals are processed through an identical pipeline:
1.  **Preprocessing**: No artifact removal (currently).
2.  **Filtering**: Bandpass filter (0.5 - 40 Hz).
3.  **Epoching**: Non-overlapping **5-second windows** (640 samples @ 128Hz).
4.  **Feature Computation**: Welch's PSD -> Band Power Integration.
    - **Delta**: 0.5 - 4 Hz
    - **Theta**: 4 - 8 Hz
    - **Alpha**: 8 - 13 Hz
    - **Beta**: 13 - 30 Hz
5.  **Feature Vector**: 14 Channels * 4 Bands = **56 Features**.

## 4. Evaluation Protocol
- **Subject-Wise Validation**: `GroupKFold` (5 Splits). strictly preventing data leakage between subjects.
- **Metrics**:
    - **Accuracy**: Overall correctness.
    - **Macro F1-Score**: To account for class imbalance.
    - **Confusion Matrix**: To visualize misclassifications (e.g., Drowsy vs Unfocused).
- **Class Handling**: `class_weight='balanced'` in Random Forest.
