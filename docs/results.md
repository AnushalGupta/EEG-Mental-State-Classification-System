# Research Report: EEG Mental State Classification

## 1. Executive Summary
We successfully implemented a **multi-dataset EEG classification pipeline** integrating both **EMOTIV** (Task-based) and **DEAP** (Affective-based) datasets. 
Using a **Random Forest Classifier** with **Frozen Feature Extraction** (14 Channels, 4 Bands), we achieved the following results:

| Experiment | Datasets | Accuracy | Macro F1 | Key Finding |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 3a** | EMOTIV Only | **66.6%** | 0.49 | Strong baseline; Unfocused class (1) is hardest to distinguish. |
| **Phase 3b** | DEAP Only | **55.3%** | 0.43 | Lower performance due to proxy label mapping complexity. |
| **Phase 3c** | Combined | **63.0%** | **0.53** | **Best Generalization**. Adding DEAP improved the overall class balance and F1 score. |

---

## 2. Methodology Check
- **Datasets**: 
    - EMOTIV (Primary): Labels derived from strict time segments.
    - DEAP (Proxy): Labels derived from Arousal mapping (>5 Focused, <3 Drowsy).
- **Features**: 14 Channels (AF3, F7... matched) x 4 Bands (Delta, Theta, Alpha, Beta) = 56 Features.
- **Validation**: Subject-wise `GroupKFold` (5 Splits).
- **Class Balancing**: `class_weight='balanced'` applied to handle the "Drowsy" dominance.

---

## 3. detailed Analysis

### Phase 3a: Baseline (EMOTIV)
- **Strengths**: High precision for **Drowsy** (Class 2) (~72%).
- **Weaknesses**: The **Unfocused** (Class 1) state is frequently misclassified as Drowsy. This suggests the transition from "distracted" to "tired" is subtle in this task.

### Phase 3b: Validation (DEAP)
- **Observations**: The model struggled more here. This is expected because:
    1.  We mapped *Emotion* (Arousal) to *Attention* (Focus), which is an imperfect proxy.
    2.  DEAP trials are shorter and have different induction methods (videos).
- **Value**: Despite lower accuracy, it proves the pipeline **mechanically supports heterogeneous data**.

### Phase 3c: Generalized Model (Combined)
- **Result**: The **Macro F1 score increased to 0.53**, surpassing the EMOTIV-only baseline (0.49).
- **Implication**: Providing the model with diverse data (Combined) helped it learn more robust features, especially for the minority classes.

---

## 4. Recommendations for Future Work
1.  **Refine DEAP Mapping**: The threshold of `Arousal > 5` for Focused might be too loose. Increasing it to `> 6` or combining with `Valence` might improve Phase 3b.
2.  **Feature selection**: We used all 56 features. Feature importance analysis (using Random Forest) could identify the top 10 markers for drowsiness.
3.  **Deep Learning**: With ~28k samples (Phase 3c), a 1D-CNN or LSTM might now outperform Random Forest.
