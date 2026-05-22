# BiLSTM-Transformer-PowerGrid-Stability
# ⚡ Transient Stability Assessment Using Hybrid BiLSTM–Transformer Architecture

> A deep-learning framework for real-time power system transient stability classification using physics-informed post-fault windowing, parallel gated fusion, and focal loss.

---

## 📄 Paper

**"Transient Stability Assessment of Power Systems Using a Hybrid BiLSTM–Transformer Architecture with Focal Loss and Physics-Informed Post-Fault Windowing"**  
Aayushi Jindal, Shreya Jaiswal, Saurabh Verma, Bibhu Prasad Padhy  
Department of Electrical Engineering, IIT Ropar

---

## 🧠 Overview

This project addresses **real-time transient stability assessment (TSA)** in power systems — determining whether synchronous generators will maintain synchronism after a large disturbance (fault, line outage, etc.).

Traditional time-domain simulation is too slow for real-time use. This framework replaces it with a compact deep learning pipeline achieving:

| Metric | Value |
|---|---|
| Accuracy | **98.21%** |
| Recall (unstable detection) | **100%** |
| F1-Score | **97.83%** |
| ROC-AUC | **99.47%** |
| Inference Time | **5.89 ms/window** |
| False Alarm Rate | **2.99%** |

The model is **556× faster** than classical TSA methods while missing zero unstable events.

---

## 🏗️ Architecture

```
Input Window (T=50 timesteps × F features)
             │
      GaussianNoise(σ=0.05)
             │
    ┌────────┴────────┐
    │                 │
 BiLSTM           Transformer
 (64 units,       Encoder
  bidirectional)  (MHA h=2, dk=32
    │              + FFN + LayerNorm)
    └────────┬────────┘
             │
    Parallel Gated Fusion
    gt = σ(Wg[hᴮ; hᵀ])
    zt = gt⊙hᴮ + (1−gt)⊙hᵀ
             │
    Temporal Attention Pooling
    (learned per-timestep weights → context c ∈ ℝ⁶⁴)
             │
    Classification Head
    Dense(32) → BN → Dropout(0.4) → Sigmoid
             │
    Stable (0) / Unstable (1)
```

**Total trainable parameters: 84,897** — intentionally compact to avoid overfitting on medium-sized simulation datasets.

---

## 📁 Repository Structure

```
├── tsa_pipeline.py                      # Complete end-to-end pipeline (single script):
│                                        #   data loading → outlier removal → feature selection
│                                        #   → physics-informed windowing → normalisation
│                                        #   → model build → training → threshold sweep
│                                        #   → evaluation → all plots saved
│
├── results/
│   ├── architecture.png                 # End-to-end workflow diagram (Fig. 1 from paper)
│   ├── roc_curve.png                    # ROC curve on test set
│   ├── confusion_matrix.png             # Confusion matrix at chosen threshold
│   ├── preprocessing_analysis.png       # 4-panel: class distribution, splits, post-fault
│   │                                    #   sample histogram, window-level counts
│   ├── imd_feature_analysis.png         # IMD discriminability scores per feature group
│   ├── spearman_feature_analysis.png    # Absolute Spearman inter-group correlation heatmap
│   ├── threshold_metrics_analysis.png   # Val Accuracy / Precision / Recall / F1 vs threshold
│   ├── threshold_probability_analysis.png # Val predicted-probability distributions
│   │                                    #   (stable vs unstable) with chosen threshold line
│   ├── feature_group_accuracy_curve.png # Ablation: line plot of metrics across 4 feature combos
│   └── feature_group_accuracy_comparison.png # Ablation: bar chart of accuracy per combo
│
├── feature_group_ablation.csv           # Ablation results: accuracy, F1, FAR per feature combo
├── feature_group_selection_table.csv    # IMD + Spearman pruning decisions per group
├── unstable_detection_times.csv         # Per-simulation: first window where instability detected,
│                                        #   detection time in seconds (at 0.02s sample interval)
├── bilstm_transformer.keras             # Saved trained model (TensorFlow/Keras format)
│
├── requirements.txt                     # Python dependencies
└── README.md                            # This file
```

> **Note on dataset files:** The raw simulation CSVs and `metadata_master_with_TSI_filtered.csv` are not included in this repository due to size. Update the `folder_path` and `meta_path` variables at the top of `tsa_pipeline.py` to point to your local dataset directory before running.

---

## ⚙️ Pipeline Summary

The entire pipeline runs as a single script (`tsa_pipeline.py`) in 12 sequential steps:

**Step 1 — Load Metadata**  
Reads `metadata_master_with_TSI_filtered.csv` for per-simulation stability labels, `Post_Fault_Samples`, and `Instability_Time`. Computes the median post-fault length (P̃ = 200 rows) across all unstable simulations — used to trim stable simulations to a comparable length.

**Step 2 — Scan & Parse Simulation CSV Files**  
Loads each RMS simulation CSV (dual-header format), normalises column names, drops metadata/time columns, coerces to float32, removes all-NaN columns/rows, imputes remaining NaNs with column means. Builds a per-simulation summary vector `[μ; σ]`.

**Step 3 — Isolation Forest Outlier Removal**  
Fits Isolation Forest (1% contamination) on the summary vectors. Removes anomalous simulations before any splitting or training.

**Step 4 — IMD + Spearman Feature Group Selection**  
Evaluates six physical feature groups (Bus Voltage, Bus Frequency, Generator Speed, Generator Rotor Angle, Load Active Power, Load Reactive Power) using Inter Mean Distance (IMD) for discriminability and Spearman rank correlation (ρₛ > 0.90) for redundancy pruning. **Bus Frequency is pruned** (ρₛ = 0.93 with Generator Speed); the remaining 5 groups are retained.

**Step 5 — File-Level Stratified Split**  
Splits simulation files 70/30 (train+val / test), then 75/25 within train+val → **52.5% train / 17.5% val / 30% test**, preserving the natural stable:unstable ratio (~2.9:1) across all splits.

**Step 6 — File-Level Balancing** *(train only, optional)*  
The script supports optional 4:1 stable:unstable undersampling of training files. The final run uses the natural distribution for all splits.

**Step 7 — Physics-Informed Windowing**  
Extracts sliding windows (T=50, stride=25) from the correct region of each simulation:
- **Unstable sims** → window only the post-fault block (last `Post_Fault_Samples` rows, per-simulation from metadata)
- **Stable sims** → window only the last P̃ = 200 rows (comparable length, avoids meaningless pre-fault data)

This ensures every label-1 training window contains genuine diverging/oscillating rotor dynamics.

**Step 8 — StandardScaler Normalisation**  
Fitted on training windows only; applied to val and test.

**Step 9 — Build Model**  
Constructs the BiLSTM–Transformer with Parallel Gated Fusion (see architecture above).

**Step 10 — Train**  
- Focal Loss (γ=2.0, α=0.75)  
- Adam optimiser (lr=2.5×10⁻³, gradient clipping 1.0)  
- Linear LR warmup over first 5 epochs  
- `ReduceLROnPlateau` on `val_pr_auc` (patience=8, factor=0.5)  
- `EarlyStopping` on `val_pr_auc` (patience=20, restores best weights)  
- Up to 150 epochs, batch size 128

**Step 11 — Threshold Sweep on Validation Set**  
Sweeps τ ∈ [0.01, 0.99] and identifies four operating points:

| Strategy | Rule |
|---|---|
| A | argmax F1(τ) |
| B | argmax Recall(τ) subject to Precision ≥ 70% |
| C | argmax Precision(τ) subject to Recall ≥ 70% |
| D | min FAR subject to Recall ≥ 95% |
| E *(default)* | FAR nearest 0.5% subject to Recall ≥ 95% |

Strategy E is chosen by default (power systems penalise missed instabilities more than false alarms). Chosen threshold: **τ = 0.33**.

**Step 12 — Evaluation + Plots**  
Reports Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Balanced Accuracy, MCC, FAR, and inference time. Saves all result plots and CSVs. Also runs a **feature-group ablation** across 4 combinations of 5 groups.

---

## 📊 Results

### Test Set Performance (τ = 0.33, Strategy E)

| Acc | Prec | Recall | F1 | ROC-AUC | PR-AUC | Bal. Acc | MCC | FAR | Time |
|---|---|---|---|---|---|---|---|---|---|
| 98.21% | 95.74% | **100%** | 97.83% | 99.47% | 99.18% | 98.51% | 0.9638 | 2.99% | 5.89 ms |

### Comparison With Prior Work

| Method | Acc (%) | FAR (%) | Time (s) |
|---|---|---|---|
| RF/DT | 93.25 | 8.39 | 0.014 |
| SVM | 90.50 | 3.07 | 0.193 |
| 2D CNN | 95.35 | 4.11 | 0.190 |
| DBN | 93.25 | 1.99 | 0.291 |
| CNN–LSTM | 96.50 | 1.99 | 0.095 |
| **Proposed** | **98.21** | 2.99 | **0.006** |

---

## 🔬 Experimental Setup

- **Test system**: IEEE 39-bus New England power system
- **Simulator**: DIgSILENT PowerFactory 2026
- **Disturbances**: Three-phase short-circuit faults, transmission-line outages, contingency events
- **Dataset**: 2046 simulations (1528 stable / 518 unstable) after Isolation Forest filtering
- **Split**: 52.5% train / 17.5% val / 30.0% test (stratified, file-level)
- **Sample interval**: 0.02 s

---

## 🚀 Getting Started

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure paths
Open `tsa_pipeline.py` and set the two path variables near the top:
```python
folder_path = r"/path/to/your/rms_simulation_csvs"
meta_path   = r"/path/to/your/metadata_master_with_TSI_filtered.csv"
```

### 3. Run the full pipeline
```bash
python tsa_pipeline.py
```

All output plots, CSVs, and the saved model (`bilstm_transformer.keras`) will be written to the working directory.

---

## 📦 Requirements

```
tensorflow>=2.9.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
```

---

## 🔭 Future Work

- Physics-informed loss incorporating swing equation residual (Mδ̈ + Dδ̇ = Pm − Pe)
- Continual online learning to adapt as new fault events are recorded
- Multi-label output: fault type + stability margin + binary label
- Model compression via knowledge distillation for substation-level embedded hardware

---

## 📚 Citation

If you use this work, please cite:

```bibtex
@article{jindal2024tsa,
  title={Transient Stability Assessment of Power Systems Using a Hybrid BiLSTM–Transformer Architecture
         with Focal Loss and Physics-Informed Post-Fault Windowing},
  author={Jindal, Aayushi and Jaiswal, Shreya and Verma, Saurabh and Padhy, Bibhu Prasad},
  institution={Department of Electrical Engineering, IIT Ropar},
  year={2024}
}
```

---

## 📜 License

This project is released for academic and research purposes.
