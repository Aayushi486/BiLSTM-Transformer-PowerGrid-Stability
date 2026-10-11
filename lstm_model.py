# ==============================
# 1. IMPORTS
# ==============================
import numpy as np
import pandas as pd
import glob
import os
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

# ==============================
# 2. SETTINGS
# ==============================
WINDOW_SIZE     = 5
STEP_SIZE       = 2
REDUNDANCY_THR  = 0.90   # drop group if spearman > this with higher IMD group

folder_path = r"C:\Users\Aayushi Jindal\Desktop\AJ codes\BTP\dataset_rms_final_real"
meta_path   = r"C:\Users\Aayushi Jindal\Desktop\AJ codes\BTP\dataset_rms_final_real\metadata_master_with_TSI_filtered.csv"

# ==============================
# 3. FEATURE GROUP DEFINITIONS
# Grouped by signal type
# exactly like the paper
# ==============================
FEATURE_GROUPS = {
    'F1_Bus_Voltage'  : 'Magnitude',
    'F2_Bus_Frequency': 'Frequency',
    'F3_Gen_Speed'    : 'Speed',
    'F4_Gen_Angle'    : 'firel',
    'F5_Load_ActiveP' : 'Active Power',
    'F6_Load_ReactiveQ': 'Reactive',
}

# ==============================
# 4. LOAD METADATA
# ==============================
meta_df = pd.read_csv(meta_path)
meta_df.columns = [c.strip() for c in meta_df.columns]
meta_dict = dict(zip(meta_df['sim_id'], meta_df['Stability_Label']))
print("Metadata Loaded:", meta_df.shape)

# ==============================
# 5. LOAD SIMULATION FILES
# ==============================
all_files = [
    f for f in glob.glob(os.path.join(folder_path, "*.csv"))
    if "metadata" not in os.path.basename(f).lower()
]
print(f"Files found: {len(all_files)}")

drop_keywords = ['Time', 'Results', 'Unnamed', 'sim_id',
                 'TSI', 'Stability', 'Instability', 'Fault']

# ==============================
# 6. STEP 1 — SCAN ALL FILES
# Build per-simulation summary
# for outlier detection AND
# group-level IMD analysis
# ==============================
print("\nStep 1: Scanning simulations...")

summary_list  = []
valid_files   = []
valid_labels  = []
col_names     = None

for i, f in enumerate(all_files):
    try:
        sim_id = os.path.basename(f).replace(".csv", "")
        if sim_id not in meta_dict:
            continue

        label = int(pd.to_numeric(meta_dict[sim_id], errors='coerce'))

        df = pd.read_csv(f, header=[0, 1])
        df.columns = [f"{str(c[0]).strip()}_{str(c[1]).strip()}"
                      for c in df.columns]

        if df.shape[1] <= 6:
            continue

        df = df.drop(columns=[c for c in df.columns
                               if any(x in c for x in drop_keywords)],
                     errors='ignore')
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna(axis=1, how='all').dropna(axis=0, how='all')

        if df.shape[1] == 0 or df.shape[0] < WINDOW_SIZE:
            continue

        df = df.fillna(df.mean())

        if col_names is None:
            col_names = df.columns.tolist()
            print(f"Raw columns per timestep: {len(col_names)}")

        summary = np.concatenate([df.mean().values, df.std().values])
        summary_list.append(summary)
        valid_files.append(f)
        valid_labels.append(label)

    except Exception:
        pass

    if i % 300 == 0:
        print(f"  Scanned {i}/{len(all_files)} files...")

print(f"Valid simulations: {len(valid_files)}")

# ==============================
# 7. STEP 2 — ISOLATION FOREST
# On summaries only — fast
# Removes outlier simulations
# BEFORE any windowing
# ==============================
print("\nStep 2: Isolation Forest outlier removal...")

summary_arr = np.nan_to_num(
    np.array(summary_list), nan=0.0, posinf=0.0, neginf=0.0
)

iso   = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
preds = iso.fit_predict(summary_arr)

clean_files   = [f for f, p in zip(valid_files,  preds) if p == 1]
clean_labels  = [l for l, p in zip(valid_labels, preds) if p == 1]
clean_summary = summary_arr[[i for i, p in enumerate(preds) if p == 1]]

print(f"Kept {len(clean_files)} simulations "
      f"(removed {len(valid_files)-len(clean_files)} outliers)")

# ==============================
# 8. STEP 3 — IMD + SPEARMAN
# Group-level feature selection
# exactly like Figure 7 in paper
# ==============================
print("\nStep 3: Group-level IMD + Spearman analysis...")

# Build summary dataframe of clean sims
summary_cols = (
    [f"mean_{c}" for c in col_names] +
    [f"std_{c}"  for c in col_names]
)
summary_df = pd.DataFrame(clean_summary, columns=summary_cols)
y_summary  = np.array(clean_labels)

# Map each group to its summary columns
group_col_map = {}
for group_name, keyword in FEATURE_GROUPS.items():
    matched = [c for c in summary_df.columns if keyword in c]
    group_col_map[group_name] = matched
    print(f"  {group_name}: {len(matched)} summary columns matched")

# ---- IMD Score per group ----
def compute_imd(X_group, y):
    X_stable   = X_group[y == 0]
    X_unstable = X_group[y == 1]
    mean_diff  = np.abs(X_stable.mean(axis=0) - X_unstable.mean(axis=0))
    std_all    = np.where(X_group.std(axis=0) < 1e-9,
                          1e-9, X_group.std(axis=0))
    return float(np.mean(mean_diff / std_all))

imd_scores = {}
for group_name, cols in group_col_map.items():
    if len(cols) == 0:
        print(f"  WARNING: {group_name} matched 0 columns!")
        imd_scores[group_name] = 0.0
        continue
    imd_scores[group_name] = compute_imd(
        summary_df[cols].values, y_summary
    )

imd_series = pd.Series(imd_scores).sort_values(ascending=False)
print("\nIMD scores per feature group (higher = more useful):")
for g, score in imd_series.items():
    print(f"  {g}: {score:.4f}")

# ---- Spearman between groups ----
# Represent each group by mean
# across all its columns
group_means = pd.DataFrame({
    name: summary_df[cols].mean(axis=1)
    for name, cols in group_col_map.items()
    if len(cols) > 0
})

spearman_matrix = group_means.corr(method='spearman').abs()

print("\nSpearman correlation between groups:")
print(spearman_matrix.round(2).to_string())

# ---- Drop redundant groups ----
# If two groups corr > 0.90,
# drop the one with lower IMD
dropped_groups = set()
checked        = set()

for g1 in spearman_matrix.columns:
    for g2 in spearman_matrix.index:
        if g1 == g2:
            continue
        if (g1, g2) in checked or (g2, g1) in checked:
            continue
        checked.add((g1, g2))

        if spearman_matrix.loc[g1, g2] > REDUNDANCY_THR:
            imd_g1 = imd_scores.get(g1, 0)
            imd_g2 = imd_scores.get(g2, 0)
            loser  = g2 if imd_g1 >= imd_g2 else g1
            dropped_groups.add(loser)
            print(f"  DROP {loser} — redundant (corr={spearman_matrix.loc[g1,g2]:.2f}), "
                  f"lower IMD than counterpart")

kept_groups = [g for g in imd_scores if g not in dropped_groups]
print(f"\nKept groups:    {kept_groups}")
print(f"Dropped groups: {list(dropped_groups)}")

# ---- Plot Figure 7 equivalent ----
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# (a) IMD bar chart
group_labels = [g.replace('_', ' ') for g in imd_series.index]
bar_colors   = ['#1D9E75' if g in kept_groups else '#B4B2A9'
                for g in imd_series.index]
short_labels = [f"F{i+1}" for i in range(len(imd_series))]

axes[0].bar(short_labels, imd_series.values,
            color=bar_colors, edgecolor='none')
axes[0].set_ylabel('IMD score', fontsize=11)
axes[0].set_xlabel('Feature Groups', fontsize=11)
axes[0].set_title('(a) IMD scores per feature group', fontsize=12)

# Legend for group names
legend_lines = [f"F{i+1} = {n.split('_',1)[1]}"
                for i, n in enumerate(imd_series.index)]
axes[0].text(0.98, 0.97, '\n'.join(legend_lines),
             transform=axes[0].transAxes,
             fontsize=8, va='top', ha='right',
             bbox=dict(boxstyle='round', facecolor='white',
                       alpha=0.8, edgecolor='lightgray'))

# (b) Spearman heatmap
ordered_names = list(imd_series.index)
spearman_ordered = spearman_matrix.loc[ordered_names, ordered_names]

sns.heatmap(
    spearman_ordered,
    ax=axes[1],
    cmap='Reds',
    vmin=0, vmax=1,
    annot=True, fmt='.2f',
    xticklabels=short_labels,
    yticklabels=short_labels,
    linewidths=0.5,
    annot_kws={'size': 9}
)
axes[1].set_title('(b) Spearman correlation between feature groups',
                  fontsize=12)

plt.suptitle('Discriminative ability and redundancy analysis '
             'of feature groups', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('feature_group_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print("Saved: feature_group_analysis.png")


# ==============================
# STEP 4 — FIXED WINDOWING
# Removed usecols trick —
# just read full file and filter
# ==============================
print("\nStep 4: Building 3D sequences...")

kept_keywords = [FEATURE_GROUPS[g] for g in kept_groups]
print(f"Keeping signal types: {kept_keywords}")
print(f"Processing {len(clean_files)} clean simulations...")

X_sequences = []
y_sequences = []
n_skipped   = 0
error_log   = []

for i, (f, label) in enumerate(zip(clean_files, clean_labels)):
    try:
        # Simple flat read — no usecols trick
        df = pd.read_csv(f, header=[0, 1])

        # Flatten column names
        df.columns = [
            f"{str(c[0]).strip()}_{str(c[1]).strip()}"
            for c in df.columns
        ]

        # Drop time and meta columns
        df = df.drop(
            columns=[c for c in df.columns
                     if any(x in c for x in drop_keywords)],
            errors='ignore'
        )

        # Keep only columns from kept groups
        feature_cols = [
            c for c in df.columns
            if any(kw in c for kw in kept_keywords)
        ]

        if len(feature_cols) == 0:
            n_skipped += 1
            error_log.append(f"No feature cols: {os.path.basename(f)}")
            continue

        df = df[feature_cols]
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna(axis=1, how='all')
        df = df.dropna(axis=0, how='all')

        if df.shape[0] < WINDOW_SIZE:
            n_skipped += 1
            error_log.append(f"Too few rows ({df.shape[0]}): {os.path.basename(f)}")
            continue

        df = df.fillna(df.mean())
        values = df.values.astype(np.float32)
        n_rows = values.shape[0]

        # Sliding window
        windows_this_file = 0
        for start in range(0, n_rows - WINDOW_SIZE + 1, STEP_SIZE):
            X_sequences.append(values[start : start + WINDOW_SIZE])
            y_sequences.append(label)
            windows_this_file += 1

        # First file success check
        if i == 0:
            print(f"\nFirst file check:")
            print(f"  File: {os.path.basename(f)}")
            print(f"  Shape after filtering: {df.shape}")
            print(f"  Feature cols sample: {feature_cols[:3]}")
            print(f"  Windows created: {windows_this_file}")
            print(f"  Label: {label}\n")

    except Exception as e:
        n_skipped += 1
        error_log.append(f"Exception {os.path.basename(f)}: {str(e)}")

    if i % 200 == 0:
        print(f"  Windowed {i}/{len(clean_files)}... "
              f"({len(X_sequences)} windows so far)")

print(f"\nDone!")
print(f"  Total windows: {len(X_sequences)}")
print(f"  Skipped files: {n_skipped}")

# Show first 5 errors to debug
if n_skipped > 0 and len(error_log) > 0:
    print(f"\nFirst 5 skip reasons:")
    for err in error_log[:5]:
        print(f"  {err}")

# Safety check before stacking
if len(X_sequences) == 0:
    raise ValueError(
        "Still 0 windows! Check error_log above. "
        "Run the single file debug below to investigate."
    )

# ==============================
# STEP 5 — STACK INTO 3D
# ==============================
print("\nStep 5: Stacking into 3D array...")

X_3d  = np.array(X_sequences, dtype=np.float32)
y_arr = np.array(y_sequences, dtype=np.int32)

del X_sequences, y_sequences

print(f"Final 3D shape: {X_3d.shape}")
print(f"  {X_3d.shape[0]} windows")
print(f"  {X_3d.shape[1]} timesteps (={WINDOW_SIZE})")
print(f"  {X_3d.shape[2]} features")
print(f"Label distribution: {dict(zip(*np.unique(y_arr, return_counts=True)))}")

# ==============================
# STEP 6 — SPLIT + NORMALIZE
# ==============================
print("\nStep 6: Splitting and normalizing...")

X_train, X_test, y_train, y_test = train_test_split(
    X_3d, y_arr,
    test_size=0.2,
    random_state=42,
    stratify=y_arr
)

del X_3d

n_train, T, F = X_train.shape
scaler = StandardScaler()
scaler.fit(X_train.reshape(-1, F))

X_train = scaler.transform(
    X_train.reshape(-1, F)).reshape(n_train, T, F)
X_test  = scaler.transform(
    X_test.reshape(-1, F)).reshape(X_test.shape[0], T, F)

print(f"X_train: {X_train.shape}")
print(f"X_test:  {X_test.shape}")

# ==============================
# STEP 7 — CLASS WEIGHTS
# ==============================
neg   = np.sum(y_train == 0)
pos   = np.sum(y_train == 1)
total = len(y_train)
class_weight = {0: total/(2*neg), 1: total/(2*pos)}
print(f"Class weights: {class_weight}")

# ==============================
# STEP 8 — LSTM MODEL
# ==============================
print("\nBuilding LSTM model...")

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(
        128, return_sequences=True,
        input_shape=(T, F)
    ),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.LSTM(64, return_sequences=False),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)
model.summary()

# ==============================
# STEP 9 — TRAIN
# ==============================
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=64,
    class_weight=class_weight,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=15,
            restore_best_weights=True,
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            patience=7,
            factor=0.5,
            min_lr=1e-6
        )
    ],
    verbose=1
)

# ==============================
# STEP 10 — EVALUATE
# ==============================
import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import roc_curve, auc

y_pred_prob = model.predict(X_test).ravel()
y_pred      = (y_pred_prob > 0.5).astype(int)

accuracy  = np.mean(y_pred == y_test)
precision = np.sum((y_pred==1)&(y_test==1)) / (np.sum(y_pred==1) + 1e-9)
recall    = np.sum((y_pred==1)&(y_test==1)) / (np.sum(y_test==1) + 1e-9)
f1        = 2*precision*recall / (precision + recall + 1e-9)
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
auc_score   = auc(fpr, tpr)

print("\n" + "="*50)
print("FINAL MODEL EVALUATION")
print("="*50)
print(f"  Accuracy  : {accuracy*100:.2f}%")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}  ← most important")
print(f"  F1 Score  : {f1:.4f}")
print(f"  AUC-ROC   : {auc_score:.4f}")
print("="*50)
print(classification_report(y_test, y_pred,
      target_names=['Stable (0)', 'Unstable (1)']))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

# Save plots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

epochs_ran = range(1, len(history.history['accuracy']) + 1)
axes[0,0].plot(epochs_ran, history.history['accuracy'],
               label='Train', color='#1D9E75')
axes[0,0].plot(epochs_ran, history.history['val_accuracy'],
               label='Val', color='#0F6E56', linestyle='--')
axes[0,0].set_title('(a) Accuracy'); axes[0,0].legend(); axes[0,0].grid(alpha=0.3)

axes[0,1].plot(epochs_ran, history.history['loss'],
               label='Train', color='#D85A30')
axes[0,1].plot(epochs_ran, history.history['val_loss'],
               label='Val', color='#993C1D', linestyle='--')
axes[0,1].set_title('(b) Loss'); axes[0,1].legend(); axes[0,1].grid(alpha=0.3)

im = axes[1,0].imshow(cm, cmap='Blues')
axes[1,0].set_title('(c) Confusion Matrix')
axes[1,0].set_xticks([0,1]); axes[1,0].set_yticks([0,1])
axes[1,0].set_xticklabels(['Stable','Unstable'])
axes[1,0].set_yticklabels(['Stable','Unstable'])
axes[1,0].set_xlabel('Predicted'); axes[1,0].set_ylabel('True')
for ii in range(2):
    for jj in range(2):
        axes[1,0].text(jj, ii, str(cm[ii,jj]), ha='center', va='center',
                       fontsize=14, fontweight='bold',
                       color='white' if cm[ii,jj] > cm.max()/2 else 'black')

axes[1,1].plot(fpr, tpr, color='#378ADD', lw=2,
               label=f'AUC = {auc_score:.4f}')
axes[1,1].plot([0,1],[0,1], 'gray', linestyle='--')
axes[1,1].set_title('(d) ROC Curve')
axes[1,1].set_xlabel('False Positive Rate')
axes[1,1].set_ylabel('True Positive Rate')
axes[1,1].legend(); axes[1,1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('lstm_evaluation.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: lstm_evaluation.png")
model.save("lstm_stability_model.h5")
print("Model saved:", os.path.abspath("lstm_stability_model.h5"))