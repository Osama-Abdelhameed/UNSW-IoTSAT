#!/usr/bin/env python3
"""
Comprehensive Machine Learning Analysis version with complete evaluation metrics and visualizations
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           confusion_matrix, classification_report, roc_curve, auc,
                           precision_recall_curve, average_precision_score)
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Keep system awake during long computations
import ctypes
import threading
import time
def keep_awake():
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    while True:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        time.sleep(30)
threading.Thread(target=keep_awake, daemon=True).start()

# Set publication-quality plot parameters
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.figsize'] = (8, 6)

print("="*80)
print(" SATELLITE ATTACK DETECTION: COMPREHENSIVE ML ANALYSIS")
print("="*80)

# ===============================================
# 1. DATA LOADING AND EXPLORATION
# ===============================================
print("\n PHASE 1: Data Loading and Exploration")
print("-"*50)

df = pd.read_csv("UNSW_IoTSAT_with_Feature_Engineering.csv")
print(f" Dataset loaded: {df.shape[0]:,} samples, {df.shape[1]} features")

# Define feature sets
safe_features = [
    # Engineered features
    'Speed_Discrepancy', 'Total_Speed_Discrepancy', 'Calculated_Horizontal_Speed', 
    'Distance_Moved', 'Acceleration_Magnitude', 
    'Power_W_deviation', 'Mag_X_uT_deviation', 'Mag_Y_uT_deviation',
    'Mag_Z_uT_deviation', 'Ambient_Light_Lux_deviation', 
    'Current_A_deviation', 'Shunt_Voltage_V_deviation',
    'Power_W_change_rate', 'Mag_X_uT_change_rate', 'Mag_Y_uT_change_rate', 
    'Mag_Z_uT_change_rate', 'Ambient_Light_Lux_change_rate', 
    'Current_A_change_rate', 'Shunt_Voltage_V_change_rate',
    'Power_W_rolling_mean', 'Power_W_rolling_std', 
    'Mag_X_uT_rolling_mean', 'Mag_X_uT_rolling_std',
    'Mag_Y_uT_rolling_mean', 'Mag_Y_uT_rolling_std', 
    'Mag_Z_uT_rolling_mean', 'Mag_Z_uT_rolling_std',
    'Ambient_Light_Lux_rolling_mean', 'Ambient_Light_Lux_rolling_std', 
    'Power_Light_Ratio', 'Magnetic_Speed_Ratio', 'Magnetic_Total', 
    'Velocity_North_ms_change', 'Velocity_East_ms_change', 'Velocity_Up_ms_change', 
    'Time_Delta', 'Composite_Anomaly_Score',
    # Original features
    'Shunt_Voltage_V', 'Current_A', 'Power_W', 
    'Mag_X_uT', 'Mag_Y_uT', 'Mag_Z_uT',
    'Proximity', 'Ambient_Light_Lux', 
    'Latitude', 'Longitude', 'Altitude_m',
    'GPS_Satellites', 'GPS_Fix_Quality', 
    'Velocity_North_ms', 'Velocity_East_ms', 'Velocity_Up_ms', 
    'Speed_ms', 'Magnetic_Magnitude_uT', 'Light_Proximity_Ratio',
    'RF_Frequency_Offset_Hz', 'RF_Doppler_Shift_Hz', 'RF_Constellation_Error'
]

# Extract features and target
X = df[[f for f in safe_features if f in df.columns]].copy()
y = df["Attack_Flag"]

print(f"\n Class Distribution:")
class_dist = y.value_counts()
print(f"  Normal: {class_dist[0]:,} ({class_dist[0]/len(y)*100:.1f}%)")
print(f"  Attack: {class_dist[1]:,} ({class_dist[1]/len(y)*100:.1f}%)")
print(f"  Imbalance Ratio: {class_dist[0]/class_dist[1]:.1f}:1")

# ===============================================
# 2. DATA PREPROCESSING
# ===============================================
print("\n  PHASE 2: Data Preprocessing")
print("-"*50)

# Handle non-numeric data
bad_cols = []
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = pd.to_numeric(X[col], errors='coerce')
    # Check for array/list entries
    if X[col].apply(lambda x: isinstance(x, (list, np.ndarray))).any():
        bad_cols.append(col)

if bad_cols:
    print(f"  Dropping {len(bad_cols)} columns with array entries")
    X = X.drop(columns=bad_cols)

# Handle missing values
missing_before = X.isnull().sum().sum()
X = X.fillna(X.median(numeric_only=True))
print(f" Filled {missing_before:,} missing values")

# Remove constant features
constant_features = [col for col in X.columns if X[col].nunique() <= 1]
if constant_features:
    print(f" Removed {len(constant_features)} constant features")
    X = X.drop(columns=constant_features)

print(f"\n Final feature matrix: {X.shape}")

# ===============================================
# 3. EXPERIMENTAL SETUP
# ===============================================
print("\n  PHASE 3: Experimental Setup")
print("-"*50)

# Stratified K-Fold Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print(f" Cross-validation: 5-fold stratified")

# Initialize results storage
results_summary = []
detailed_results = {}

# ===============================================
# 4. MODEL EVALUATION FUNCTION
# ===============================================
def evaluate_model_comprehensive(model, name, X_data, y_data, scale_data=False, 
                               plot_importance=False, handle_imbalance=False):
    """
    Comprehensive model evaluation with multiple metrics and visualizations
    """
    print(f"\n  Evaluating: {name}")
    print("-"*40)
    
    # Storage for metrics
    accs, precs, recs, f1s = [], [], [], []
    aucs, aps = [], []
    y_true_all, y_pred_all = [], []
    y_proba_all = []
    
    # Cross-validation
    for fold, (train_idx, test_idx) in enumerate(cv.split(X_data, y_data), 1):
        X_train, X_test = X_data.iloc[train_idx], X_data.iloc[test_idx]
        y_train, y_test = y_data.iloc[train_idx], y_data.iloc[test_idx]
        
        # Handle class imbalance with SMOTE
        if handle_imbalance and y_train.value_counts().min() > 5:
            smote = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            print(f"  Fold {fold}: Applied SMOTE - {Counter(y_train_balanced)}")
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
        
        # Scale data if needed
        if scale_data:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_balanced)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train_balanced
            X_test_scaled = X_test
        
        # Train and predict
        model_clone = model.__class__(**model.get_params())
        model_clone.fit(X_train_scaled, y_train_balanced)
        y_pred = model_clone.predict(X_test_scaled)
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        accs.append(acc)
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)
        
        # Store predictions
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
        
        # Get probabilities for ROC/PR curves
        if hasattr(model_clone, "predict_proba"):
            y_proba = model_clone.predict_proba(X_test_scaled)[:, 1]
            y_proba_all.extend(y_proba)
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            ap = average_precision_score(y_test, y_proba)
            aps.append(ap)
        
        print(f"  Fold {fold}: Acc={acc:.2%} | Prec={prec:.2%} | Rec={rec:.2%} | F1={f1:.2%}", end="")
        if aucs:
            print(f" | AUC={aucs[-1]:.3f}")
        else:
            print()
    
    # Calculate summary statistics
    results = {
        "Model": name,
        "Accuracy": np.mean(accs),
        "Acc_Std": np.std(accs),
        "Precision": np.mean(precs),
        "Prec_Std": np.std(precs),
        "Recall": np.mean(recs),
        "Rec_Std": np.std(recs),
        "F1-Score": np.mean(f1s),
        "F1_Std": np.std(f1s),
        "AUC": np.mean(aucs) if aucs else None,
        "AUC_Std": np.std(aucs) if aucs else None,
        "Avg_Precision": np.mean(aps) if aps else None
    }
    
    results_summary.append(results)
    detailed_results[name] = {
        'y_true': y_true_all,
        'y_pred': y_pred_all,
        'y_proba': y_proba_all if y_proba_all else None
    }
    
    print(f"\n   Average Performance:")
    print(f"     Accuracy: {results['Accuracy']:.2%} (±{results['Acc_Std']:.2%})")
    print(f"     F1-Score: {results['F1-Score']:.2%} (±{results['F1_Std']:.2%})")
    if results['AUC']:
        print(f"     AUC: {results['AUC']:.3f} (±{results['AUC_Std']:.3f})")
    
    # Generate visualizations
    generate_model_plots(model_clone, name, y_true_all, y_pred_all, 
                        y_proba_all if y_proba_all else None, 
                        X_train_scaled if scale_data else X_train, 
                        plot_importance)
    
    return results

def generate_model_plots(model, name, y_true, y_pred, y_proba, X_train, plot_importance):
    """Generate comprehensive plots for model evaluation"""
    
    # 1. Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'])
    plt.title(f'{name} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Add metrics to plot
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    plt.text(0.5, -0.15, f'Specificity: {specificity:.2%} | Sensitivity: {sensitivity:.2%}', 
             ha='center', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig(f"{name.replace(' ', '_')}_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Feature Importance (if applicable)
    if plot_importance and hasattr(model, "feature_importances_"):
        plt.figure(figsize=(10, 8))
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:20]
        
        plt.barh(range(20), importances[indices])
        plt.yticks(range(20), [X.columns[i] for i in indices])
        plt.xlabel('Importance Score')
        plt.title(f'{name} - Top 20 Feature Importances')
        plt.tight_layout()
        plt.savefig(f"{name.replace(' ', '_')}_feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. ROC Curve (if probabilities available)
    if y_proba is not None:
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{name} - ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{name.replace(' ', '_')}_roc_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Precision-Recall Curve
        plt.figure(figsize=(8, 6))
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        avg_precision = average_precision_score(y_true, y_proba)
        
        plt.plot(recall, precision, color='darkorange', lw=2,
                label=f'PR curve (AP = {avg_precision:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{name} - Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{name.replace(' ', '_')}_pr_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Probability Distribution
        y_proba = np.array(y_proba)
        plt.figure(figsize=(10, 6))
        plt.hist(y_proba[np.array(y_true) == 0], bins=50, alpha=0.6, 
                label='Normal', color='blue', density=True)
        plt.hist(y_proba[np.array(y_true) == 1], bins=50, alpha=0.6, 
                label='Attack', color='red', density=True)
        plt.xlabel('Predicted Probability')
        plt.ylabel('Density')
        plt.title(f'{name} - Probability Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{name.replace(' ', '_')}_probability_distribution.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()

# ===============================================
# 5. TRAIN AND EVALUATE ALL MODELS
# ===============================================
print("\n  PHASE 4: Model Training and Evaluation")
print("="*80)

# Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
evaluate_model_comprehensive(rf, "Random Forest", X, y, scale_data=False, 
                           plot_importance=True, handle_imbalance=False)

# XGBoost
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=class_dist[0]/class_dist[1],
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
evaluate_model_comprehensive(xgb_model, "XGBoost", X, y, scale_data=False, 
                           plot_importance=True, handle_imbalance=False)

# Gradient Boosting
gb = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42
)
evaluate_model_comprehensive(gb, "Gradient Boosting", X, y, scale_data=False, 
                           plot_importance=True, handle_imbalance=False)

# SVM (with proper scaling and parameters)
#svm = SVC(
 #  kernel='rbf',
  #  C=1.0,
   # gamma='scale',
    #class_weight='balanced',
    #probability=True,
    #random_state=42
#)
#evaluate_model_comprehensive(svm, "SVM (RBF)", X, y, scale_data=True, plot_importance=False, handle_imbalance=False)

svm_linear = LinearSVC(
    C=1.0,
    class_weight='balanced',
    max_iter=5000,
    random_state=42
)
evaluate_model_comprehensive(svm_linear, "Linear SVM", X, y, scale_data=True, 
                           plot_importance=False, handle_imbalance=False)

# Neural Network (with proper scaling)
mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42
)
evaluate_model_comprehensive(mlp, "Neural Network", X, y, scale_data=True, 
                           plot_importance=False, handle_imbalance=False)

# Isolation Forest (unsupervised)
print("\n Evaluating: Isolation Forest (Unsupervised)")
print("-"*40)
iso_accs = []
for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Train only on normal data
    X_train_normal = X_train[y_train == 0]
    contamination = max(0.05, min(0.5, y_train.sum()/len(y_train)))
    
    iso = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42
    )
    iso.fit(X_train_normal)
    
    y_pred = (iso.predict(X_test) == -1).astype(int)
    acc = accuracy_score(y_test, y_pred)
    iso_accs.append(acc)
    print(f"  Fold {fold}: Acc={acc:.2%}")

results_summary.append({
    "Model": "Isolation Forest",
    "Accuracy": np.mean(iso_accs),
    "Acc_Std": np.std(iso_accs),
    "Precision": None,
    "Prec_Std": None,
    "Recall": None,
    "Rec_Std": None,
    "F1-Score": None,
    "F1_Std": None,
    "AUC": None,
    "AUC_Std": None,
    "Avg_Precision": None
})

# ===============================================
# 6. GENERATE SUMMARY VISUALIZATIONS
# ===============================================
print("\n  PHASE 5: Generating Summary Visualizations")
print("-"*50)

# Create summary DataFrame
summary_df = pd.DataFrame(results_summary)
summary_df.to_csv("model_results.csv", index=False)
print(" Saved detailed results: model_results.csv")

# 1. Comprehensive Model Comparison
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Accuracy comparison
ax1 = axes[0, 0]
models = summary_df['Model']
x_pos = np.arange(len(models))
ax1.bar(x_pos, summary_df['Accuracy'], yerr=summary_df['Acc_Std'], 
        capsize=5, color='skyblue', edgecolor='navy')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(models, rotation=45, ha='right')
ax1.set_ylabel('Accuracy')
ax1.set_title('Model Accuracy Comparison')
ax1.set_ylim(0, 1)
ax1.grid(True, alpha=0.3)

# Add value labels
for i, v in enumerate(summary_df['Accuracy']):
    if not np.isnan(v):
        ax1.text(i, v + 0.01, f'{v:.2%}', ha='center', va='bottom')

# F1-Score comparison
ax2 = axes[0, 1]
f1_values = summary_df['F1-Score'].fillna(0)
ax2.bar(x_pos, f1_values, yerr=summary_df['F1_Std'].fillna(0), 
        capsize=5, color='lightgreen', edgecolor='darkgreen')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(models, rotation=45, ha='right')
ax2.set_ylabel('F1-Score')
ax2.set_title('Model F1-Score Comparison')
ax2.set_ylim(0, 1)
ax2.grid(True, alpha=0.3)

# Precision vs Recall
ax3 = axes[1, 0]
valid_models = summary_df[summary_df['Precision'].notna()]
ax3.scatter(valid_models['Recall'], valid_models['Precision'], s=200, alpha=0.6)
for idx, row in valid_models.iterrows():
    ax3.annotate(row['Model'], (row['Recall'], row['Precision']), 
                xytext=(5, 5), textcoords='offset points')
ax3.set_xlabel('Recall')
ax3.set_ylabel('Precision')
ax3.set_title('Precision vs Recall Trade-off')
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)

# AUC comparison (if available)
ax4 = axes[1, 1]
auc_models = summary_df[summary_df['AUC'].notna()]
if not auc_models.empty:
    ax4.bar(np.arange(len(auc_models)), auc_models['AUC'], 
            yerr=auc_models['AUC_Std'], capsize=5, 
            color='coral', edgecolor='darkred')
    ax4.set_xticks(np.arange(len(auc_models)))
    ax4.set_xticklabels(auc_models['Model'], rotation=45, ha='right')
    ax4.set_ylabel('AUC')
    ax4.set_title('Model AUC Comparison')
    ax4.set_ylim(0, 1)
    ax4.grid(True, alpha=0.3)
else:
    ax4.text(0.5, 0.5, 'AUC not available for all models', 
             ha='center', va='center', transform=ax4.transAxes)

plt.tight_layout()
plt.savefig("model_comparison_comprehensive.png", dpi=300, bbox_inches='tight')
plt.close()

# 2. Performance Summary Table
print("\n PERFORMANCE SUMMARY")
print("="*80)
print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'AUC':<12}")
print("-"*80)

for _, row in summary_df.iterrows():
    acc = f"{row['Accuracy']:.2%} ±{row['Acc_Std']:.2%}" if not np.isnan(row['Accuracy']) else "N/A"
    prec = f"{row['Precision']:.2%}" if row['Precision'] is not None and not np.isnan(row['Precision']) else "N/A"
    rec = f"{row['Recall']:.2%}" if row['Recall'] is not None and not np.isnan(row['Recall']) else "N/A"
    f1 = f"{row['F1-Score']:.2%}" if row['F1-Score'] is not None and not np.isnan(row['F1-Score']) else "N/A"
    auc_score = f"{row['AUC']:.3f}" if row['AUC'] is not None and not np.isnan(row['AUC']) else "N/A"
    
    print(f"{row['Model']:<20} {acc:<12} {prec:<12} {rec:<12} {f1:<12} {auc_score:<12}")

# 3. Generate LaTeX table for publication
latex_table = summary_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']].to_latex(
    index=False,
    float_format="%.3f",
    caption="Performance comparison of machine learning models for satellite attack detection",
    label="tab:model_comparison"
)

with open("results_table.tex", 'w') as f:
    f.write(latex_table)
print("\n Saved LaTeX table: results_table.tex")

# 4. Feature Analysis Summary
print("\n TOP FEATURES BY IMPORTANCE (Random Forest)")
print("-"*50)

# Get feature importances from Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X, y)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(15).to_string(index=False))

# Save feature importance
feature_importance.to_csv("feature_importance_analysis.csv", index=False)

print("\n" + "="*80)
print(" ANALYSIS COMPLETE!")
print("="*80)
print("\nGenerated files:")
print("   model_results_publication.csv - Detailed performance metrics")
print("   feature_importance_analysis.csv - Feature importance ranking")
print("   results_table.tex - LaTeX table for publication")
print("   model_comparison_comprehensive.png - Summary visualization")
print("   [Model]_*.png - Individual model visualizations")
print("\n All results saved!")
