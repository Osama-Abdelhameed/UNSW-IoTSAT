import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# === CONFIG ===
sns.set(style="whitegrid")
plt.rcParams.update({'figure.max_open_warning': 0})  # suppress matplotlib warnings

# === LOAD DATA ===
# Adjust to your filename
df = pd.read_csv("UNSW_IoTSAT_With_Feature_Engineering.csv", low_memory=False)

# === HISTOGRAM OF COMPOSITE ANOMALY SCORE ===
plt.figure(figsize=(8,5))
sns.histplot(df['Composite_Anomaly_Score'].dropna(), bins=50, kde=True, color='royalblue')
plt.title("Distribution of Composite Anomaly Scores")
plt.xlabel("Composite Anomaly Score")
plt.ylabel("Frequency")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("Composite_Anomaly_Score_Histogram.png", dpi=300)
plt.close()

# === BOXPLOT OF COMPOSITE SCORE BY ATTACK ===
plt.figure(figsize=(8,5))
sns.boxplot(x='Attack_Flag', y='Composite_Anomaly_Score', data=df)
plt.title("Composite Anomaly Score by Attack Flag")
plt.xlabel("Attack Flag (0=Normal, 1=Attack)")
plt.ylabel("Composite Anomaly Score")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("Composite_Anomaly_Score_Boxplot.png", dpi=300)
plt.close()

# === CORRELATION HEATMAP OF ENGINEERED FEATURES ===
engineered_cols = [
    'Speed_Discrepancy', 'Distance_Moved', 'Acceleration_Magnitude',
    'RF_Signal_Stability', 'RF_SNR_Stability', 
    'Power_Light_Ratio', 'Magnetic_Speed_Ratio', 
    'Composite_Anomaly_Score'
]

# Only use columns that exist
existing_cols = [col for col in engineered_cols if col in df.columns]

plt.figure(figsize=(10,8))
corr = df[existing_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, linewidths=0.5)
plt.title("Correlation Matrix of Engineered Features")
plt.tight_layout()
plt.savefig("Engineered_Features_Correlation_Heatmap.png", dpi=300)
plt.close()

# === PAIRPLOT TO COMPARE NORMAL VS ATTACK ===
if 'Attack_Flag' in df.columns:
    subset_cols = existing_cols + ['Attack_Flag']
    sns.pairplot(df[subset_cols], hue='Attack_Flag', corner=True, 
                 plot_kws={'alpha':0.5, 's':15})
    plt.suptitle("Pairwise Relationships of Engineered Features by Attack Flag", y=1.02)
    plt.savefig("Engineered_Features_Pairplot.png", dpi=300)
    plt.close()

print("All plots generated and saved as PNG files.")
