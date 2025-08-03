import pandas as pd

# ============
# Config
# ============
file_path = "UNSW_IoTSAT.csv"  # <=== The dataset file name

# ============
# Load dataset
# ============
df = pd.read_csv(file_path)
print("=== Dataset loaded ===")
print(f"Total records: {df.shape[0]}")
print()

# ============
# Completeness check
# ============
required_columns = ['Timestamp', 'RF_SNR_dB', 'RF_Throughput_bps', 'Attack_Flag', 'Attack_Type']

completeness = 100 * (df.dropna(subset=required_columns).shape[0] / df.shape[0])
print(f"Completeness (rows with required fields present): {completeness:.2f}%")

# ============
# Consistency check
# ============
try:
    df['RF_SNR_dB'] = pd.to_numeric(df['RF_SNR_dB'])
    df['RF_Throughput_bps'] = pd.to_numeric(df['RF_Throughput_bps'])
    consistency = 100.0
except:
    consistency = 0.0
print(f"Consistency (RF fields can be parsed as numeric): {consistency:.2f}%")

# ============
# Missing values
# ============
overall_missing = df.isnull().mean().mean() * 100
print(f"Overall missing values across all fields: {overall_missing:.2f}%")

# ============
# Duplicate records
# ============
duplicate_percentage = df.duplicated().mean() * 100
print(f"Duplicate records: {duplicate_percentage:.2f}%")

# ============
# Label correctness
# ============
bad_labels = df[((df['Attack_Flag']==0) & (df['Attack_Type'] != 'normal')) |
                ((df['Attack_Flag']==1) & (df['Attack_Type'] == 'normal'))]
label_correctness = 100 * (1 - bad_labels.shape[0] / df.shape[0])
print(f"Label correctness: {label_correctness:.2f}%")

# ============
# Summary
# ============
print("\n=== Summary ===")
print(f"Completeness: {completeness:.2f}%")
print(f"Consistency: {consistency:.2f}%")
print(f"Missing values: {overall_missing:.2f}%")
print(f"Duplicate records: {duplicate_percentage:.2f}%")
print(f"Label correctness: {label_correctness:.2f}%")
