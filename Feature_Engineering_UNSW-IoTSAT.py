#!/usr/bin/env python3
"""
The newly added Engineering Feature for UNSW-IoTSAT dataset
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print(" Robust Feature Engineering for Attack Detection")
print("="*60)

# Load dataset
dataset_file = 'UNSW_IoTSAT.csv'
print(f"\n Loading: {dataset_file}")

try:
    data = pd.read_csv(dataset_file)
    print(f" Loaded {len(data):,} records")
except Exception as e:
    print(f" Error: {e}")
    exit(1)

# Create a copy for feature engineering
df = data.copy()

# IMPORTANT: Add original index to preserve order
df['_original_index'] = df.index

print("\n Data Type Cleaning...")
# Convert all numeric columns to proper types
numeric_columns = [
    'Shunt_Voltage_V', 'Current_A', 'Power_W',
    'Mag_X_uT', 'Mag_Y_uT', 'Mag_Z_uT',
    'Proximity', 'Ambient_Light_Lux',
    'Latitude', 'Longitude', 'Altitude_m',
    'GPS_Satellites', 'GPS_Fix_Quality',
    'Velocity_North_ms', 'Velocity_East_ms', 'Velocity_Up_ms',
    'Speed_ms', 'RF_Signal_Strength_dBm', 'RF_SNR_dB',
    'RF_Bit_Error_Rate', 'RF_Packet_Error_Rate', 'RF_Throughput_bps',
    'RF_Frequency_Offset_Hz', 'RF_Doppler_Shift_Hz',
    'RF_CRC_Errors', 'RF_Constellation_Error'
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        null_count = df[col].isnull().sum()
        if null_count > 0:
            print(f"   {col}: {null_count} non-numeric values converted to NaN")

print("\n🔧 Creating Advanced Features...")

# 1. TIMESTAMP FEATURES
print("\n Timestamp Features:")
try:
    # Convert to datetime and then to Unix timestamp
    df['Timestamp_dt'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df['Unix_Timestamp'] = df['Timestamp_dt'].astype(np.int64) // 10**9
    
    # Sort by satellite and time for temporal features (but keep original index)
    df_sorted = df.sort_values(['Satellite_ID', 'Unix_Timestamp']).copy()
    
    # Time between consecutive readings per satellite
    df_sorted['Time_Delta'] = df_sorted.groupby('Satellite_ID')['Unix_Timestamp'].diff()
    
    # Merge back to original order
    df['Time_Delta'] = df_sorted.set_index('_original_index')['Time_Delta']
    
    print("    Unix timestamp created")
    print("    Time delta between readings calculated")
    print("    Original row order preserved")
except Exception as e:
    print(f"    Error with timestamp: {e}")

# 2. SPEED DISCREPANCY FEATURES
print("\n Speed Discrepancy Features:")
try:
    # Ensure numeric types
    vel_cols = ['Velocity_North_ms', 'Velocity_East_ms', 'Velocity_Up_ms']
    for col in vel_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Calculate horizontal speed from velocity components
    df['Calculated_Horizontal_Speed'] = np.sqrt(
        df['Velocity_North_ms']**2 + df['Velocity_East_ms']**2
    )
    
    # Check if Horizontal_Speed_ms is numeric
    df['Horizontal_Speed_ms_numeric'] = pd.to_numeric(df['Horizontal_Speed_ms'], errors='coerce')
    
    # Speed discrepancy
    df['Speed_Discrepancy'] = abs(
        df['Calculated_Horizontal_Speed'] - df['Horizontal_Speed_ms_numeric']
    )
    
    # Also check against reported Speed_ms
    df['Speed_ms'] = pd.to_numeric(df['Speed_ms'], errors='coerce').fillna(0)
    df['Total_Speed_Discrepancy'] = abs(
        df['Speed_ms'] - np.sqrt(
            df['Velocity_North_ms']**2 + 
            df['Velocity_East_ms']**2 + 
            df['Velocity_Up_ms']**2
        )
    )
    
    print("    Calculated vs reported speed discrepancy")
    valid_discrepancy = df['Speed_Discrepancy'].notna().sum()
    print(f"   Valid discrepancy values: {valid_discrepancy:,}")
except Exception as e:
    print(f"  Error with speed calculations: {e}")

# 3. MOVEMENT AND ACCELERATION FEATURES
print("\n Movement and Acceleration Features:")
try:
    # Work on sorted data for temporal features
    df_sorted = df.sort_values(['Satellite_ID', 'Unix_Timestamp']).copy()
    
    # Ensure numeric types for position columns
    pos_cols = ['Latitude', 'Longitude', 'Altitude_m']
    for col in pos_cols:
        df_sorted[col] = pd.to_numeric(df_sorted[col], errors='coerce')
    
    # Calculate distance moved between consecutive readings
    for col in pos_cols:
        df_sorted[f'{col}_prev'] = df_sorted.groupby('Satellite_ID')[col].shift(1)
    
    # Simple Euclidean distance (approximation)
    df_sorted['Distance_Moved'] = np.sqrt(
        (df_sorted['Latitude'] - df_sorted['Latitude_prev'])**2 +
        (df_sorted['Longitude'] - df_sorted['Longitude_prev'])**2 +
        ((df_sorted['Altitude_m'] - df_sorted['Altitude_m_prev'])/1000)**2  # Convert to km
    ) * 111.32  # Approximate km per degree at equator
    
    # Velocity changes (acceleration proxies)
    for vel in ['Velocity_North_ms', 'Velocity_East_ms', 'Velocity_Up_ms']:
        df_sorted[f'{vel}_change'] = df_sorted.groupby('Satellite_ID')[vel].diff()
    
    # Safe division for acceleration
    time_delta_safe = df_sorted['Time_Delta'].fillna(1).replace(0, 1)
    df_sorted['Acceleration_Magnitude'] = np.sqrt(
        df_sorted['Velocity_North_ms_change']**2 +
        df_sorted['Velocity_East_ms_change']**2 +
        df_sorted['Velocity_Up_ms_change']**2
    ) / time_delta_safe
    
    # Merge back to original order
    temp_cols = ['Latitude_prev', 'Longitude_prev', 'Altitude_m_prev', 'Distance_Moved',
                 'Velocity_North_ms_change', 'Velocity_East_ms_change', 'Velocity_Up_ms_change',
                 'Acceleration_Magnitude']
    
    for col in temp_cols:
        if col in df_sorted.columns:
            df[col] = df_sorted.set_index('_original_index')[col]
    
    print("    Distance moved between readings calculated")
    print("    Acceleration magnitude calculated")
    valid_accel = df['Acceleration_Magnitude'].notna().sum()
    print(f"    Valid acceleration values: {valid_accel:,}")
except Exception as e:
    print(f"    Error with movement calculations: {e}")
    import traceback
    traceback.print_exc()

# 4. SENSOR ANOMALY DETECTION FEATURES
print("\n  Sensor Anomaly Detection Features:")
try:
    # Work on sorted data for rolling windows
    df_sorted = df.sort_values(['Satellite_ID', 'Unix_Timestamp']).copy()
    
    # Changes in sensor readings
    sensor_cols = ['Power_W', 'Mag_X_uT', 'Mag_Y_uT', 'Mag_Z_uT', 
                   'Ambient_Light_Lux', 'Current_A', 'Shunt_Voltage_V']
    
    feature_count = 0
    for sensor in sensor_cols:
        if sensor in df_sorted.columns:
            # Ensure numeric
            df_sorted[sensor] = pd.to_numeric(df_sorted[sensor], errors='coerce')
            
            # Rate of change
            sensor_diff = df_sorted.groupby('Satellite_ID')[sensor].diff()
            time_delta_safe = df_sorted['Time_Delta'].fillna(1).replace(0, 1)
            df_sorted[f'{sensor}_change_rate'] = sensor_diff / time_delta_safe
            
            # Rolling statistics (5-sample window)
            df_sorted[f'{sensor}_rolling_mean'] = df_sorted.groupby('Satellite_ID')[sensor].rolling(
                window=5, min_periods=1
            ).mean().reset_index(0, drop=True)
            
            df_sorted[f'{sensor}_rolling_std'] = df_sorted.groupby('Satellite_ID')[sensor].rolling(
                window=5, min_periods=1
            ).std().reset_index(0, drop=True)
            
            # Deviation from rolling mean
            df_sorted[f'{sensor}_deviation'] = abs(df_sorted[sensor] - df_sorted[f'{sensor}_rolling_mean'])
            
            # Merge back to original order
            for suffix in ['_change_rate', '_rolling_mean', '_rolling_std', '_deviation']:
                col_name = f'{sensor}{suffix}'
                if col_name in df_sorted.columns:
                    df[col_name] = df_sorted.set_index('_original_index')[col_name]
                    feature_count += 1
    
    print("    Sensor change rates calculated")
    print("    Rolling statistics computed")
    print(f"   Created {feature_count} sensor features")
except Exception as e:
    print(f"   Error with sensor features: {e}")

# 5. RF COMMUNICATION FEATURES
print("\n  RF Communication Features:")
try:
    # Work on sorted data
    df_sorted = df.sort_values(['Satellite_ID', 'Unix_Timestamp']).copy()
    
    # Ensure numeric
    df_sorted['RF_Signal_Strength_dBm'] = pd.to_numeric(df_sorted['RF_Signal_Strength_dBm'], errors='coerce')
    df_sorted['RF_SNR_dB'] = pd.to_numeric(df_sorted['RF_SNR_dB'], errors='coerce')
    df_sorted['RF_Throughput_bps'] = pd.to_numeric(df_sorted['RF_Throughput_bps'], errors='coerce')
    df_sorted['RF_Packet_Error_Rate'] = pd.to_numeric(df_sorted['RF_Packet_Error_Rate'], errors='coerce')
    
    # Signal stability (standard deviation over window)
    df_sorted['RF_Signal_Stability'] = df_sorted.groupby('Satellite_ID')['RF_Signal_Strength_dBm'].rolling(
        window=5, min_periods=1
    ).std().reset_index(0, drop=True)
    
    # SNR stability
    df_sorted['RF_SNR_Stability'] = df_sorted.groupby('Satellite_ID')['RF_SNR_dB'].rolling(
        window=5, min_periods=1
    ).std().reset_index(0, drop=True)
    
    # Merge back
    df['RF_Signal_Stability'] = df_sorted.set_index('_original_index')['RF_Signal_Stability']
    df['RF_SNR_Stability'] = df_sorted.set_index('_original_index')['RF_SNR_Stability']
    
    # Packet-related features (can be done on unsorted data)
    df['Effective_Throughput'] = df['RF_Throughput_bps'] * (1 - df['RF_Packet_Error_Rate'])
    
    # Safe division for efficiency
    signal_adjusted = df['RF_Signal_Strength_dBm'] + 100
    signal_adjusted = signal_adjusted.replace(0, 1)  # Avoid division by zero
    df['Communication_Efficiency'] = df['Effective_Throughput'] / signal_adjusted
    
    print("    RF signal stability calculated")
    print("   Communication efficiency metrics created")
except Exception as e:
    print(f"    Error with RF features: {e}")

# 6. CROSS-DOMAIN FEATURES
print("\n  Cross-Domain Features:")
try:
    # Ensure numeric types
    df['Power_W'] = pd.to_numeric(df['Power_W'], errors='coerce').fillna(0)
    df['Ambient_Light_Lux'] = pd.to_numeric(df['Ambient_Light_Lux'], errors='coerce').fillna(0)
    
    # Power efficiency features (safe division)
    df['Power_Light_Ratio'] = df['Power_W'] / (df['Ambient_Light_Lux'] + 1)
    
    signal_adjusted = df['RF_Signal_Strength_dBm'] + 100
    signal_adjusted = signal_adjusted.replace(0, 1)
    df['Power_Per_Signal'] = df['Power_W'] / signal_adjusted
    
    # Magnetic field vs movement
    mag_cols = ['Mag_X_uT', 'Mag_Y_uT', 'Mag_Z_uT']
    for col in mag_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    df['Magnetic_Total'] = np.sqrt(df['Mag_X_uT']**2 + df['Mag_Y_uT']**2 + df['Mag_Z_uT']**2)
    
    speed_safe = df['Speed_ms'] + 1
    df['Magnetic_Speed_Ratio'] = df['Magnetic_Total'] / speed_safe
    
    # Position vs signal strength (distance effects)
    df['Altitude_m'] = pd.to_numeric(df['Altitude_m'], errors='coerce').fillna(0)
    df['Altitude_Signal_Ratio'] = df['Altitude_m'] / signal_adjusted
    
    print("   Cross-domain ratios calculated")
except Exception as e:
    print(f"  Error with cross-domain features: {e}")

# 7. COMPOSITE ANOMALY SCORE
print("\n  Composite Anomaly Scores:")
try:
    # Create composite anomaly scores
    anomaly_features = []
    
    # Collect all anomaly indicators
    if 'Speed_Discrepancy' in df.columns:
        anomaly_features.append('Speed_Discrepancy')
    
    # Add sensor deviations
    deviation_cols = [col for col in df.columns if col.endswith('_deviation')]
    anomaly_features.extend(deviation_cols)
    
    # Normalize and combine
    if anomaly_features:
        normalized_features = []
        for feat in anomaly_features:
            if feat in df.columns:
                # Remove NaN and infinite values
                feat_clean = df[feat].replace([np.inf, -np.inf], np.nan)
                
                # Normalize using robust scaling (median and MAD)
                median = feat_clean.median()
                mad = (feat_clean - median).abs().median()
                if mad > 0:
                    df[f'{feat}_normalized'] = (feat_clean - median) / mad
                    normalized_features.append(f'{feat}_normalized')
        
        # Composite anomaly score
        if normalized_features:
            df['Composite_Anomaly_Score'] = df[normalized_features].abs().mean(axis=1)
            print(f"   ✓ Composite anomaly score from {len(normalized_features)} features")
    
except Exception as e:
    print(f"   Error with anomaly scores: {e}")

# Remove the temporary index column before saving
df = df.drop('_original_index', axis=1)

# VERIFICATION - Check order preservation
print("\n  Verifying Order Preservation:")
order_preserved = True
for i in range(min(5, len(data))):
    if (data.iloc[i]['Timestamp'] != df.iloc[i]['Timestamp'] or 
        data.iloc[i]['Satellite_ID'] != df.iloc[i]['Satellite_ID']):
        order_preserved = False
        break

if order_preserved:
    print("   Original row order preserved!")
else:
    print("    Warning: Row order may have changed")

# SUMMARY OF NEW FEATURES
print("\n  Feature Engineering Summary:")
print("="*60)

original_features = data.shape[1]
new_features = df.shape[1] - original_features
print(f"Original features: {original_features}")
print(f"New features created: {new_features}")
print(f"Total features: {df.shape[1]}")

# Count valid values for key features
print("\n Key New Features Created:")
key_features = [
    'Unix_Timestamp', 'Time_Delta',
    'Speed_Discrepancy', 'Total_Speed_Discrepancy',
    'Distance_Moved', 'Acceleration_Magnitude',
    'RF_Signal_Stability', 'RF_SNR_Stability',
    'Power_Light_Ratio', 'Magnetic_Speed_Ratio',
    'Composite_Anomaly_Score'
]

for feat in key_features:
    if feat in df.columns:
        non_null = df[feat].notna().sum()
        print(f"   {feat}: {non_null:,} valid values")

# Save the enhanced dataset
output_file = 'UNSW_IoTSAT_With_Feature_Engineering.csv'
df.to_csv(output_file, index=False)
print(f"\n  Enhanced dataset saved to: {output_file}")
print("    Original row order preserved!")
print("    Data type issues handled!")

# Quick analysis of new features vs attacks
print("\n  Quick Analysis - New Features vs Attack Detection:")
print("="*60)

if 'Composite_Anomaly_Score' in df.columns and 'Attack_Flag' in df.columns:
    df['Attack_Flag'] = pd.to_numeric(df['Attack_Flag'], errors='coerce')
    
    normal_mask = df['Attack_Flag'] == 0
    attack_mask = df['Attack_Flag'] == 1
    
    if normal_mask.sum() > 0 and attack_mask.sum() > 0:
        normal_score = df.loc[normal_mask, 'Composite_Anomaly_Score'].mean()
        attack_score = df.loc[attack_mask, 'Composite_Anomaly_Score'].mean()
        
        if not np.isnan(normal_score) and not np.isnan(attack_score) and normal_score > 0:
            print(f"Composite Anomaly Score:")
            print(f"  Normal traffic: {normal_score:.3f}")
            print(f"  Attack traffic: {attack_score:.3f}")
            print(f"  Ratio: {attack_score/normal_score:.2f}x higher for attacks")

# Final verification
print(f"\n Final Verification:")
print(f"  Original dataset: {len(data):,} rows")
print(f"  Enhanced dataset: {len(df):,} rows")
print(f"  Rows match: {' Yes' if len(data) == len(df) else 'No'}")

print("\n Robust feature engineering complete!")
