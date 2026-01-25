# UNSW-IoTSAT Dataset Feature Documentation

## Dataset Description

UNSW-IoTSAT is a labeled cybersecurity dataset generated from a hybrid satellite IoT testbed integrating sensor telemetry, CCSDS-compliant RF communication, and controlled cyber-physical attack injection. It supports intrusion detection, anomaly detection, and vulnerability assessment research in satellite systems.

**Total records:** 404,798  
**Total features:** 109

## Feature List

| Name | Category | Type | Description |
|---|---|---|---|
| `Timestamp` | Telemetry | string | Timestamp of the telemetry record. |
| `Satellite_ID` | Metadata | string | Identifier associated with this telemetry record. |
| `Shunt_Voltage_V` | Telemetry | float | Shunt voltage v measurement (V). |
| `Current_A` | Telemetry | float | Current a measurement (A). |
| `Power_W` | Telemetry | float | Power w measurement (W). |
| `Mag_X_uT` | Telemetry | float | Mag x ut measurement (µT). |
| `Mag_Y_uT` | Telemetry | float | Mag y ut measurement (µT). |
| `Mag_Z_uT` | Telemetry | float | Mag z ut measurement (µT). |
| `Proximity` | Telemetry | float | Proximity measurement. |
| `Ambient_Light_Lux` | Telemetry | float | Ambient light lux measurement (lux). |
| `Magnetic_Magnitude_uT` | Telemetry | float | Magnetic magnitude ut measurement (µT). |
| `Power_Density` | Telemetry | float | Power density measurement. |
| `Light_Proximity_Ratio` | Engineered | float | Light proximity ratio measurement. |
| `Power_Magnetic_Ratio` | Engineered | float | Power magnetic ratio measurement. |
| `Latitude` | Telemetry | float | Latitude measurement. |
| `Longitude` | Telemetry | float | Longitude measurement. |
| `Altitude_m` | Telemetry | float | Altitude m measurement (m). |
| `GPS_Satellites` | Telemetry | float | Gps satellites measurement. |
| `GPS_Fix_Quality` | Telemetry | float | Gps fix quality measurement. |
| `Velocity_North_ms` | Telemetry | float | Velocity north ms measurement (m/s). |
| `Velocity_East_ms` | Telemetry | float | Velocity east ms measurement (m/s). |
| `Velocity_Up_ms` | Telemetry | float | Velocity up ms measurement (m/s). |
| `Speed_ms` | Telemetry | float | Speed ms measurement (m/s). |
| `Position_Anomaly` | Telemetry | float | Position anomaly measurement. |
| `Distance_From_Origin` | Telemetry | float | Distance from origin measurement. |
| `Velocity_Bearing_deg` | Telemetry | float | Velocity bearing deg measurement. |
| `Vertical_Category` | Telemetry | float | Vertical category measurement. |
| `Horizontal_Speed_ms` | Telemetry | string | Horizontal speed ms measurement (m/s). |
| `Reception_Time` | Metadata | string | Reception time measurement. |
| `Data_Quality_Score` | Engineered | float | Data quality score measurement. |
| `RF_Signal_Strength_dBm` | RF | float | Signal strength dbm (dBm). |
| `RF_SNR_dB` | RF | float | Snr db (dB). |
| `RF_Bit_Error_Rate` | RF | float | Bit error rate. |
| `RF_Packet_Error_Rate` | RF | float | Packet error rate. |
| `RF_Throughput_bps` | RF | float | Throughput bps (bps). |
| `RF_Frequency_Offset_Hz` | RF | float | Frequency offset hz (Hz). |
| `RF_Doppler_Shift_Hz` | RF | float | Doppler shift hz (Hz). |
| `RF_CRC_Errors` | RF | int | Crc errors. |
| `RF_Sync_Word_Detections` | RF | int | Sync word detections. |
| `RF_Constellation_Error` | RF | float | Constellation error. |
| `Quality_Score` | Engineered | float | Overall RF link quality score (0–1). |
| `Attack_Flag` | Attack | int | Binary attack indicator (0 = normal, 1 = attack). |
| `Attack_Type` | Attack | string | High-level attack category label. |
| `Attack_Subtype` | Attack | string | Specific attack subtype label. |
| `Attack_Severity` | Attack | float | Normalized attack severity score (0–1). |
| `Attack_Duration` | Attack | float | Attack duration measurement. |
| `Attack_Source_Type` | Attack | string | Attack source type measurement. |
| `Detection_Confidence` | Telemetry | float | Model confidence score for attack detection (0–1). |
| `Attack_Timeline_ID` | Attack | string | Identifier associated with this telemetry record. |
| `Timestamp_dt` | Metadata | string | Timestamp dt measurement. |
| `Unix_Timestamp` | Engineered | int | Unix timestamp measurement. |
| `Time_Delta` | Engineered | float | Time delta measurement. |
| `Calculated_Horizontal_Speed` | Engineered | float | Calculated horizontal speed measurement. |
| `Horizontal_Speed_ms_numeric` | Engineered | float | Horizontal speed ms numeric measurement. |
| `Speed_Discrepancy` | Engineered | float | Speed discrepancy measurement. |
| `Total_Speed_Discrepancy` | Engineered | float | Total speed discrepancy measurement. |
| `Latitude_prev` | Telemetry | float | Latitude prev measurement. |
| `Longitude_prev` | Telemetry | float | Longitude prev measurement. |
| `Altitude_m_prev` | Telemetry | float | Altitude m prev measurement. |
| `Distance_Moved` | Telemetry | float | Distance moved measurement. |
| `Velocity_North_ms_change` | Telemetry | float | Velocity north ms change measurement. |
| `Velocity_East_ms_change` | Telemetry | float | Velocity east ms change measurement. |
| `Velocity_Up_ms_change` | Telemetry | float | Velocity up ms change measurement. |
| `Acceleration_Magnitude` | Engineered | float | Acceleration magnitude measurement. |
| `Power_W_change_rate` | Telemetry | float | Power w change rate measurement. |
| `Power_W_rolling_mean` | Engineered | float | Five-sample rolling mean of the corresponding feature. |
| `Power_W_rolling_std` | Engineered | float | Five-sample rolling standard deviation of the corresponding feature. |
| `Power_W_deviation` | Engineered | float | Deviation from rolling mean of the corresponding feature. |
| `Mag_X_uT_change_rate` | Telemetry | float | Mag x ut change rate measurement. |
| `Mag_X_uT_rolling_mean` | Engineered | float | Five-sample rolling mean of the corresponding feature. |
| `Mag_X_uT_rolling_std` | Engineered | float | Five-sample rolling standard deviation of the corresponding feature. |
| `Mag_X_uT_deviation` | Engineered | float | Deviation from rolling mean of the corresponding feature. |
| `Mag_Y_uT_change_rate` | Telemetry | float | Mag y ut change rate measurement. |
| `Mag_Y_uT_rolling_mean` | Engineered | float | Five-sample rolling mean of the corresponding feature. |
| `Mag_Y_uT_rolling_std` | Engineered | float | Five-sample rolling standard deviation of the corresponding feature. |
| `Mag_Y_uT_deviation` | Engineered | float | Deviation from rolling mean of the corresponding feature. |
| `Mag_Z_uT_change_rate` | Telemetry | float | Mag z ut change rate measurement. |
| `Mag_Z_uT_rolling_mean` | Engineered | float | Five-sample rolling mean of the corresponding feature. |
| `Mag_Z_uT_rolling_std` | Engineered | float | Five-sample rolling standard deviation of the corresponding feature. |
| `Mag_Z_uT_deviation` | Engineered | float | Deviation from rolling mean of the corresponding feature. |
| `Ambient_Light_Lux_change_rate` | Telemetry | float | Ambient light lux change rate measurement. |
| `Ambient_Light_Lux_rolling_mean` | Engineered | float | Five-sample rolling mean of the corresponding feature. |
| `Ambient_Light_Lux_rolling_std` | Engineered | float | Five-sample rolling standard deviation of the corresponding feature. |
| `Ambient_Light_Lux_deviation` | Engineered | float | Deviation from rolling mean of the corresponding feature. |
| `Current_A_change_rate` | Telemetry | float | Current a change rate measurement. |
| `Current_A_rolling_mean` | Engineered | float | Five-sample rolling mean of the corresponding feature. |
| `Current_A_rolling_std` | Engineered | float | Five-sample rolling standard deviation of the corresponding feature. |
| `Current_A_deviation` | Engineered | float | Deviation from rolling mean of the corresponding feature. |
| `Shunt_Voltage_V_change_rate` | Telemetry | float | Shunt voltage v change rate measurement. |
| `Shunt_Voltage_V_rolling_mean` | Engineered | float | Five-sample rolling mean of the corresponding feature. |
| `Shunt_Voltage_V_rolling_std` | Engineered | float | Five-sample rolling standard deviation of the corresponding feature. |
| `Shunt_Voltage_V_deviation` | Engineered | float | Deviation from rolling mean of the corresponding feature. |
| `RF_Signal_Stability` | RF | float | Signal stability. |
| `RF_SNR_Stability` | RF | float | Snr stability. |
| `Effective_Throughput` | Telemetry | float | Effective throughput measurement. |
| `Communication_Efficiency` | Engineered | float | Communication efficiency measurement. |
| `Power_Light_Ratio` | Engineered | float | Power light ratio measurement. |
| `Power_Per_Signal` | RF | float | Power per signal measurement. |
| `Magnetic_Total` | Telemetry | float | Magnetic total measurement. |
| `Magnetic_Speed_Ratio` | Engineered | float | Magnetic speed ratio measurement. |
| `Altitude_Signal_Ratio` | Engineered | float | Altitude signal ratio measurement. |
| `Speed_Discrepancy_normalized` | Engineered | float | Speed discrepancy normalized measurement. |
| `Mag_X_uT_deviation_normalized` | Engineered | float | Deviation from rolling mean of the corresponding feature. |
| `Mag_Y_uT_deviation_normalized` | Engineered | float | Deviation from rolling mean of the corresponding feature. |
| `Mag_Z_uT_deviation_normalized` | Engineered | float | Deviation from rolling mean of the corresponding feature. |
| `Ambient_Light_Lux_deviation_normalized` | Engineered | float | Deviation from rolling mean of the corresponding feature. |
| `Current_A_deviation_normalized` | Engineered | float | Deviation from rolling mean of the corresponding feature. |
| `Shunt_Voltage_V_deviation_normalized` | Engineered | float | Deviation from rolling mean of the corresponding feature. |
| `Composite_Anomaly_Score` | Engineered | float | Composite anomaly score measurement. |
