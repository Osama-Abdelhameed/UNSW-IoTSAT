# UNSW-IoTSAT Dataset Feature Documentation

## Dataset description

UNSW-IoTSAT is a labelled cybersecurity dataset generated from a **hybrid satellite IoT testbed** that combines real on-board sensor hardware, real CCSDS-compliant BPSK RF transmission over a GNU Radio SDR chain, and a deterministic orbital-trajectory simulator that supplies position and velocity context during ground-based laboratory testing. The dataset supports intrusion detection, anomaly detection, and vulnerability-assessment research in satellite systems.

- **Total records:** 404,798
- **Main-release features:** 109 (49 base + 60 engineered)
- **CCSDS companion features:** 3 (released separately via `CCSDS_field_augmentation.py`)


### Hybrid architecture — what is hardware, what is simulated

UNSW-IoTSAT is produced by a ground-based laboratory testbed that cannot generate genuine orbital motion. To provide realistic satellite-ground geometry, the testbed combines three categories of data sources:

1. **Hardware-measured sensor data** — the Adafruit INA228 (power), LIS3MDL (3-axis magnetometer), and VCNL4040 (ambient-light and proximity) sensors are sampled by the satellite-node software (`Sensors_data_collector.py`) and transmitted in every telemetry frame.
2. **Simulated orbital-trajectory data** — geodetic position (latitude, longitude, altitude) and velocity components are produced by a deterministic sinusoidal LEO-orbit model embedded in the satellite-node software. This choice is intentional: stationary laboratory GPS readings cannot provide the trajectory dynamics needed for realistic slant-range and Doppler modelling. The SparkFun MAX-M10S GNSS receiver is present on the satellite-node hardware but is not sampled during generation (`USE_GPS_MODULE = False` in `Sensors_data_collector.py`); live GNSS integration is planned for the next dataset release.
3. **Real CCSDS RF transmission** — every telemetry record is packed into a CCSDS Transfer Frame (ASM `0x1ACFFC1D` + 241-byte data field + FECF CRC), transmitted over a real BPSK link by the satellite-node GNU Radio flowgraph, and received by the ground-station flowgraph with CCSDS ASM correlation and FECF CRC verification. The `RF_CRC_Errors` and `RF_Sync_Word_Detections` columns are direct outputs of CCSDS-compliant GNU Radio blocks.

Each column in the feature list below carries an explicit **Origin** (`MEASURED`, `COMPUTED`, `SIMULATED`, `LABEL`) and a **Source** (the hardware chip, software module, or script that produces the value). This makes the hardware/simulation boundary transparent to downstream researchers.


### Feature counts by category

| Category         | Base | Engineered | Companion | Total |
|------------------|-----:|-----------:|----------:|------:|
| Metadata         |    3 |          2 |         0 |     5 |
| Telemetry        |   28 |          0 |         0 |    28 |
| RF               |   10 |          0 |         0 |    10 |
| Label (attack)   |    8 |          0 |         0 |     8 |
| Engineered       |    0 |         58 |         0 |    58 |
| CCSDS (companion)|    0 |          0 |         3 |     3 |
| **Total**        | **49** | **60** | **3** | **112** |

### Origin breakdown

| Origin     | Count | Meaning |
|------------|------:|---------|
| MEASURED   |    13 | Sampled directly from hardware sensors or GNU Radio receiver blocks |
| SIMULATED  |     9 | Produced by the orbital-trajectory simulator in the satellite-node software |
| COMPUTED   |    82 | Derived quantities (onboard, ground station, feature-engineering script, or CCSDS companion script) |
| LABEL      |     8 | Attack-injection labels from `EnhancedAttackManager` |
| **Total**  | **112** | |

### Seven engineered-feature subcategories

The 60 engineered features are organised into seven subcategories reflecting their role in anomaly detection:

1. **Temporal (4):** Previous-record anchors for consecutive-record comparisons and inter-record timing.
2. **Physical-consistency (9):** Speed, distance, and acceleration checks between reported and derived kinematic quantities.
3. **Sensor-anomaly (28):** Per-sensor change-rate, 5-sample rolling mean, 5-sample rolling standard deviation, and absolute deviation from rolling mean for 7 sensors (Power_W, Mag_X/Y/Z, Ambient_Light_Lux, Current_A, Shunt_Voltage_V).
4. **RF-stability (4):** Rolling std of signal strength and SNR, effective throughput, and communication efficiency.
5. **Cross-domain (5):** Power-to-light, power-per-signal, magnetic-total, magnetic-to-speed, altitude-to-signal ratios.
6. **Composite (8):** MAD-robust-normalised per-feature deviations and a composite anomaly score.
7. **Engineered-metadata (2):** Parsed datetime (`Timestamp_dt`) and Unix timestamp (`Unix_Timestamp`).

---

## Base features (49)

Column legend: **Origin** — MEASURED (hardware sensor or CCSDS receiver block) / COMPUTED (derived) / SIMULATED (orbital-trajectory simulator) / LABEL (attack-injection label).

### Metadata (3 base features)

| # | Name | Type | Unit | Origin | Source | Description |
|---|------|------|------|--------|--------|-------------|
| 1 | `Timestamp` | string | ISO 8601 | MEASURED | Satellite node | Timestamp of the telemetry record at the time of sensor sampling on the satellite node. |
| 2 | `Satellite_ID` | string | -- | MEASURED | Satellite node | Identifier of the source satellite node (Satellite1 or Satellite2). |
| 29 | `Reception_Time` | string | ISO 8601 | MEASURED | Ground station | Timestamp at which the ground station logged the received CCSDS frame. |

(`Vertical_Category` is a Metadata-class string label — see `# 27` below under Telemetry.)

### Telemetry — hardware-measured (8 base features)

| # | Name | Type | Unit | Origin | Source | Description |
|---|------|------|------|--------|--------|-------------|
| 3 | `Shunt_Voltage_V` | float | V | MEASURED | Adafruit INA228 | Shunt voltage across the INA228 power-monitoring shunt resistor. |
| 4 | `Current_A` | float | A | MEASURED | Adafruit INA228 | Instantaneous current draw measured by the INA228 power sensor. |
| 5 | `Power_W` | float | W | MEASURED | Adafruit INA228 | Instantaneous power consumption reported by the INA228 power sensor. |
| 6 | `Mag_X_uT` | float | µT | MEASURED | Adafruit LIS3MDL | Magnetic field strength along the X-axis from the LIS3MDL 3-axis magnetometer. |
| 7 | `Mag_Y_uT` | float | µT | MEASURED | Adafruit LIS3MDL | Magnetic field strength along the Y-axis from the LIS3MDL 3-axis magnetometer. |
| 8 | `Mag_Z_uT` | float | µT | MEASURED | Adafruit LIS3MDL | Magnetic field strength along the Z-axis from the LIS3MDL 3-axis magnetometer. |
| 9 | `Proximity` | int | count | MEASURED | Adafruit VCNL4040 | Proximity reading from the VCNL4040 sensor (raw sensor counts). |
| 10 | `Ambient_Light_Lux` | float | lux | MEASURED | Adafruit VCNL4040 | Ambient light intensity from the VCNL4040 ambient-light sensor. |

### Telemetry — onboard-computed (4 base features)

| # | Name | Type | Unit | Origin | Source | Description |
|---|------|------|------|--------|--------|-------------|
| 11 | `Magnetic_Magnitude_uT` | float | µT | COMPUTED | Satellite node | sqrt(Mag_X² + Mag_Y² + Mag_Z²); computed onboard from LIS3MDL readings. |
| 12 | `Power_Density` | float | W/A | COMPUTED | Satellite node | Power_W / \|Current_A\|. |
| 13 | `Light_Proximity_Ratio` | float | ratio | COMPUTED | Satellite node | Ambient_Light_Lux / Proximity. |
| 14 | `Power_Magnetic_Ratio` | float | ratio | COMPUTED | Satellite node | \|Power_W\| / Magnetic_Magnitude_uT. |

### Telemetry — simulated orbital trajectory (9 base features)

| # | Name | Type | Unit | Origin | Source | Description |
|---|------|------|------|--------|--------|-------------|
| 15 | `Latitude` | float | deg | SIMULATED | Orbital-trajectory simulator | Geodetic latitude from the sinusoidal LEO-orbit simulator embedded in the satellite-node software. |
| 16 | `Longitude` | float | deg | SIMULATED | Orbital-trajectory simulator | Geodetic longitude from the LEO-orbit simulator. |
| 17 | `Altitude_m` | float | m | SIMULATED | Orbital-trajectory simulator | Altitude from the LEO-orbit simulator (base ~400 km with periodic variation). |
| 18 | `GPS_Satellites` | int | count | SIMULATED | Orbital-trajectory simulator | Simulated GPS-in-view count (fixed at 8 during generation). |
| 19 | `GPS_Fix_Quality` | int | code | SIMULATED | Orbital-trajectory simulator | Simulated GPS fix-quality code (fixed at 1 during generation). |
| 20 | `Velocity_North_ms` | float | m/s | SIMULATED | Orbital-trajectory simulator | North velocity derived from simulated position history via a 3-sample sliding window. |
| 21 | `Velocity_East_ms` | float | m/s | SIMULATED | Orbital-trajectory simulator | East velocity derived from simulated position history via a 3-sample sliding window. |
| 22 | `Velocity_Up_ms` | float | m/s | SIMULATED | Orbital-trajectory simulator | Vertical velocity derived from simulated position history via a 3-sample sliding window. |
| 23 | `Speed_ms` | float | m/s | SIMULATED | Orbital-trajectory simulator | sqrt(V_N² + V_E² + V_U²); scalar speed from simulated velocity components. |

### Telemetry — ground-station-computed from simulated trajectory (5 base features)

| # | Name | Type | Unit | Origin | Source | Description |
|---|------|------|------|--------|--------|-------------|
| 24 | `Position_Anomaly` | int | flag | COMPUTED | Satellite node | Flag (0/1) raised when simulated speed exceeds 10 km/s; coarse position-spoofing guard. |
| 25 | `Distance_From_Origin` | float | deg | COMPUTED | Ground-station collector | sqrt(Latitude² + Longitude²); coarse distance indicator. |
| 26 | `Velocity_Bearing_deg` | float | deg | COMPUTED | Ground-station collector | atan2(Velocity_East, Velocity_North) in degrees. |
| 27 | `Vertical_Category` | string | category | COMPUTED | Ground-station collector | Label ("ascending" / "descending" / "stable") based on Velocity_Up_ms. |
| 28 | `Horizontal_Speed_ms` | float | m/s | COMPUTED | Ground-station collector | sqrt(V_North² + V_East²); horizontal-speed magnitude. |

### Telemetry — ground-station quality indicator (1 base feature)

| # | Name | Type | Unit | Origin | Source | Description |
|---|------|------|------|--------|--------|-------------|
| 30 | `Data_Quality_Score` | float | [0–1] | COMPUTED | Ground-station collector | Data-quality score for the telemetry record (default 1.0 if no validation failures). |

### RF — CCSDS-compliant receiver outputs (2 base features, MEASURED)

| # | Name | Type | Unit | Origin | Source | Description |
|---|------|------|------|--------|--------|-------------|
| 38 | `RF_CRC_Errors` | int | flag | MEASURED | GNU Radio CCSDS FECF CRC | Per-frame CCSDS Frame Error Control Field CRC outcome from `digital.crc32_bb` in `Ground_Station.grc`: 0 = valid, 1 = FECF CRC failure. |
| 39 | `RF_Sync_Word_Detections` | int | flag | MEASURED | GNU Radio CCSDS ASM correlator | Per-frame CCSDS Attached Sync Marker (ASM, 0x1ACFFC1D) detection flag from `digital.correlate_access_code_bb_ts`: 1 = ASM locked, 0 = no lock. |

### RF — simulator-generated link metrics (8 base features, COMPUTED)

| # | Name | Type | Unit | Origin | Source | Description |
|---|------|------|------|--------|--------|-------------|
| 31 | `RF_Signal_Strength_dBm` | float | dBm | COMPUTED | EnhancedRFAnalyzer | Per-frame signal-strength scalar driven by slant-range geometry (Eq. 2). |
| 32 | `RF_SNR_dB` | float | dB | COMPUTED | EnhancedRFAnalyzer | Per-frame SNR-shaped scalar (Eq. 2). |
| 33 | `RF_Bit_Error_Rate` | float | ratio | COMPUTED | EnhancedRFAnalyzer | Per-frame BER computed from RF_SNR_dB via Eq. 4. |
| 34 | `RF_Packet_Error_Rate` | float | ratio | COMPUTED | EnhancedRFAnalyzer | Per-frame PER from BER and 1928-bit CCSDS frame length (Eq. 5). |
| 35 | `RF_Throughput_bps` | float | bps | COMPUTED | EnhancedRFAnalyzer | R_eff = 12500 × (1 − PER) bps. |
| 36 | `RF_Frequency_Offset_Hz` | float | Hz | COMPUTED | EnhancedRFAnalyzer | Local-oscillator drift sampled from U(−100, +100) Hz per frame. |
| 37 | `RF_Doppler_Shift_Hz` | float | Hz | COMPUTED | EnhancedRFAnalyzer | Slant-range-scaled Doppler offset sampled per frame (Eq. 9). |
| 40 | `RF_Constellation_Error` | float | ratio | COMPUTED | EnhancedRFAnalyzer | EVM-like indicator: 1–5% for high SNR, 5–15% for mid-range, 15–50% for low SNR. |
| 41 | `Quality_Score` | float | [0–1] | COMPUTED | EnhancedRFAnalyzer | Q = min(1, max(0, (SNR − 10)/50)). |

### Label — attack injection (8 base features, LABEL)

| # | Name | Type | Unit | Origin | Source | Description |
|---|------|------|------|--------|--------|-------------|
| 42 | `Attack_Flag` | int | flag | LABEL | EnhancedAttackManager | Binary attack indicator (0 = normal, 1 = attack). |
| 43 | `Attack_Type` | string | -- | LABEL | EnhancedAttackManager | High-level attack category (normal, jamming, spoofing, dos, mitm, replay, eavesdropping). |
| 44 | `Attack_Subtype` | string | -- | LABEL | EnhancedAttackManager | Specific attack subtype within the broader Attack_Type. |
| 45 | `Attack_Severity` | float | [0–1] | LABEL | EnhancedAttackManager | Normalised attack-severity score assigned at injection. |
| 46 | `Attack_Duration` | float | s | LABEL | EnhancedAttackManager | Duration of the currently active attack instance. |
| 47 | `Attack_Source_Type` | string | -- | LABEL | EnhancedAttackManager | Symbolic label for the attack source (network_flood, rf_transmitter, signal_generator, ...). |
| 48 | `Detection_Confidence` | float | [0–1] | LABEL | EnhancedAttackManager | Synthetic detection-confidence score for observability analysis. |
| 49 | `Attack_Timeline_ID` | string | -- | LABEL | EnhancedAttackManager | Unique identifier linking consecutive records that belong to the same attack instance. |

---

## Engineered features (60)

All engineered features are produced by `Feature_Engineering_UNSW-IoTSAT.py` from the 49 base features; the script preserves the original row order of the input dataset.

### 7.1 Engineered-metadata (2 features)

| # | Name | Type | Unit | Description |
|---|------|------|------|-------------|
| 50 | `Timestamp_dt` | datetime | -- | Parsed datetime object from the `Timestamp` string. |
| 51 | `Unix_Timestamp` | int | s | `Timestamp_dt` converted to seconds since the Unix epoch. |

### 7.2 Temporal (4 features)

| # | Name | Type | Unit | Description |
|---|------|------|------|-------------|
| 52 | `Time_Delta` | float | s | Seconds elapsed since the previous record from the same `Satellite_ID`. |
| 57 | `Latitude_prev` | float | deg | Latitude of the previous record from the same `Satellite_ID` (shift-by-1). |
| 58 | `Longitude_prev` | float | deg | Longitude of the previous record from the same `Satellite_ID` (shift-by-1). |
| 59 | `Altitude_m_prev` | float | m | Altitude of the previous record from the same `Satellite_ID` (shift-by-1). |

### 7.3 Physical-consistency (9 features)

| # | Name | Type | Unit | Formula / Description |
|---|------|------|------|-----------------------|
| 53 | `Calculated_Horizontal_Speed` | float | m/s | sqrt(V_N² + V_E²) from simulated-trajectory velocity components. |
| 54 | `Horizontal_Speed_ms_numeric` | float | m/s | `Horizontal_Speed_ms` cast to float for downstream numeric operations. |
| 55 | `Speed_Discrepancy` | float | m/s | \|Calculated_Horizontal_Speed − Horizontal_Speed_ms_numeric\|. |
| 56 | `Total_Speed_Discrepancy` | float | m/s | \|Speed_ms − sqrt(V_N² + V_E² + V_U²)\|. |
| 60 | `Distance_Moved` | float | km | Great-circle-approximated distance between consecutive simulated GPS positions. |
| 61 | `Velocity_North_ms_change` | float | m/s | Difference of `Velocity_North_ms` between consecutive records from the same satellite. |
| 62 | `Velocity_East_ms_change` | float | m/s | Difference of `Velocity_East_ms` between consecutive records from the same satellite. |
| 63 | `Velocity_Up_ms_change` | float | m/s | Difference of `Velocity_Up_ms` between consecutive records from the same satellite. |
| 64 | `Acceleration_Magnitude` | float | m/s² | sqrt(dV_N² + dV_E² + dV_U²) / `Time_Delta`. |

### 7.4 Sensor-anomaly (28 features)

For each of seven sensors — `Power_W`, `Mag_X_uT`, `Mag_Y_uT`, `Mag_Z_uT`, `Ambient_Light_Lux`, `Current_A`, `Shunt_Voltage_V` — four engineered features are produced. All rolling windows are 5-sample and are computed per `Satellite_ID`.

| Suffix | Type | Formula |
|--------|------|---------|
| `_change_rate` | float | (current − previous) / Time_Delta |
| `_rolling_mean` | float | 5-sample rolling mean, per satellite |
| `_rolling_std` | float | 5-sample rolling standard deviation, per satellite |
| `_deviation` | float | \|current − rolling_mean\| |

Complete list (28 features, in order of appearance): `Power_W_change_rate`, `Power_W_rolling_mean`, `Power_W_rolling_std`, `Power_W_deviation`, `Mag_X_uT_change_rate`, `Mag_X_uT_rolling_mean`, `Mag_X_uT_rolling_std`, `Mag_X_uT_deviation`, `Mag_Y_uT_change_rate`, `Mag_Y_uT_rolling_mean`, `Mag_Y_uT_rolling_std`, `Mag_Y_uT_deviation`, `Mag_Z_uT_change_rate`, `Mag_Z_uT_rolling_mean`, `Mag_Z_uT_rolling_std`, `Mag_Z_uT_deviation`, `Ambient_Light_Lux_change_rate`, `Ambient_Light_Lux_rolling_mean`, `Ambient_Light_Lux_rolling_std`, `Ambient_Light_Lux_deviation`, `Current_A_change_rate`, `Current_A_rolling_mean`, `Current_A_rolling_std`, `Current_A_deviation`, `Shunt_Voltage_V_change_rate`, `Shunt_Voltage_V_rolling_mean`, `Shunt_Voltage_V_rolling_std`, `Shunt_Voltage_V_deviation`.

### 7.5 RF-stability (4 features)

| # | Name | Type | Unit | Formula |
|---|------|------|------|---------|
| 93 | `RF_Signal_Stability` | float | dB | 5-sample rolling std of `RF_Signal_Strength_dBm` per satellite. |
| 94 | `RF_SNR_Stability` | float | dB | 5-sample rolling std of `RF_SNR_dB` per satellite. |
| 95 | `Effective_Throughput` | float | bps | `RF_Throughput_bps` × (1 − `RF_Packet_Error_Rate`). |
| 96 | `Communication_Efficiency` | float | bps/dB | `Effective_Throughput` / (`RF_Signal_Strength_dBm` + 100). |

### 7.6 Cross-domain (5 features)

| # | Name | Type | Unit | Formula |
|---|------|------|------|---------|
| 97 | `Power_Light_Ratio` | float | W/lux | `Power_W` / (`Ambient_Light_Lux` + 1). |
| 98 | `Power_Per_Signal` | float | W/dB | `Power_W` / (`RF_Signal_Strength_dBm` + 100). |
| 99 | `Magnetic_Total` | float | µT | sqrt(Mag_X² + Mag_Y² + Mag_Z²) computed in post-processing. |
| 100 | `Magnetic_Speed_Ratio` | float | µT / (m/s) | `Magnetic_Total` / (`Speed_ms` + 1). |
| 101 | `Altitude_Signal_Ratio` | float | m/dB | `Altitude_m` / (`RF_Signal_Strength_dBm` + 100). |

### 7.7 Composite (8 features)

Each of the selected deviation features is robustly scaled using median and median-absolute-deviation (MAD):

```
z(f) = (f − median(f)) / MAD(f)
```

| # | Name | Formula |
|---|------|---------|
| 102 | `Speed_Discrepancy_normalized` | z(`Speed_Discrepancy`) |
| 103 | `Mag_X_uT_deviation_normalized` | z(`Mag_X_uT_deviation`) |
| 104 | `Mag_Y_uT_deviation_normalized` | z(`Mag_Y_uT_deviation`) |
| 105 | `Mag_Z_uT_deviation_normalized` | z(`Mag_Z_uT_deviation`) |
| 106 | `Ambient_Light_Lux_deviation_normalized` | z(`Ambient_Light_Lux_deviation`) |
| 107 | `Current_A_deviation_normalized` | z(`Current_A_deviation`) |
| 108 | `Shunt_Voltage_V_deviation_normalized` | z(`Shunt_Voltage_V_deviation`) |
| 109 | `Composite_Anomaly_Score` | mean of \|normalised-deviation\| across the seven features above. |

---

## CCSDS companion features (3)

Released separately via `CCSDS_field_augmentation.py`, these three columns expose CCSDS framing sub-fields at the dataset-column level. They are derived deterministically from the attack-label state and per-satellite row order; they do **not** modify the 109-feature main release used for baseline ML training.

| # | Name | Type | Range | Description |
|---|------|------|-------|-------------|
| 110 | `CCSDS_MC_Frame_Count` | int | 0–255 | 8-bit Master Channel Frame Counter per satellite. Monotonic under nominal operation; exhibits gaps under jamming/DoS and duplicates under replay. |
| 111 | `CCSDS_Packet_Sequence_Count` | int | 0–16383 | 14-bit CCSDS Space Packet Sequence Count. Same anomaly behaviour as `CCSDS_MC_Frame_Count`. |
| 112 | `CCSDS_APID` | int | 0–2047 | 11-bit Application Process Identifier. Nominal per satellite (257 / 258 in the released data); anomalous value 0x7FF (2047) substituted under beacon/command spoofing. |

---

## Reproducibility and provenance

| Artefact | Produces | Notes |
|---|---|---|
| `Sensors_data_collector.py` | Per-sample CSV rows (hardware sensors + simulated trajectory) | Runs on each satellite-node Raspberry Pi. |
| `received_sensors_data_on_satellite_node.py` | Collects decoded telemetry at the satellite node after CCSDS RF reception | Adds reception time and data-quality indicator. |
| `combined_data_from_satellite_nodes_on_GS.py` | Merges satellite-node streams on the ground station | Adds `Reception_Time`, CCSDS framing context. |
| `Satellite_1.grc` / `Satellite_2.grc` / `Ground_Station.grc` | Real CCSDS BPSK RF transmission and reception | GNU Radio flowgraphs with CCSDS ASM and FECF CRC blocks. |
| `UNSW-IoTSAT_dataset_generator.py` | Attack injection + RF-metric scoring | EnhancedRFAnalyzer produces `RF_*` link metrics (except ASM / FECF, which come from GNU Radio). EnhancedAttackManager produces `Attack_*` labels. |
| `Feature_Engineering_UNSW-IoTSAT.py` | 60 engineered features from the 49 base features | Preserves original row order. |
| `CCSDS_field_augmentation.py` | 3 CCSDS companion columns from the 109-feature release | Deterministic given the attack labels; seeded random generator for reproducibility. |

All scripts are included in the public GitHub repository: https://github.com/Osama-Abdelhameed/UNSW-IoTSAT.
