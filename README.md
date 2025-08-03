# UNSW-IoTSAT: A CCSDS-Compliant Dataset for IoT-Based Satellite Cybersecurity

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![Dataset](https://img.shields.io/badge/dataset-available-green.svg)](#repository-contents)

**UNSW-IoTSAT** is a protocol-compliant dataset and toolset developed to support research in cybersecurity for Internet of Things (IoT)-enabled satellite communication systems. It provides a realistic simulation environment for generating telemetry data, injecting cyber-physical attacks, and evaluating intrusion detection models under space-relevant constraints.

This repository includes a collection of Python scripts designed to simulate satellite nodes and ground stations, generate labeled datasets, extract RF and telemetry features, and evaluate the effectiveness of machine learning-based detection algorithms.

---

##  Key Features

- **Realistic Simulation of Space-to-Ground Communication**  
  Implements a multi-node setup using CCSDS-compliant telemetry formats to simulate satellite and ground station interactions.

- **Cyber-Physical Attack Injection**  
  Models six categories of attacks relevant to satellite communications: jamming, spoofing, replay, denial-of-service, man-in-the-middle, and eavesdropping.

- **High-Fidelity Dataset Generation**  
  Produces a labeled dataset with detailed telemetry, RF metrics, attack types, severity annotations, and detection confidence scores.

- **Feature Engineering and Statistical Analysis**  
  Includes scripts to extract time-series, RF, and system-level features, visualize trends, and quantify attack impact.

- **Machine Learning Evaluation**  
  Supports supervised classification using models such as Random Forest, XGBoost, and Neural Networks. Evaluation includes confusion matrices, ROC/PR curves, and feature importance plots.

---

##  System Requirements and Setup

### Hardware Components

- **Ground Segment:**  
  Ubuntu 22.04 LTS workstation representing a ground station responsible for receiving and processing satellite telemetry data.

- **Space Segments:**  
  Two Ubuntu 22.04 LTS workstations simulating independent satellite nodes.

- **Sensor Platform:**  
  Raspberry Pi 4 Model B with:
  - Adafruit PCT2075 (Temperature)
  - Adafruit INA228 (Current/Voltage)
  - Adafruit LIS3MDL (Magnetometer)
  - Adafruit BNO055 (9-DoF Orientation)
  - Adafruit ICM-20948 (IMU)

- **Software-Defined Radios (SDRs):**  
  bladeRF x40 devices to establish flexible wireless communication between satellite and ground nodes.

---

###  Software Requirements

Install the following packages:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install gnuradio -y
sudo apt install cmake build-essential git pkg-config \
  libusb-1.0-0-dev libtecla-dev doxygen libsqlite3-dev \
  python3-dev python3-numpy python3-pip soapysdr-module-all \
  libncurses5-dev libncursesw5-dev libcurl4-openssl-dev \
  libsoapysdr-dev
```

###  bladeRF x40 Installation

```bash
cd ~
git clone https://github.com/Nuand/bladeRF.git
cd bladeRF
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install
sudo ldconfig
```

###  Install SoapySDR Module for bladeRF

```bash
cd ~
git clone https://github.com/pothosware/SoapyBladeRF.git
cd SoapyBladeRF
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install
sudo ldconfig
```

###  Update FPGA and Firmware

```bash
bladeRF-cli -f <path/to/bladeRF_fw_v2.6.0.img>
bladeRF-cli -l <path/to/hostedx40.rbf>
bladeRF-cli -e version
```

###  Raspberry Pi Sensor Configuration

####  Enable I2C Interface

```bash
sudo raspi-config
# Navigate to: Interfacing Options → I2C → Enable
```

####  Update System and Install Libraries

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip -y
pip3 install adafruit-blinka
pip3 install adafruit-circuitpython-pct2075
pip3 install adafruit-circuitpython-ina228
pip3 install adafruit-circuitpython-lis3mdl
pip3 install adafruit-circuitpython-bno055
pip3 install adafruit-circuitpython-icm20x
```

####  NTP Clock Synchronization

```bash
sudo apt install ntp -y
sudo systemctl enable ntp
sudo systemctl start ntp
```

Ensures all satellite and ground station nodes have synchronized timestamps for accurate telemetry and RF logging.

---

## Repository Contents

| Script Name | Purpose |
|-------------|---------|
| `Sensors_data_collector.py` | Collects sensor data on Raspberry Pi and transmits via serial |
| `received_sensors_data_on_satellite_node.py` | Receives sensor data from Raspberry Pi on satellite node |
| `Satellite_1.py` | Simulates telemetry generation and transmission for satellite node 1 |
| `Satellite_2.py` | Simulates telemetry generation and transmission for satellite node 2 |
| `Ground_Station.py` | Receives and decodes telemetry at the ground segment |
| `combined_data_from_satellite_nodes_on_GS.py` | Aggregates data from multiple satellites on the ground station |
| `UNSW-IoTSAT_dataset_generator.py` | Dynamically labels and logs telemetry with attack events |
| `Dataset_statistics_generator.py` | Computes statistical summaries (attack type, duration, severity) |
| `Feature_Engineering_UNSW-IoTSAT.py` | Extracts temporal, RF, and signal integrity features |
| `Generate_Graphes_For_Feature_Engineering.py` | Visualizes telemetry, attack injection, and RF patterns |
| `ML_Models_Attack_Detection.py` | Trains and evaluates detection models (RF, XGBoost, NN, etc.) |

---

##  Example Usage

### Data Collection and Simulation Pipeline

1. **Launch sensor data collection on Raspberry Pi:**
   ```bash
   # On Raspberry Pi (satellite sensor node)
   python3 Sensors_data_collector.py
   ```

2. **Start sensor data reception on satellite node:**
   ```bash
   # On Ubuntu machine (satellite node)
   python3 received_sensors_data_on_satellite_node.py
   ```

3. **Run simulated satellite nodes:**
   ```bash
   python3 Satellite_1.py
   python3 Satellite_2.py
   ```

4. **Start ground station to receive packets and log events:**
   ```bash
   python3 Ground_Station.py
   ```

### Dataset Generation and Analysis

5. **Generate labeled dataset with cyber-physical attacks:**
   ```bash
   python3 UNSW-IoTSAT_dataset_generator.py
   ```

6. **Perform feature extraction and visualization:**
   ```bash
   python3 Feature_Engineering_UNSW-IoTSAT.py
   python3 Generate_Graphes_For_Feature_Engineering.py
   ```

7. **Train and evaluate ML models for anomaly detection:**
   ```bash
   python3 ML_Models_Attack_Detection.py
   ```

###  Data Flow Architecture

```
Raspberry Pi (Sensor Node)         Ubuntu Machine (Satellite Node)      Ubuntu Machine (Ground Station)
┌─────────────────────────┐        ┌──────────────────────────────┐     ┌──────────────────────────────┐
│ Sensors_data_collector.py│ Serial │received_sensors_data_on_     │ RF  │ Ground_Station.py            │
│ • INA228 (Power)        │ ────▶  │  satellite_node.py           │────▶│ • Receives telemetry         │
│ • LIS3MDL (Magnetometer)│        │ • Processes sensor data      │     │ • Attack detection           │
│ • VCNL4040 (Proximity)  │        │ • Satellite_1.py/2.py       │     │ • Dataset generation         │
│ • GPS/Simulated coords  │        │ • Transmits via bladeRF      │     │ • ML model evaluation       │
└─────────────────────────┘        └──────────────────────────────┘     └──────────────────────────────┘
```

---

##  License

This project is licensed under the MIT License. See `LICENSE` file for details.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change or improve.

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes and commit:**
   ```bash
   git commit -m "Add your descriptive commit message"
   ```
4. **Push to your branch:**
   ```bash
   git push origin feature/your-feature-name
   ```
5. **Create a Pull Request**

### Contribution Guidelines

- Ensure your code follows the existing style and conventions
- Add appropriate comments and documentation
- Test your changes thoroughly
- Update the README if necessary
- Include relevant unit tests for new functionality

---

## Contact

For questions, issues, or collaboration opportunities, please contact:

- **Project Team**: o.abdelhameed@unsw.edu.au
- **Institution**: University of New South Wales (UNSW Canberra)
- **Issues**: [GitHub Issues](https://github.com/Osama-Abdelhameed/UNSW-IoTSAT/issues)

---

**If you find this dataset useful for your research, please consider starring this repository and citing our work!**
