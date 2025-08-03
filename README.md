# UNSW-IoTSAT: A CCSDS-Compliant Dataset for IoT-Based Satellite Cybersecurity

UNSW-IoTSAT is a protocol-compliant dataset and toolset developed to support research in cybersecurity for Internet of Things (IoT)-enabled satellite communication systems. It provides a realistic simulation environment for generating telemetry data, injecting cyber-physical attacks, and evaluating intrusion detection models under space-relevant constraints.

This repository includes a collection of Python scripts designed to simulate satellite nodes and ground stations, generate labeled datasets, extract RF and telemetry features, and evaluate the effectiveness of machine learning-based detection algorithms.

## Key Features

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


