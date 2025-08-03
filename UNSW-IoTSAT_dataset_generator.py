#!/usr/bin/env python3
"""
UNSW-IoTSAT_DATASET_GENERATOR
"""

import pandas as pd
import numpy as np
import json
import os
import sys
import time
import signal
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import argparse
import random
from collections import deque, Counter

# Set up logging
def setup_logging():
    """Setup logging configuration"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'realtime_balanced.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class EnhancedRFAnalyzer:
    """Enhanced RF analyzer with realistic attack simulation capabilities"""
    
    def __init__(self, sample_rate=50e3, tx_freq=917e6, rx_freq=915e6, 
                 packet_length=241, tx_gain=55.0, rx_gain=45.0, sps=4):
        self.sample_rate = sample_rate
        self.tx_freq = tx_freq
        self.rx_freq = rx_freq
        self.packet_length = packet_length
        self.tx_gain = tx_gain
        self.rx_gain = rx_gain
        self.sps = sps
        
        # Calculated parameters
        self.symbol_rate = self.sample_rate / self.sps
        self.bit_rate = self.symbol_rate
        
        # Quality tracking for realistic generation
        self.quality_history = deque(maxlen=1000)
        
        logger.info(f"   Enhanced RF Analyzer initialized")
        logger.info(f"   Sample Rate: {self.sample_rate/1000:.1f} kHz")
        logger.info(f"   TX/RX: {self.tx_freq/1e6:.1f}/{self.rx_freq/1e6:.1f} MHz")
    
    def calculate_rf_metrics(self, row_data: Dict, force_quality: str = None) -> Dict:
        """Calculate RF metrics with optional quality forcing for balance"""
        
        # Extract coordinates with defaults
        try:
            lat = float(row_data.get('Latitude', 0))
            lon = float(row_data.get('Longitude', 0))
            alt = float(row_data.get('Altitude_m', 400000))
        except (ValueError, TypeError):
            lat, lon, alt = 0, 0, 400000
        
        # Distance calculation
        if lat == 0 and lon == 0:
            distance_km = np.random.uniform(400, 2000)
        else:
            earth_radius = 6371000
            lat_rad = np.radians(abs(lat))
            lon_rad = np.radians(abs(lon))
            ground_distance = earth_radius * np.sqrt(lat_rad**2 + lon_rad**2)
            distance_km = np.sqrt(ground_distance**2 + alt**2) / 1000
            distance_km = max(distance_km, 300)
        
        # Base RF calculations
        fspl = 20 * np.log10(distance_km * 1000) + 20 * np.log10(self.tx_freq) - 147.55
        tx_power_dbm = 20
        rx_signal_strength = tx_power_dbm + self.tx_gain - fspl + self.rx_gain
        
        # Apply quality forcing for balanced datasets
        if force_quality == 'good':
            rx_signal_strength += np.random.uniform(5, 15)
        elif force_quality == 'poor':
            rx_signal_strength -= np.random.uniform(10, 25)
        else:
            rx_signal_strength += np.random.uniform(-2, 2)
        
        # SNR and dependent calculations
        noise_floor = -114
        snr_db = rx_signal_strength - noise_floor
        
        # BER calculation
        if snr_db > 0:
            snr_linear = 10**(snr_db/10)
            ber = 0.5 * np.exp(-snr_linear * 0.5)
        else:
            ber = 0.5
        
        ber = np.clip(ber, 1e-12, 0.5)
        
        # PER and throughput
        per = 1 - (1 - ber)**(self.packet_length * 8)
        per = min(0.99, per)
        throughput_bps = self.bit_rate * (1 - per)
        
        # Other RF parameters
        freq_offset = np.random.uniform(-100, 100)
        velocity_ms = 7800
        max_doppler = (velocity_ms / 3e8) * self.rx_freq
        doppler_shift = np.random.uniform(-max_doppler, max_doppler)
        
        crc_errors = 1 if np.random.random() < per else 0
        sync_detections = 1 if np.random.random() > per else 0
        
        # Constellation error
        if snr_db > 20:
            const_error = np.random.uniform(0.01, 0.05)
        elif snr_db > 10:
            const_error = np.random.uniform(0.05, 0.15)
        else:
            const_error = np.random.uniform(0.15, 0.5)
        
        # Track quality for balancing
        quality_score = min(1.0, max(0.0, (snr_db - 10) / 50))
        self.quality_history.append(quality_score)
        
        return {
            'RF_Signal_Strength_dBm': round(rx_signal_strength, 2),
            'RF_SNR_dB': round(snr_db, 2),
            'RF_Bit_Error_Rate': ber,
            'RF_Packet_Error_Rate': round(per, 6),
            'RF_Throughput_bps': round(throughput_bps, 2),
            'RF_Frequency_Offset_Hz': round(freq_offset, 2),
            'RF_Doppler_Shift_Hz': round(doppler_shift, 2),
            'RF_CRC_Errors': crc_errors,
            'RF_Sync_Word_Detections': sync_detections,
            'RF_Constellation_Error': round(const_error, 4),
            'Quality_Score': round(quality_score, 3)
        }


class EnhancedAttackManager:
    """Enhanced attack manager with realistic attack patterns and timelines"""
    
    def __init__(self, target_attack_ratio: float = 0.5, balance_window: int = 1000):
        self.target_attack_ratio = target_attack_ratio
        self.balance_window = balance_window
        
        # Track recent records for balance
        self.recent_records = deque(maxlen=balance_window)
        self.attack_history = deque(maxlen=balance_window)
        
        # Enhanced attack types with realistic characteristics
        self.attack_types = ['jamming', 'spoofing', 'replay', 'dos', 'mitm', 'eavesdrop']
        self.attack_subtypes = {
            'jamming': ['continuous', 'pulsed', 'swept', 'barrage'],
            'spoofing': ['gps', 'telemetry', 'command', 'beacon'],
            'replay': ['command', 'telemetry', 'authentication'],
            'dos': ['flooding', 'resource_exhaustion', 'protocol_exploit'],
            'mitm': ['proxy', 'ssl_strip', 'arp_poison'],
            'eavesdrop': ['passive', 'traffic_analysis', 'key_extraction']
        }
        
        # Attack characteristics database
        self.attack_characteristics = {
            'jamming': {
                'typical_duration': (10, 300),
                'severity_range': (0.7, 0.95),
                'detection_difficulty': 0.2,
                'frequency': 0.3
            },
            'spoofing': {
                'typical_duration': (60, 1800),
                'severity_range': (0.6, 0.9),
                'detection_difficulty': 0.6,
                'frequency': 0.25
            },
            'replay': {
                'typical_duration': (5, 120),
                'severity_range': (0.4, 0.7),
                'detection_difficulty': 0.7,
                'frequency': 0.15
            },
            'dos': {
                'typical_duration': (30, 600),
                'severity_range': (0.7, 0.9),
                'detection_difficulty': 0.3,
                'frequency': 0.2
            },
            'mitm': {
                'typical_duration': (120, 3600),
                'severity_range': (0.5, 0.8),
                'detection_difficulty': 0.8,
                'frequency': 0.05
            },
            'eavesdrop': {
                'typical_duration': (300, 7200),
                'severity_range': (0.2, 0.5),
                'detection_difficulty': 0.9,
                'frequency': 0.05
            }
        }
        
        self.attack_type_counter = Counter()
        self.current_attack_sequence = None
        self.sequence_remaining = 0
        self.attack_start_time = None
        self.attack_timeline = None
        
        logger.info(f"   Enhanced Attack Manager initialized")
        logger.info(f"   Target attack ratio: {target_attack_ratio:.1%}")
        logger.info(f"   Balance window: {balance_window} records")
    
    def should_generate_attack(self) -> bool:
        """Determine if next record should be an attack to maintain balance"""
        
        if len(self.recent_records) < 10:
            return random.random() < self.target_attack_ratio
        
        current_attacks = sum(self.attack_history)
        current_ratio = current_attacks / len(self.attack_history)
        
        if current_ratio < self.target_attack_ratio:
            attack_probability = min(0.9, self.target_attack_ratio * 1.5)
        elif current_ratio > self.target_attack_ratio:
            attack_probability = max(0.1, self.target_attack_ratio * 0.5)
        else:
            attack_probability = self.target_attack_ratio
        
        return random.random() < attack_probability
    
    def get_realistic_attack_type(self) -> Tuple[str, str, Dict]:
        """Get attack type with realistic characteristics and timeline"""
        
        if self.sequence_remaining > 0:
            self.sequence_remaining -= 1
            return (*self.current_attack_sequence, self.attack_timeline)
        
        attack_weights = [self.attack_characteristics[atype]['frequency'] for atype in self.attack_types]
        attack_type = np.random.choice(self.attack_types, p=np.array(attack_weights)/sum(attack_weights))
        attack_subtype = random.choice(self.attack_subtypes[attack_type])
        
        self.attack_timeline = self._generate_attack_timeline(attack_type, attack_subtype)
        
        duration_seconds = self.attack_timeline['duration']
        self.sequence_remaining = max(1, int(duration_seconds))
        
        self.attack_type_counter[attack_type] += 1
        self.current_attack_sequence = (attack_type, attack_subtype)
        self.attack_start_time = datetime.now()
        
        return attack_type, attack_subtype, self.attack_timeline
    
    def _generate_attack_timeline(self, attack_type: str, attack_subtype: str) -> Dict:
        """Generate realistic attack timeline and characteristics"""
        
        chars = self.attack_characteristics[attack_type]
        
        min_dur, max_dur = chars['typical_duration']
        duration = np.random.uniform(min_dur, max_dur)
        
        min_sev, max_sev = chars['severity_range']
        severity = np.random.uniform(min_sev, max_sev)
        
        detection_confidence = 1.0 - chars['detection_difficulty'] + np.random.uniform(-0.1, 0.1)
        detection_confidence = np.clip(detection_confidence, 0.1, 0.95)
        
        attack_source = self._generate_attack_source(attack_type)
        
        return {
            'duration': duration,
            'severity': severity,
            'detection_confidence': detection_confidence,
            'start_time': datetime.now().isoformat(),
            'attack_source': attack_source,
            'progression_factor': 0.0
        }
    
    def _generate_attack_source(self, attack_type: str) -> Dict:
        """Generate realistic attack source characteristics"""
        
        sources = {
            'jamming': {
                'type': 'rf_transmitter',
                'power_dbm': np.random.uniform(20, 50),
                'frequency_range': np.random.uniform(10e6, 100e6),
                'location': 'ground_based'
            },
            'spoofing': {
                'type': 'signal_generator',
                'sophistication': np.random.choice(['low', 'medium', 'high'], p=[0.4, 0.4, 0.2]),
                'timing_accuracy': np.random.uniform(0.1, 1.0),
                'location': 'ground_based'
            },
            'dos': {
                'type': 'network_flood',
                'packet_rate': np.random.uniform(1000, 50000),
                'source_ips': np.random.randint(1, 100),
                'location': 'distributed'
            },
            'mitm': {
                'type': 'proxy_server',
                'encryption_bypass': np.random.choice([True, False], p=[0.3, 0.7]),
                'latency_ms': np.random.uniform(50, 500),
                'location': 'ground_based'
            },
            'eavesdrop': {
                'type': 'passive_receiver',
                'sensitivity_dbm': np.random.uniform(-100, -80),
                'bandwidth_mhz': np.random.uniform(1, 20),
                'location': 'mobile'
            },
            'replay': {
                'type': 'record_replay',
                'delay_seconds': np.random.uniform(0.1, 5.0),
                'accuracy': np.random.uniform(0.7, 0.99),
                'location': 'ground_based'
            }
        }
        
        return sources.get(attack_type, {'type': 'unknown', 'location': 'unknown'})
    
    def record_decision(self, is_attack: bool, attack_type: str = None):
        """Record the attack decision for balance tracking"""
        self.recent_records.append(1 if is_attack else 0)
        self.attack_history.append(1 if is_attack else 0)
    
    def get_balance_statistics(self) -> Dict:
        """Get current balance statistics"""
        if len(self.recent_records) == 0:
            return {'current_ratio': 0, 'target_ratio': self.target_attack_ratio, 'records_tracked': 0}
        
        current_attacks = sum(self.attack_history)
        current_ratio = current_attacks / len(self.attack_history)
        
        return {
            'current_ratio': current_ratio,
            'target_ratio': self.target_attack_ratio,
            'records_tracked': len(self.recent_records),
            'attack_distribution': dict(self.attack_type_counter),
            'balance_error': abs(current_ratio - self.target_attack_ratio)
        }


class EnhancedRealTimeBalancedGenerator:
    """Real-time balanced dataset generator"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.input_file = config['input_file']
        self.output_configs = config.get('output_configs', {})
        
        # Initialize enhanced components
        self.rf_analyzer = EnhancedRFAnalyzer(**config.get('rf_config', {}))
        self.attack_managers = {
            name: EnhancedAttackManager(cfg['ratio'], config.get('balance_window', 1000))
            for name, cfg in self.output_configs.items()
        }
        
        # Position persistence
        input_basename = os.path.basename(self.input_file).replace('.csv', '')
        self.position_file = os.path.join("logs", f"position_{input_basename}.json")
        self.state_file = os.path.join("logs", f"state_{input_basename}.json")
        
        # Load last position and state
        self.last_position, self.last_line_count = self._load_last_position()
        
        # Statistics
        self.start_time = datetime.now()
        self.total_processed = 0
        
        # Control
        self.running = False
        
        # Setup output files
        self._setup_output_files()
    
    def _load_last_position(self) -> Tuple[int, int]:
        """Load last processed position from file"""
        try:
            if os.path.exists(self.position_file):
                with open(self.position_file, 'r') as f:
                    data = json.load(f)
                    self.total_processed = data.get('total_processed', 0)
                    logger.info(f" Resuming from line {data['last_line_count']} (processed {self.total_processed} records)")
                    return data['last_position'], data['last_line_count']
            else:
                logger.info(" Starting - no previous position found")
                return 0, 0
        except Exception as e:
            logger.warning(f"Could not load position file: {e}")
            return 0, 0
    
    def _save_position(self):
        """Save current position to file"""
        try:
            position_data = {
                'last_position': self.last_position,
                'last_line_count': self.last_line_count,
                'last_update': datetime.now().isoformat(),
                'total_processed': self.total_processed,
                'input_file': self.input_file
            }
            with open(self.position_file, 'w') as f:
                json.dump(position_data, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save position: {e}")
    
    def _setup_output_files(self):
        """Setup all output files with enhanced headers"""
        try:
            for name, config in self.output_configs.items():
                # Create output directories
                os.makedirs(os.path.dirname(config['csv']), exist_ok=True)
                os.makedirs(os.path.dirname(config['json']), exist_ok=True)
                
                # Check if we're resuming
                csv_exists = os.path.exists(config['csv']) and os.path.getsize(config['csv']) > 0
                json_exists = os.path.exists(config['json']) and os.path.getsize(config['json']) > 0
                
                # Setup CSV files with enhanced headers
                if not csv_exists:
                    with open(config['csv'], 'w') as f:
                        header = [
                            # Original columns
                            "Timestamp", "Satellite_ID", "Shunt_Voltage_V", "Current_A", "Power_W",
                            "Mag_X_uT", "Mag_Y_uT", "Mag_Z_uT", "Proximity", "Ambient_Light_Lux",
                            "Magnetic_Magnitude_uT", "Power_Density", "Light_Proximity_Ratio", 
                            "Power_Magnetic_Ratio", "Latitude", "Longitude", "Altitude_m", 
                            "GPS_Satellites", "GPS_Fix_Quality", "Velocity_North_ms", 
                            "Velocity_East_ms", "Velocity_Up_ms", "Speed_ms", "Position_Anomaly",
                            "Distance_From_Origin", "Velocity_Bearing_deg", "Vertical_Category", 
                            "Horizontal_Speed_ms", "Reception_Time", "Data_Quality_Score",
                            # RF columns
                            'RF_Signal_Strength_dBm', 'RF_SNR_dB', 'RF_Bit_Error_Rate',
                            'RF_Packet_Error_Rate', 'RF_Throughput_bps', 'RF_Frequency_Offset_Hz',
                            'RF_Doppler_Shift_Hz', 'RF_CRC_Errors', 'RF_Sync_Word_Detections',
                            'RF_Constellation_Error', 'Quality_Score',
                            # Enhanced Attack columns
                            'Attack_Flag', 'Attack_Type', 'Attack_Subtype', 'Attack_Severity',
                            'Attack_Duration', 'Attack_Source_Type', 'Detection_Confidence', 'Attack_Timeline_ID'
                        ]
                        f.write(','.join(header) + '\n')
                        logger.info(f" Enhanced CSV header written: {config['csv']}")
                else:
                    logger.info(f" CSV file exists, resuming: {config['csv']}")
                
                # Setup JSON files
                if not json_exists:
                    with open(config['json'], 'w') as f:
                        f.write('[\n')
                    logger.info(f" JSON file initialized: {config['json']}")
                else:
                    logger.info(f" JSON file exists, resuming: {config['json']}")
                    try:
                        with open(config['json'], 'r+') as f:
                            f.seek(0, 2)
                            f.seek(f.tell() - 2)
                            f.truncate()
                    except:
                        pass
            
            logger.info(f"All output files ready for {len(self.output_configs)} enhanced balanced datasets")
            
        except Exception as e:
            logger.error(f"Error setting up output files: {e}")
            raise
    
    def _get_file_position(self):
        """Get current file position and line count"""
        try:
            if not os.path.exists(self.input_file):
                return 0, 0
            
            with open(self.input_file, 'r') as f:
                line_count = sum(1 for _ in f)
                f.seek(0, 2)
                file_size = f.tell()
            
            return file_size, line_count
            
        except Exception as e:
            logger.error(f"Error getting file position: {e}")
            return self.last_position, self.last_line_count
    
    def _read_new_lines(self):
        """Read new lines from input file"""
        try:
            current_pos, current_lines = self._get_file_position()
            
            if current_lines <= self.last_line_count:
                return []
            
            new_lines = []
            with open(self.input_file, 'r') as f:
                # Skip to last processed line
                for i in range(self.last_line_count):
                    f.readline()
                
                # Read new lines
                while True:
                    line = f.readline()
                    if not line:
                        break
                    new_lines.append(line.strip())
            
            # Update position
            self.last_position = current_pos
            self.last_line_count = current_lines
            
            # Save position periodically
            if len(new_lines) > 0 and self.total_processed % 50 == 0:
                self._save_position()
            
            return new_lines
            
        except Exception as e:
            logger.error(f"Error reading new lines: {e}")
            return []
    
    def _parse_csv_line(self, line: str) -> Optional[Dict]:
        """Parse CSV line into dictionary"""
        try:
            if not line or line.startswith('Timestamp') or line.startswith('timestamp'):
                return None
            
            values = [val.strip().strip('"') for val in line.split(',')]
            
            if len(values) < 2:
                return None
            
            # Map to column names
            column_names = [
                "Timestamp", "Satellite_ID", "Shunt_Voltage_V", "Current_A", "Power_W",
                "Mag_X_uT", "Mag_Y_uT", "Mag_Z_uT", "Proximity", "Ambient_Light_Lux",
                "Magnetic_Magnitude_uT", "Power_Density", "Light_Proximity_Ratio", 
                "Power_Magnetic_Ratio", "Latitude", "Longitude", "Altitude_m", 
                "GPS_Satellites", "GPS_Fix_Quality", "Velocity_North_ms", 
                "Velocity_East_ms", "Velocity_Up_ms", "Speed_ms", "Position_Anomaly",
                "Distance_From_Origin", "Velocity_Bearing_deg", "Vertical_Category", 
                "Horizontal_Speed_ms", "Reception_Time", "Data_Quality_Score"
            ]
            
            record = {}
            for i, col_name in enumerate(column_names):
                if i < len(values):
                    record[col_name] = values[i]
                else:
                    record[col_name] = '0' if col_name not in ['Timestamp', 'Satellite_ID', 'Reception_Time'] else ''
            
            return record
            
        except Exception as e:
            logger.warning(f"Error parsing CSV line: {e}")
            return None
    
    def _generate_enhanced_balanced_records(self, base_record: Dict) -> Dict[str, Dict]:
        """Generate enhanced balanced records with realistic attack simulation"""
        
        balanced_records = {}
        
        for name, config in self.output_configs.items():
            attack_manager = self.attack_managers[name]
            
            # Decide if this should be an attack
            should_attack = attack_manager.should_generate_attack()
            
            # Create record copy
            record = base_record.copy()
            
            if should_attack:
                # Generate realistic attack with timeline
                attack_type, attack_subtype, attack_timeline = attack_manager.get_realistic_attack_type()
                
                # Calculate RF metrics with realistic attack effects
                rf_metrics = self.rf_analyzer.calculate_rf_metrics(record, force_quality='poor')
                rf_metrics = self._apply_realistic_attack_effects(
                    rf_metrics, attack_type, attack_subtype, attack_timeline
                )
                
                # Add enhanced attack information
                record['Attack_Flag'] = 1
                record['Attack_Type'] = attack_type
                record['Attack_Subtype'] = attack_subtype
                record['Attack_Severity'] = round(attack_timeline['severity'], 3)
                record['Attack_Duration'] = round(attack_timeline['duration'], 1)
                record['Attack_Source_Type'] = attack_timeline['attack_source']['type']
                record['Detection_Confidence'] = round(attack_timeline['detection_confidence'], 3)
                record['Attack_Timeline_ID'] = f"{attack_type}_{int(time.time())}"
                
            else:
                # Normal record
                rf_metrics = self.rf_analyzer.calculate_rf_metrics(record, force_quality='good')
                record['Attack_Flag'] = 0
                record['Attack_Type'] = 'normal'
                record['Attack_Subtype'] = 'normal'
                record['Attack_Severity'] = 0.0
                record['Attack_Duration'] = 0.0
                record['Attack_Source_Type'] = 'none'
                record['Detection_Confidence'] = 0.95
                record['Attack_Timeline_ID'] = 'normal'
            
            # Add RF metrics
            record.update(rf_metrics)
            
            # Record decision for balance tracking
            attack_manager.record_decision(should_attack, record.get('Attack_Type'))
            
            balanced_records[name] = record
        
        return balanced_records
    
    def _apply_realistic_attack_effects(self, rf_metrics: Dict, attack_type: str, 
                                      attack_subtype: str, attack_timeline: Dict) -> Dict:
        """Apply realistic attack effects based on actual attack characteristics"""
        
        modified_rf = rf_metrics.copy()
        severity = attack_timeline['severity']
        progression = attack_timeline['progression_factor']
        
        # Apply severity scaling
        severity_multiplier = severity * (0.5 + 0.5 * progression)
        
        if attack_type == 'jamming':
            if attack_subtype == 'continuous':
                snr_reduction = 20 + (severity_multiplier * 15)
                modified_rf['RF_SNR_dB'] = max(-25, modified_rf['RF_SNR_dB'] - snr_reduction)
                modified_rf['RF_CRC_Errors'] = 1
                modified_rf['RF_Packet_Error_Rate'] = min(0.95, modified_rf['RF_Packet_Error_Rate'] + 0.8 * severity_multiplier)
                modified_rf['RF_Sync_Word_Detections'] = 0
                modified_rf['RF_Throughput_bps'] = modified_rf['RF_Throughput_bps'] * (1 - severity_multiplier)
                
            elif attack_subtype == 'pulsed':
                if np.random.random() < (0.5 + 0.3 * severity_multiplier):
                    snr_reduction = 15 + (severity_multiplier * 15)
                    modified_rf['RF_SNR_dB'] = max(-15, modified_rf['RF_SNR_dB'] - snr_reduction)
                    modified_rf['RF_Packet_Error_Rate'] = min(0.9, modified_rf['RF_Packet_Error_Rate'] + 0.6 * severity_multiplier)
                    modified_rf['RF_CRC_Errors'] = 1
                
            elif attack_subtype == 'swept':
                freq_offset_range = 300 + (severity_multiplier * 200)
                modified_rf['RF_Frequency_Offset_Hz'] = np.random.uniform(-freq_offset_range, freq_offset_range)
                modified_rf['RF_SNR_dB'] = max(-10, modified_rf['RF_SNR_dB'] - (10 + 10 * severity_multiplier))
                modified_rf['RF_Doppler_Shift_Hz'] = modified_rf['RF_Doppler_Shift_Hz'] + np.random.uniform(-100, 100)
                
            elif attack_subtype == 'barrage':
                snr_reduction = 25 + (severity_multiplier * 15)
                modified_rf['RF_SNR_dB'] = max(-30, modified_rf['RF_SNR_dB'] - snr_reduction)
                modified_rf['RF_Constellation_Error'] = min(0.9, modified_rf['RF_Constellation_Error'] + 0.5 * severity_multiplier)
                modified_rf['RF_Throughput_bps'] = modified_rf['RF_Throughput_bps'] * 0.1
                modified_rf['RF_Packet_Error_Rate'] = 0.95
        
        elif attack_type == 'spoofing':
            if attack_subtype == 'gps':
                modified_rf['Quality_Score'] = 0.05 + (0.15 * (1 - severity_multiplier))
                modified_rf['RF_Constellation_Error'] = 0.2 + (0.3 * severity_multiplier)
                modified_rf['RF_Signal_Strength_dBm'] = modified_rf['RF_Signal_Strength_dBm'] + np.random.uniform(-3, 3)
                
            elif attack_subtype == 'telemetry':
                modified_rf['RF_Constellation_Error'] = 0.3 + (0.5 * severity_multiplier)
                modified_rf['Quality_Score'] = 0.1 + (0.3 * (1 - severity_multiplier))
                modified_rf['RF_Bit_Error_Rate'] = min(0.1, modified_rf['RF_Bit_Error_Rate'] * (1 + 2 * severity_multiplier))
                
            elif attack_subtype == 'command':
                modified_rf['RF_CRC_Errors'] = 1 if severity_multiplier > 0.3 else 0
                modified_rf['RF_Packet_Error_Rate'] = 0.3 + (0.4 * severity_multiplier)
                modified_rf['RF_Sync_Word_Detections'] = 0 if severity_multiplier > 0.5 else 1
                
            elif attack_subtype == 'beacon':
                modified_rf['RF_Signal_Strength_dBm'] = modified_rf['RF_Signal_Strength_dBm'] + np.random.uniform(-5, 5)
                modified_rf['Quality_Score'] = 0.4 + (0.2 * (1 - severity_multiplier))
        
        elif attack_type == 'replay':
            modified_rf['RF_Constellation_Error'] = 0.2 + (0.3 * severity_multiplier)
            modified_rf['RF_Bit_Error_Rate'] = min(0.05, modified_rf['RF_Bit_Error_Rate'] * (1 + severity_multiplier))
            modified_rf['RF_CRC_Errors'] = 1 if np.random.random() < (0.3 * severity_multiplier) else 0
            
        elif attack_type == 'dos':
            if attack_subtype == 'flooding':
                modified_rf['RF_Packet_Error_Rate'] = 0.6 + (0.35 * severity_multiplier)
                modified_rf['RF_Throughput_bps'] = modified_rf['RF_Throughput_bps'] * (1 - 0.8 * severity_multiplier)
                modified_rf['RF_CRC_Errors'] = 1
                
            elif attack_subtype == 'resource_exhaustion':
                modified_rf['RF_Sync_Word_Detections'] = 0 if severity_multiplier > 0.7 else 1
                modified_rf['RF_Packet_Error_Rate'] = 0.4 + (0.5 * severity_multiplier)
                
        elif attack_type == 'mitm':
            modified_rf['RF_Constellation_Error'] = 0.15 + (0.25 * severity_multiplier)
            modified_rf['RF_Bit_Error_Rate'] = min(0.02, modified_rf['RF_Bit_Error_Rate'] * (1 + 0.5 * severity_multiplier))
            delay_factor = 1 + (0.3 * severity_multiplier)
            modified_rf['RF_Frequency_Offset_Hz'] = modified_rf['RF_Frequency_Offset_Hz'] * delay_factor
            
        elif attack_type == 'eavesdrop':
            # Eavesdropping is passive, minimal RF impact
            modified_rf['Quality_Score'] = modified_rf['Quality_Score'] * (0.95 - 0.1 * severity_multiplier)
            if np.random.random() < (0.1 * severity_multiplier):
                modified_rf['RF_Constellation_Error'] = modified_rf['RF_Constellation_Error'] * 1.1
        
        # Apply environmental factors that could mask or enhance attack signatures
        if np.random.random() < 0.1:
            environmental_factor = np.random.uniform(0.8, 1.2)
            modified_rf['RF_SNR_dB'] = modified_rf['RF_SNR_dB'] * environmental_factor
            modified_rf['RF_Signal_Strength_dBm'] = modified_rf['RF_Signal_Strength_dBm'] * environmental_factor
        
        return modified_rf
    
    def _write_to_enhanced_outputs(self, balanced_records: Dict[str, Dict]):
        """Write enhanced balanced records to all output files"""
        
        for name, record in balanced_records.items():
            config = self.output_configs[name]
            
            try:
                # Write to CSV with enhanced columns
                columns = [
                    # Original columns
                    "Timestamp", "Satellite_ID", "Shunt_Voltage_V", "Current_A", "Power_W",
                    "Mag_X_uT", "Mag_Y_uT", "Mag_Z_uT", "Proximity", "Ambient_Light_Lux",
                    "Magnetic_Magnitude_uT", "Power_Density", "Light_Proximity_Ratio", 
                    "Power_Magnetic_Ratio", "Latitude", "Longitude", "Altitude_m", 
                    "GPS_Satellites", "GPS_Fix_Quality", "Velocity_North_ms", 
                    "Velocity_East_ms", "Velocity_Up_ms", "Speed_ms", "Position_Anomaly",
                    "Distance_From_Origin", "Velocity_Bearing_deg", "Vertical_Category", 
                    "Horizontal_Speed_ms", "Reception_Time", "Data_Quality_Score",
                    # RF columns
                    'RF_Signal_Strength_dBm', 'RF_SNR_dB', 'RF_Bit_Error_Rate',
                    'RF_Packet_Error_Rate', 'RF_Throughput_bps', 'RF_Frequency_Offset_Hz',
                    'RF_Doppler_Shift_Hz', 'RF_CRC_Errors', 'RF_Sync_Word_Detections',
                    'RF_Constellation_Error', 'Quality_Score',
                    # Labeling Attack columns
                    'Attack_Flag', 'Attack_Type', 'Attack_Subtype', 'Attack_Severity',
                    'Attack_Duration', 'Attack_Source_Type', 'Detection_Confidence', 'Attack_Timeline_ID'
                ]
                
                values = [str(record.get(col, '')) for col in columns]
                csv_line = ','.join(values) + '\n'
                
                with open(config['csv'], 'a') as f:
                    f.write(csv_line)
                    f.flush()
                
                # Write to JSON with metadata
                json_record = {k: v for k, v in record.items()}
                json_record['processing_timestamp'] = datetime.now().isoformat()
                json_record['record_number'] = self.total_processed
                json_record['balance_ratio'] = config['ratio']
                json_record['enhanced_simulation'] = True
                
                # Convert numpy types
                for key, value in json_record.items():
                    if isinstance(value, (np.integer, np.floating)):
                        json_record[key] = value.item()
                
                with open(config['json'], 'a') as f:
                    if self.total_processed > 1:
                        f.write(',\n')
                    json.dump(json_record, f, indent=2)
                    f.flush()
                
            except Exception as e:
                logger.error(f"Error writing to {name} outputs: {e}")
    
    def _display_enhanced_progress(self, balanced_records: Dict[str, Dict]):
        """Display enhanced real-time progress with attack details"""
        
        sample_record = next(iter(balanced_records.values()))
        
        timestamp = sample_record.get('Timestamp', 'Unknown')
        satellite_id = sample_record.get('Satellite_ID', 'Unknown')
        
        balance_status = []
        attack_details = []
        
        for name, record in balanced_records.items():
            attack_flag = record.get('Attack_Flag', 0)
            ratio = self.output_configs[name]['ratio']
            
            if attack_flag:
                attack_type = record.get('Attack_Type', 'unknown')[:3]
                severity = record.get('Attack_Severity', 0)
                indicator = f"{attack_type.upper()}"
                
                if name == 'balanced_50':
                    attack_details.append(f"Sev:{severity:.2f}")
            else:
                indicator = " NOR"
            
            balance_status.append(f"{int(ratio*100):02d}%:{indicator}")
        
        satellite_indicator = "SAT 1" if "1" in str(satellite_id) else "SAT 2"
        snr = sample_record.get('RF_SNR_dB', 0)
        
        attack_info = f" [{','.join(attack_details)}]" if attack_details else ""
        print(f"{timestamp} | {satellite_indicator} | [{' '.join(balance_status)}] | SNR: {snr:>6.1f}dB{attack_info} | Total: {self.total_processed}")
    
    def start_enhanced_monitoring(self):
        """Start enhanced real-time balanced dataset generation"""
        
        logger.info("Starting Real-Time Balanced Dataset Generator")
        logger.info("=" * 80)
        logger.info(f" Input file: {self.input_file}")
        logger.info(f" Output datasets: {len(self.output_configs)} (Enhanced with realistic attack simulation)")
        
        for name, config in self.output_configs.items():
            logger.info(f"   {name}: {config['ratio']:.1%} attacks → {config['csv']}")
        
        logger.info("=" * 80)
        logger.info(" Enhanced Features:")
        logger.info("    Realistic attack timelines and progression")
        logger.info("    Attack severity modeling")
        logger.info("    Detection confidence scoring")
        logger.info("    Attack source type simulation")
        logger.info("=" * 80)
        
        if self.last_line_count > 0:
            logger.info(f" RESUMING from line {self.last_line_count} (processed {self.total_processed} records)")
        else:
            logger.info(f" Starting from beginning")
        
        self.running = True
        
        try:
            logger.info("Monitoring for new data...")
            print("\nEnhanced Real-Time Balanced Dataset Generation:")
            print("Timestamp               | Sat | [Balance Status] | SNR      | Attack Details | Total")
            print("Format: 50%: NOR or 50%: JAM (for jamming attacks)")
            print("-" * 100)
            
            while self.running:
                new_lines = self._read_new_lines()
                
                if new_lines:
                    for line in new_lines:
                        record = self._parse_csv_line(line)
                        if not record:
                            continue
                        
                        balanced_records = self._generate_enhanced_balanced_records(record)
                        self._write_to_enhanced_outputs(balanced_records)
                        self._display_enhanced_progress(balanced_records)
                        
                        self.total_processed += 1
                
                else:
                    time.sleep(1)
        
        except KeyboardInterrupt:
            logger.info("\n  Stopping Dataset generator ")
        except Exception as e:
            logger.error(f" Error in monitoring loop: {e}")
        finally:
            self.stop_enhanced_monitoring()
    
    def stop_enhanced_monitoring(self):
        """Stop enhanced monitoring and cleanup"""
        self.running = False
        
        self._save_position()
        
        try:
            for name, config in self.output_configs.items():
                with open(config['json'], 'a') as f:
                    f.write('\n]')
                logger.info(f" JSON file closed: {config['json']}")
        except Exception as e:
            logger.error(f"Error closing JSON files: {e}")
        
        runtime = datetime.now() - self.start_time
        logger.info("\n" + "=" * 80)
        logger.info(" FINAL DATASET STATISTICS")
        logger.info("=" * 80)
        logger.info(f"Total records processed: {self.total_processed}")
        logger.info(f"Runtime: {runtime}")
        logger.info(f"Processing rate: {self.total_processed / max(runtime.total_seconds(), 1):.2f} records/sec")
        logger.info("dataset generation stopped")


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\n Stopping generator...")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced Real-Time Balanced Dataset Generator with Realistic Attack Simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument('--input', required=True, 
                       help='Input CSV file to monitor (sat1_and_sat2_combined.csv)')
    parser.add_argument('--output-dir', default='output/',
                       help='Output directory for enhanced balanced datasets (default: output/)')
    parser.add_argument('--ratios', default='0.1,0.3,0.5,0.7,0.9',
                       help='Comma-separated attack ratios (default: 0.1,0.3,0.5,0.7,0.9)')
    parser.add_argument('--balance-window', type=int, default=1000,
                       help='Number of records to consider for balance (default: 1000)')
    parser.add_argument('--reset', action='store_true',
                       help='Reset position and start from beginning')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose debug logging')
    
    # RF Configuration
    parser.add_argument('--sample-rate', type=float, default=50e3, 
                       help='Sample rate in Hz (default: 50000)')
    parser.add_argument('--tx-freq', type=float, default=917e6,
                       help='TX frequency in Hz (default: 917000000)')
    parser.add_argument('--rx-freq', type=float, default=915e6,
                       help='RX frequency in Hz (default: 915000000)')
    parser.add_argument('--packet-length', type=int, default=241,
                       help='Packet length in bytes (default: 241)')
    parser.add_argument('--tx-gain', type=float, default=55.0,
                       help='TX gain in dB (default: 55.0)')
    parser.add_argument('--rx-gain', type=float, default=45.0,
                       help='RX gain in dB (default: 45.0)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Reset position if requested
    if args.reset:
        input_basename = os.path.basename(args.input).replace('.csv', '')
        position_file = os.path.join("logs", f"position_{input_basename}.json")
        state_file = os.path.join("logs", f"state_{input_basename}.json")
        
        for file_path in [position_file, state_file]:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Removed {file_path}")
        
        logger.info("Reset complete - will start from beginning")
    
    # Validate input file
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        logger.info("Make sure your satellite data collection is writing to this file")
        sys.exit(1)
    
    # Parse ratios
    try:
        ratios = [float(r.strip()) for r in args.ratios.split(',')]
        ratios = [r for r in ratios if 0.0 <= r <= 1.0]
        if not ratios:
            raise ValueError("No valid ratios provided")
    except ValueError as e:
        logger.error(f"Invalid ratios: {e}")
        sys.exit(1)
    
    # Create output configurations
    output_configs = {}
    for ratio in ratios:
        name = f"balanced_{int(ratio*100):02d}"
        output_configs[name] = {
            'ratio': ratio,
            'csv': os.path.join(args.output_dir, f"{name}_UNSW_IoTSAT.csv"),
            'json': os.path.join(args.output_dir, f"{name}_UNSW_IoTSAT.json")
        }
    
    # Enhanced configuration
    config = {
        'input_file': args.input,
        'output_configs': output_configs,
        'balance_window': args.balance_window,
        'rf_config': {
            'sample_rate': args.sample_rate,
            'tx_freq': args.tx_freq,
            'rx_freq': args.rx_freq,
            'packet_length': args.packet_length,
            'tx_gain': args.tx_gain,
            'rx_gain': args.rx_gain
        }
    }
    
    # Create and start generator
    generator = EnhancedRealTimeBalancedGenerator(config)
    
    try:
        generator.start_enhanced_monitoring()
    except Exception as e:
        logger.error(f" Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()