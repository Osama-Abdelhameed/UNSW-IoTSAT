import serial
import csv
import time
import os
from datetime import datetime

# Configuration
SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 115200
SATELLITE_ID = "Satellite1 / Satellite2"  # Match your Raspberry Pi satellite ID

# File paths
RAW_FILE = "tx_input.txt"
CSV_FILE = "local_input.csv"
LOG_FILE = "receiver_log.txt"

def log_message(message):
    """Log messages with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    
    # Also write to log file
    try:
        with open(LOG_FILE, "a") as logfile:
            logfile.write(log_entry + "\n")
    except Exception as e:
        print(f"Failed to write to log file: {e}")

def calculate_additional_features(values):
    """Calculate additional features from received data"""
    try:
        # Extract coordinate and velocity data (if available)
        if len(values) >= 20:  # Enhanced format with coordinates
            lat = float(values[14])
            lon = float(values[15]) 
            alt = float(values[16])
            vel_north = float(values[19])
            vel_east = float(values[20])
            vel_up = float(values[21])
            speed = float(values[22])
            
            # Calculate additional derived features
            # Distance from origin (rough estimate)
            distance_from_origin = (lat**2 + lon**2)**0.5
            
            # Velocity direction (bearing in degrees)
            import math
            bearing = math.atan2(vel_east, vel_north) * 180 / math.pi
            if bearing < 0:
                bearing += 360
            
            # Vertical speed category
            if abs(vel_up) < 10:
                vertical_category = "stable"
            elif vel_up > 10:
                vertical_category = "ascending" 
            else:
                vertical_category = "descending"
            
            return {
                'distance_from_origin': round(distance_from_origin, 6),
                'velocity_bearing_deg': round(bearing, 2),
                'vertical_category': vertical_category,
                'horizontal_speed': round((vel_north**2 + vel_east**2)**0.5, 4)
            }
    except (ValueError, IndexError) as e:
        log_message(f"Error calculating additional features: {e}")
    
    return {
        'distance_from_origin': 0.0,
        'velocity_bearing_deg': 0.0,
        'vertical_category': 'unknown',
        'horizontal_speed': 0.0
    }

# Open serial port
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    log_message(f"Serial port {SERIAL_PORT} opened successfully at {BAUD_RATE} baud")
except Exception as e:
    log_message(f"Failed to open serial port: {e}")
    exit(1)

# Open files in append mode
try:
    with open(RAW_FILE, "ab") as txtfile, \
         open(CSV_FILE, "a", newline='') as csvfile:
        
        # CSV writer
        writer = csv.writer(csvfile)
        
        # Write enhanced header if file is empty
        if csvfile.tell() == 0:
            # Enhanced header with coordinate features
            writer.writerow([
                "Timestamp", "Satellite ID", "Shunt Voltage (V)", "Current (A)", "Power (W)",
                "Mag X (uT)", "Mag Y (uT)", "Mag Z (uT)", "Proximity", "Ambient Light (Lux)",
                "Magnetic Magnitude (uT)", "Power Density", "Light-Proximity Ratio", "Power-Magnetic Ratio",
                "Latitude", "Longitude", "Altitude (m)", "GPS Satellites", "GPS Fix Quality",
                "Velocity North (m/s)", "Velocity East (m/s)", "Velocity Up (m/s)", "Speed (m/s)", "Position Anomaly",
                "Distance From Origin", "Velocity Bearing (deg)", "Vertical Category", "Horizontal Speed (m/s)",
                "Reception Time", "Data Quality Score"
            ])
            log_message("Enhanced CSV header written to local_input.csv")
        
        log_message(f"Listening for data from {SATELLITE_ID}...")
        
        data_count = 0
        error_count = 0
        
        while True:
            try:
                line = ser.readline()
                if not line:
                    continue
                
                # Write to raw TX file (backup)
                txtfile.write(line)
                txtfile.flush()
                
                # Decode and process
                decoded = line.decode('utf-8').strip()
                
                # Skip empty lines and headers
                if not decoded or decoded.startswith('Timestamp'):
                    continue
                
                print(f"Received: {decoded}")
                values = decoded.split(",")
                
                # Process data based on format
                if len(values) == 14:
                    # Original format (14 fields) - extend with default coordinate values
                    enhanced_values = values + [
                        "0.00000000", "0.00000000", "0.00",  # Lat, Lon, Alt (default)
                        "0", "0",                            # GPS Satellites, Fix Quality
                        "0.0000", "0.0000", "0.0000", "0.0000", "0",  # Velocities, Speed, Anomaly
                        "0.000000", "0.00", "unknown", "0.0000",      # Additional features
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # Reception time
                        "1.0"  # Data quality score
                    ]
                    
                    writer.writerow(enhanced_values)
                    csvfile.flush()
                    
                    data_count += 1
                    log_message(f"Processed original format data with defaults (record #{data_count})")
                    
                elif len(values) >= 20:
                    # Enhanced format with coordinates (20+ fields)
                    
                    # Calculate additional features
                    additional_features = calculate_additional_features(values)
                    
                    # Create complete enhanced record
                    enhanced_values = values + [
                        additional_features['distance_from_origin'],
                        additional_features['velocity_bearing_deg'],
                        additional_features['vertical_category'],
                        additional_features['horizontal_speed'],
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # Reception time
                        "1.0"  # Data quality score
                    ]
                    
                    writer.writerow(enhanced_values)
                    csvfile.flush()
                    
                    data_count += 1
                    
                    # Check for position anomalies
                    if len(values) >= 24 and values[23] == "1":
                        log_message(f" POSITION ANOMALY detected in record #{data_count}!")
                    
                    log_message(f"Processed enhanced format data (record #{data_count})")
                    
                else:
                    # Malformed data
                    error_count += 1
                    log_message(f"Skipped malformed line - expected 14 or 20+ values, got {len(values)} (error #{error_count})")
                    
                    # Log the malformed data for debugging
                    with open("malformed_data.log", "a") as error_log:
                        error_log.write(f"{datetime.now()}: {decoded}\n")
                
                # Print statistics every 100 records
                if data_count % 100 == 0:
                    log_message(f"Statistics: {data_count} records processed, {error_count} errors")
                    
            except UnicodeDecodeError as e:
                error_count += 1
                log_message(f"Unicode decode error: {e}")
                
            except Exception as e:
                error_count += 1
                log_message(f"Processing error: {e}")
                time.sleep(0.1)  # Brief pause on errors
                
except KeyboardInterrupt:
    log_message("Data collection stopped by user")
    
except Exception as e:
    log_message(f"Fatal error: {e}")
    
finally:
    try:
        ser.close()
        log_message("Serial port closed")
    except:
        pass
    
    log_message(f"Final statistics: {data_count} records processed, {error_count} errors")
    print(f"\nData collection completed!")
    print(f"Files created:")
    print(f"  - {RAW_FILE} (raw data backup)")
    print(f"  - {CSV_FILE} (enhanced format with coordinates and velocity)")
    print(f"  - {LOG_FILE} (processing log)")
    if error_count > 0:
        print(f"  - malformed_data.log (debug log for {error_count} errors)")
