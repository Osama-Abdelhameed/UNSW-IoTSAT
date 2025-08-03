import csv
import threading
import time
import json
import os
from datetime import datetime
import queue
import re

# ──────────────────────────────
# Configuration - File Mapping
# ──────────────────────────────

# SATELLITE 1 
SAT1_CSV_FILE = "sat1.csv"              # From Satellite 1
SAT1_TXT_FILE = "sat1.txt"              # From Satellite 1 (same data, different format)

# SATELLITE 2 
SAT2_CSV_FILE = "sat2.csv"          # From Satellite 2
SAT2_TXT_FILE = "sat2.txt"          # From Satellite 2 (same data, different format)

# Output files
COMBINED_CSV = "sat1_and_sat2_combined.csv"
COMBINED_TXT = "sat1_and_sat2_combined.txt"
COMBINED_LOG = "combined_logger.log"
COMBINED_STATS = "combined_stats.json"
POSITION_FILE = "combined_positions.json"  # NEW: Position persistence

# File monitoring interval
FILE_CHECK_INTERVAL = 1  # seconds

# Data queue for thread-safe communication with timestamp sorting
data_queue = queue.PriorityQueue()

class CombinedDataLogger:
    def __init__(self):
        self.csv_file = None
        self.txt_file = None
        self.csv_writer = None
        self.header_written = False
        
        # File monitoring positions for both satellites - NOW PERSISTENT
        self.sat1_csv_position = 0
        self.sat1_txt_position = 0
        self.sat2_csv_position = 0
        self.sat2_txt_position = 0
        
        # Load saved positions
        self._load_positions()
        
        # Track which files we've already processed to avoid duplicates
        self.sat1_processed_csv = False
        self.sat2_processed_csv = False
        
        # Statistics tracking
        self.start_time = datetime.now()
        self.sat1_records = 0
        self.sat2_records = 0
        self.total_records = 0
        self.coordinate_records = 0
        self.position_anomalies = 0
        
        # Thread lock for statistics
        self.stats_lock = threading.Lock()
        
        self.log_message("Combined Data Logger initialized")
        self.log_message("SATELLITE 1 : sat1.csv and sat1.txt")
        self.log_message("SATELLITE 2 : sat2.csv and sat2.txt")
        
        # Show resume status
        if any([self.sat1_csv_position, self.sat1_txt_position, 
                self.sat2_csv_position, self.sat2_txt_position]):
            self.log_message(f"   RESUMING from saved positions:")
            self.log_message(f"   SAT1 CSV: {self.sat1_csv_position}, TXT: {self.sat1_txt_position}")
            self.log_message(f"   SAT2 CSV: {self.sat2_csv_position}, TXT: {self.sat2_txt_position}")
        else:
            self.log_message(" Starting  - no previous positions found")
    
    def _load_positions(self):
        """Load saved file positions from previous run"""
        try:
            if os.path.exists(POSITION_FILE):
                with open(POSITION_FILE, 'r') as f:
                    positions = json.load(f)
                
                self.sat1_csv_position = positions.get('sat1_csv_position', 0)
                self.sat1_txt_position = positions.get('sat1_txt_position', 0)
                self.sat2_csv_position = positions.get('sat2_csv_position', 0)
                self.sat2_txt_position = positions.get('sat2_txt_position', 0)
                
                self.log_message("  Loaded saved positions from previous run")
            else:
                self.log_message("  No position file found - starting fresh")
        except Exception as e:
            self.log_message(f"  Could not load positions: {e}")
            # Reset to safe defaults
            self.sat1_csv_position = 0
            self.sat1_txt_position = 0
            self.sat2_csv_position = 0
            self.sat2_txt_position = 0
    
    def _save_positions(self):
        """Save current file positions for resumption"""
        try:
            positions = {
                'sat1_csv_position': self.sat1_csv_position,
                'sat1_txt_position': self.sat1_txt_position,
                'sat2_csv_position': self.sat2_csv_position,
                'sat2_txt_position': self.sat2_txt_position,
                'last_save_time': datetime.now().isoformat(),
                'total_records_processed': self.total_records
            }
            
            with open(POSITION_FILE, 'w') as f:
                json.dump(positions, f, indent=2)
                
        except Exception as e:
            self.log_message(f"  Could not save positions: {e}")
    
    def log_message(self, message):
        """Log messages with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        
        try:
            with open(COMBINED_LOG, "a") as logfile:
                logfile.write(log_entry + "\n")
        except Exception as e:
            print(f"Failed to write to log file: {e}")
    
    def parse_timestamp(self, timestamp_str):
        """Parse timestamp string and return datetime object"""
        try:
            # Clean the timestamp string
            timestamp_str = str(timestamp_str).strip().strip('"').strip("'")
            
            # Try different timestamp formats
            formats = [
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d",
                "%H:%M:%S"
            ]
            
            for fmt in formats:
                try:
                    if fmt == "%H:%M:%S":
                        # Add today's date for time-only format
                        today = datetime.now().strftime("%Y-%m-%d")
                        timestamp_str = f"{today} {timestamp_str}"
                        fmt = "%Y-%m-%d %H:%M:%S"
                    elif fmt == "%Y-%m-%d":
                        # Add current time for date-only format
                        timestamp_str = f"{timestamp_str} 00:00:00"
                        fmt = "%Y-%m-%d %H:%M:%S"
                    
                    return datetime.strptime(timestamp_str, fmt)
                except ValueError:
                    continue
            
            # If all formats fail, return current time
            return datetime.now()
            
        except Exception:
            return datetime.now()
    
    def clean_csv_line(self, line):
        """Clean and properly parse CSV line with improved handling"""
        try:
            # Remove any trailing commas and whitespace
            line = line.strip().rstrip(',')
            
            # Skip empty lines or header lines
            if not line or line.lower().startswith('timestamp') or line.lower().startswith('time'):
                return []
            
            # Handle the malformed timestamp issue: "2025-06-23,"""20:36:36"
            # This pattern occurs when the timestamp is split incorrectly
            timestamp_pattern = r'^(\d{4}-\d{2}-\d{2}),"""(\d{2}:\d{2}:\d{2})"'
            match = re.match(timestamp_pattern, line)
            
            if match:
                # Reconstruct proper timestamp and get the rest of the line
                date_part = match.group(1)
                time_part = match.group(2)
                proper_timestamp = f"{date_part} {time_part}"
                
                # Find where the timestamp ends and extract the rest
                rest_start = match.end()
                rest_of_line = line[rest_start:].lstrip(',').strip()
                
                # Reconstruct the line with proper timestamp
                line = f"{proper_timestamp},{rest_of_line}"
            
            # Handle CSV parsing properly
            import csv
            from io import StringIO
            
            try:
                csv_reader = csv.reader(StringIO(line))
                values = next(csv_reader)
            except:
                # Fallback to simple split if CSV parsing fails
                values = [val.strip() for val in line.split(",")]
            
            # Clean each value and filter out unwanted entries
            cleaned_values = []
            for value in values:
                cleaned_value = str(value).strip().strip('"').strip("'")
                
                # Skip empty values, "stable" entries, and malformed entries
                if (cleaned_value and 
                    cleaned_value.lower() != "stable" and 
                    not cleaned_value.startswith('"""') and
                    cleaned_value != '0"""' and
                    cleaned_value != '"""0'):
                    cleaned_values.append(cleaned_value)
            
            return cleaned_values
            
        except Exception as e:
            self.log_message(f"Error cleaning CSV line: {e}")
            return []
    
    def normalize_data_row(self, values, source_id):
        """Normalize data to exactly 30 columns with proper formatting"""
        try:
            if len(values) < 2:
                return None
            
            # Remove any remaining "stable" entries
            values = [val for val in values if str(val).lower() != "stable"]
            
            if len(values) < 2:
                return None
            
            # Start with cleaned values
            normalized = []
            
            # Expected column count
            expected_columns = 30
            
            # Process each column according to expected format
            for i in range(expected_columns):
                if i < len(values):
                    value = str(values[i]).strip().strip('"').strip("'")
                    
                    # Special handling for timestamp (column 0)
                    if i == 0:
                        # Ensure proper timestamp format
                        parsed_time = self.parse_timestamp(value)
                        normalized.append(parsed_time.strftime("%Y-%m-%d %H:%M:%S"))
                    
                    # Special handling for satellite ID (column 1)
                    elif i == 1:
                        # Ensure consistent satellite naming
                        if "satellite1" in value.lower() or "sat1" in value.lower():
                            normalized.append("Satellite1")
                        elif "satellite2" in value.lower() or "sat2" in value.lower():
                            normalized.append("Satellite2")
                        else:
                            normalized.append(value)
                    
                    # All other columns - handle numeric values
                    else:
                        # Check if it's a numeric value
                        try:
                            # Try to convert to float to validate
                            float_val = float(value)
                            normalized.append(str(float_val))
                        except ValueError:
                            # If not numeric, keep as string but clean it
                            if value and value not in ['0"""', '"""0', 'stable']:
                                normalized.append(value)
                            else:
                                # Use default value for this column position
                                if i == 28:  # Reception time
                                    normalized.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                                elif i == 29:  # Data quality score
                                    normalized.append("1.0")
                                else:
                                    normalized.append("0.0")
                else:
                    # Default values for missing columns
                    if i == 28:  # Reception time
                        normalized.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    elif i == 29:  # Data quality score
                        normalized.append("1.0")
                    else:
                        normalized.append("0.0")
            
            return normalized[:expected_columns]
            
        except Exception as e:
            self.log_message(f"Error normalizing data from {source_id}: {e}")
            return None
    
    def detect_coordinate_data(self, values):
        """Check if data contains valid coordinate information"""
        try:
            if len(values) >= 17:
                lat = float(values[14])
                lon = float(values[15])
                alt = float(values[16])
                
                if abs(lat) <= 90 and abs(lon) <= 180 and alt < 500000:
                    return True, lat, lon, alt
            return False, 0, 0, 0
        except (ValueError, IndexError):
            return False, 0, 0, 0
    
    def open_files(self):
        """Open output files"""
        try:
            self.txt_file = open(COMBINED_TXT, "ab")
            self.csv_file = open(COMBINED_CSV, "a", newline='')
            self.csv_writer = csv.writer(self.csv_file)
            
            # Write header if file is empty
            if self.csv_file.tell() == 0:
                header = [
                    "Timestamp", "Satellite_ID", "Shunt_Voltage_V", "Current_A", "Power_W",
                    "Mag_X_uT", "Mag_Y_uT", "Mag_Z_uT", "Proximity", "Ambient_Light_Lux",
                    "Magnetic_Magnitude_uT", "Power_Density", "Light_Proximity_Ratio", "Power_Magnetic_Ratio",
                    "Latitude", "Longitude", "Altitude_m", "GPS_Satellites", "GPS_Fix_Quality",
                    "Velocity_North_ms", "Velocity_East_ms", "Velocity_Up_ms", "Speed_ms", "Position_Anomaly",
                    "Distance_From_Origin", "Velocity_Bearing_deg", "Vertical_Category", "Horizontal_Speed_ms",
                    "Reception_Time", "Data_Quality_Score"
                ]
                self.csv_writer.writerow(header)
                self.header_written = True
                self.log_message(f"CSV header written ({len(header)} columns)")
            
        except Exception as e:
            self.log_message(f"Error opening output files: {e}")
            raise
    
    def close_files(self):
        """Close output files"""
        try:
            if self.txt_file:
                self.txt_file.close()
            if self.csv_file:
                self.csv_file.close()
            self.log_message("Output files closed")
        except Exception as e:
            self.log_message(f"Error closing files: {e}")
    
    def read_new_lines_from_file(self, file_path, last_position):
        """Read new lines from a file since last check"""
        try:
            if not os.path.exists(file_path):
                return [], last_position
            
            with open(file_path, 'r') as file:
                file.seek(last_position)
                new_lines = file.readlines()
                new_position = file.tell()
                return new_lines, new_position
                
        except Exception as e:
            self.log_message(f"Error reading {file_path}: {e}")
            return [], last_position
    
    def process_data_line(self, line, source_id):
        """Process a data line from either satellite"""
        try:
            line = line.strip()
            if not line or line.startswith('Timestamp') or line.startswith('timestamp'):
                return None
            
            # Clean and parse the CSV line properly
            values = self.clean_csv_line(line)
            
            # Skip if no valid values after cleaning
            if not values or len(values) < 2:
                return None
            
            # Normalize to 30 columns
            normalized_data = self.normalize_data_row(values, source_id)
            if not normalized_data:
                return None
            
            # Get timestamp for ordering
            timestamp_str = normalized_data[0]
            timestamp_obj = self.parse_timestamp(timestamp_str)
            
            # Check for coordinate data
            has_coords, lat, lon, alt = self.detect_coordinate_data(normalized_data)
            
            # Update statistics
            with self.stats_lock:
                if source_id == "Satellite1":
                    self.sat1_records += 1
                else:
                    self.sat2_records += 1
                
                self.total_records += 1
                
                if has_coords:
                    self.coordinate_records += 1
                    
                    # Check for position anomaly (column 23)
                    try:
                        if len(normalized_data) > 23 and normalized_data[23] == "1":
                            self.position_anomalies += 1
                    except:
                        pass
            
            return timestamp_obj, normalized_data
                
        except Exception as e:
            self.log_message(f"Error processing {source_id} line: {e}")
            return None
    
    def save_statistics(self):
        """Save current statistics to JSON file"""
        try:
            with self.stats_lock:
                runtime = datetime.now() - self.start_time
                stats = {
                    'start_time': self.start_time.isoformat(),
                    'runtime_seconds': int(runtime.total_seconds()),
                    'sat1_records': self.sat1_records,
                    'sat2_records': self.sat2_records,
                    'total_records': self.total_records,
                    'coordinate_records': self.coordinate_records,
                    'position_anomalies': self.position_anomalies,
                    'output_csv': COMBINED_CSV,
                    'output_txt': COMBINED_TXT,
                    'last_updated': datetime.now().isoformat()
                }
            
            with open(COMBINED_STATS, "w") as f:
                json.dump(stats, f, indent=2)
                
        except Exception as e:
            self.log_message(f"Error saving statistics: {e}")
    
    def sat1_monitor_thread(self):
        """Monitor Satellite 1 files """
        self.log_message("Satellite 1 monitor thread started")
        
        # FIXED: Only process NEW data if resuming, or all data if starting fresh
        for file_path, position_attr in [(SAT1_CSV_FILE, 'sat1_csv_position'), (SAT1_TXT_FILE, 'sat1_txt_position')]:
            current_position = getattr(self, position_attr)
            
            if os.path.exists(file_path) and current_position == 0:
                # Starting fresh - process all existing data
                self.log_message(f"Processing existing Satellite 1 data from {file_path}...")
                with open(file_path, 'r') as file:
                    for line in file:
                        result = self.process_data_line(line, "Satellite1")
                        if result:
                            timestamp_obj, normalized_data = result
                            priority = timestamp_obj.timestamp()
                            data_queue.put((priority, "Satellite1", normalized_data))
                
                # Update position to end of file
                setattr(self, position_attr, os.path.getsize(file_path))
                self.log_message(f"Finished processing existing Satellite 1 data from {file_path}")
            
            elif os.path.exists(file_path) and current_position > 0:
                # Resuming - set position to where we left off
                setattr(self, position_attr, current_position)
                self.log_message(f"Resuming Satellite 1 monitoring from position {current_position} in {file_path}")
        
        # Monitor for new data
        while True:
            try:
                # Check for new data in both files
                for file_path, position_attr in [(SAT1_CSV_FILE, 'sat1_csv_position'), (SAT1_TXT_FILE, 'sat1_txt_position')]:
                    if os.path.exists(file_path):
                        new_lines, new_position = self.read_new_lines_from_file(
                            file_path, getattr(self, position_attr)
                        )
                        setattr(self, position_attr, new_position)
                        
                        for line in new_lines:
                            result = self.process_data_line(line, "Satellite1")
                            if result:
                                timestamp_obj, normalized_data = result
                                priority = timestamp_obj.timestamp()
                                data_queue.put((priority, "Satellite1", normalized_data))
                
                time.sleep(FILE_CHECK_INTERVAL)
                
            except Exception as e:
                self.log_message(f"Satellite 1 monitor error: {e}")
                time.sleep(5)
    
    def sat2_monitor_thread(self):
        """Monitor Satellite 2 files """
        self.log_message("Satellite 2 monitor thread started")
        
        # FIXED: Only process NEW data if resuming, or all data if starting fresh
        for file_path, position_attr in [(SAT2_CSV_FILE, 'sat2_csv_position'), (SAT2_TXT_FILE, 'sat2_txt_position')]:
            current_position = getattr(self, position_attr)
            
            if os.path.exists(file_path) and current_position == 0:
                # Starting fresh - process all existing data
                self.log_message(f"Processing existing Satellite 2 data from {file_path}...")
                with open(file_path, 'r') as file:
                    for line in file:
                        result = self.process_data_line(line, "Satellite2")
                        if result:
                            timestamp_obj, normalized_data = result
                            priority = timestamp_obj.timestamp()
                            data_queue.put((priority, "Satellite2", normalized_data))
                
                # Update position to end of file
                setattr(self, position_attr, os.path.getsize(file_path))
                self.log_message(f"Finished processing existing Satellite 2 data from {file_path}")
            
            elif os.path.exists(file_path) and current_position > 0:
                # Resuming - set position to where we left off
                setattr(self, position_attr, current_position)
                self.log_message(f"Resuming Satellite 2 monitoring from position {current_position} in {file_path}")
        
        # Monitor for new data
        while True:
            try:
                # Check for new data in both files
                for file_path, position_attr in [(SAT2_CSV_FILE, 'sat2_csv_position'), (SAT2_TXT_FILE, 'sat2_txt_position')]:
                    if os.path.exists(file_path):
                        new_lines, new_position = self.read_new_lines_from_file(
                            file_path, getattr(self, position_attr)
                        )
                        setattr(self, position_attr, new_position)
                        
                        for line in new_lines:
                            result = self.process_data_line(line, "Satellite2")
                            if result:
                                timestamp_obj, normalized_data = result
                                priority = timestamp_obj.timestamp()
                                data_queue.put((priority, "Satellite2", normalized_data))
                
                time.sleep(FILE_CHECK_INTERVAL)
                
            except Exception as e:
                self.log_message(f"Satellite 2 monitor error: {e}")
                time.sleep(5)
    
    def data_writer_thread(self):
        """Write combined data to output files in chronological order"""
        self.log_message("Data writer thread started")
        
        while True:
            try:
                # Get data from priority queue (automatically sorted by timestamp)
                priority, source_type, data = data_queue.get(timeout=1)
                
                # Write to CSV
                if self.csv_writer:
                    self.csv_writer.writerow(data)
                    self.csv_file.flush()
                
                # Write to text file
                if self.txt_file:
                    line_str = ",".join(map(str, data)) + "\n"
                    self.txt_file.write(line_str.encode('utf-8'))
                    self.txt_file.flush()
                
                # Show clean progress - EXACTLY as requested
                timestamp = data[0] if len(data) > 0 else "Unknown"
                satellite_id = data[1] if len(data) > 1 else "Unknown"
                print(f"{timestamp} | {satellite_id}")
                
                # Save statistics and positions periodically
                if self.total_records % 100 == 0:
                    self.save_statistics()
                    self._save_positions()  # NEW: Save positions regularly
                
            except queue.Empty:
                continue
            except Exception as e:
                self.log_message(f"Data writer error: {e}")
    
    def statistics_thread(self):
        """Periodic statistics reporting"""
        while True:
            time.sleep(300)  # Every 5 minutes
            runtime = datetime.now() - self.start_time
            
            with self.stats_lock:
                self.log_message(f"Statistics: {self.total_records} total records "
                               f"(SAT1: {self.sat1_records}, SAT2: {self.sat2_records}), "
                               f"{self.coordinate_records} with coordinates, "
                               f"{self.position_anomalies} anomalies, "
                               f"runtime: {int(runtime.total_seconds())}s")
            
            self.save_statistics()
            self._save_positions()  # NEW: Save positions with statistics
    
    def start_combined_logging(self):
        """Start the combined logging system"""
        self.log_message("=" * 60)
        self.log_message("Starting Time-Ordered Combined Satellite Data Collection")
        self.log_message("=" * 60)
        
        # Check file status
        files_status = []
        for file_path, description in [
            (SAT1_CSV_FILE, "Satellite 1  CSV"),
            (SAT1_TXT_FILE, "Satellite 1  TXT"),
            (SAT2_CSV_FILE, "Satellite 2  CSV"),
            (SAT2_TXT_FILE, "Satellite 2  TXT")
        ]:
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                files_status.append(f"Found {description}: {file_path} ({size} bytes)")
            else:
                files_status.append(f"Missing {description}: {file_path}")
        
        for status in files_status:
            self.log_message(status)
        
        # Open output files
        self.open_files()
        
        try:
            # Start monitoring threads
            threads = []
            
            # Satellite 1 monitor
            sat1_thread = threading.Thread(target=self.sat1_monitor_thread, daemon=True)
            sat1_thread.start()
            threads.append(sat1_thread)
            
            # Satellite 2  monitor
            sat2_thread = threading.Thread(target=self.sat2_monitor_thread, daemon=True)
            sat2_thread.start()
            threads.append(sat2_thread)
            
            # Data writer
            writer_thread = threading.Thread(target=self.data_writer_thread, daemon=True)
            writer_thread.start()
            threads.append(writer_thread)
            
            # Statistics
            stats_thread = threading.Thread(target=self.statistics_thread, daemon=True)
            stats_thread.start()
            threads.append(stats_thread)
            
            self.log_message("=" * 60)
            self.log_message("MONITORING STARTED - Data will be displayed in chronological order")
            self.log_message("=" * 60)
            self.log_message(f"Output files:")
            self.log_message(f"   CSV: {COMBINED_CSV}")
            self.log_message(f"   TXT: {COMBINED_TXT}")
            self.log_message(f"   LOG: {COMBINED_LOG}")
            self.log_message(f"   STATS: {COMBINED_STATS}")
            self.log_message(f"   POSITIONS: {POSITION_FILE}")
            self.log_message("Format: YYYY-MM-DD HH:MM:SS | SatelliteID")
            self.log_message("Press Ctrl+C to stop")
            self.log_message("=" * 60)
            
            # Keep alive
            while True:
                time.sleep(1)
        
        except KeyboardInterrupt:
            self.log_message("=" * 60)
            self.log_message("Stopping combined data logger...")
        except Exception as e:
            self.log_message(f"Main thread error: {e}")
        finally:
            # Save final statistics and positions
            self.save_statistics()
            self._save_positions()  # NEW: Save final positions
            self.close_files()
            
            # Final summary
            runtime = datetime.now() - self.start_time
            self.log_message("Final Summary:")
            self.log_message(f"   Runtime: {runtime}")
            self.log_message(f"   Total records: {self.total_records}")
            self.log_message(f"   Satellite 1  records: {self.sat1_records}")
            self.log_message(f"   Satellite 2  records: {self.sat2_records}")
            self.log_message(f"   Coordinate records: {self.coordinate_records}")
            self.log_message(f"   Position anomalies: {self.position_anomalies}")
            self.log_message(f"   Positions saved to: {POSITION_FILE}")
            self.log_message("Data collection stopped.")
            self.log_message("=" * 60)

def reset_positions():
    """Reset position tracking (start from beginning)"""
    if os.path.exists(POSITION_FILE):
        os.remove(POSITION_FILE)
        print(f" Removed position file: {POSITION_FILE}")
        print(" Next run will start from the beginning")
    else:
        print(" No position file found - already starting fresh")

def show_positions():
    """Show current saved positions"""
    try:
        if os.path.exists(POSITION_FILE):
            with open(POSITION_FILE, 'r') as f:
                positions = json.load(f)
            
            print(" Current saved positions:")
            print(f"   SAT1 CSV: {positions.get('sat1_csv_position', 0)} bytes")
            print(f"   SAT1 TXT: {positions.get('sat1_txt_position', 0)} bytes")
            print(f"   SAT2 CSV: {positions.get('sat2_csv_position', 0)} bytes")
            print(f"   SAT2 TXT: {positions.get('sat2_txt_position', 0)} bytes")
            print(f"   Last saved: {positions.get('last_save_time', 'Unknown')}")
            print(f"   Total records processed: {positions.get('total_records_processed', 0)}")
        else:
            print(" No position file found - will start fresh")
    except Exception as e:
        print(f" Error reading positions: {e}")

def main():
    import sys
    
    print("Time-Ordered Combined Satellite Data Logger")
    print("=" * 60)
    print("SATELLITE 1 : sat1.csv and sat1.txt")
    print("SATELLITE 2 : sat2.csv and sat2.txt")
    print("Output: Chronologically ordered data in sat1_and_sat2_combined.csv")
    print("Display format: YYYY-MM-DD HH:MM:SS | SatelliteID")
    print("=" * 60)
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--reset":
            reset_positions()
            return
        elif sys.argv[1] == "--show-positions":
            show_positions()
            return
        elif sys.argv[1] == "--help":
            print("\nUsage:")
            print("  python combined_data_sat1_sat2.py               # Start monitoring (resumes from last position)")
            print("  python combined_data_sat1_sat2.py --reset       # Reset positions and start from beginning")
            print("  python combined_data_sat1_sat2.py --show-positions  # Show current saved positions")
            print("  python combined_data_sat1_sat2.py --help        # Show this help")
            print("\nFeatures:")
            print("   Position persistence - resumes from where it left off")
            print("   Real-time monitoring - processes new data as it arrives")
            print("   Chronological ordering - combines data from both satellites in time order")
            print("   Duplicate prevention - won't reprocess data on restart")
            print("\nFiles created:")
            print(f"   {COMBINED_CSV} - Combined CSV output")
            print(f"   {COMBINED_TXT} - Combined TXT output")
            print(f"   {COMBINED_LOG} - Activity log")
            print(f"   {COMBINED_STATS} - Statistics JSON")
            print(f"   {POSITION_FILE} - Position tracking (for resume capability)")
            return
    
    logger = CombinedDataLogger()
    logger.start_combined_logging()

if __name__ == "__main__":
    main()
