import time
from datetime import datetime
import board
import busio
import serial
import adafruit_ina228
import adafruit_lis3mdl
import adafruit_vcnl4040
import math

# ──────────────────────────────
# Coordinate Features Configuration
# ──────────────────────────────
# Set to True if you have a GPS module connected
USE_GPS_MODULE = False  
GPS_UART_PORT = None  # Set to your GPS UART port if available

# Sliding window for velocity calculation
position_history = []
WINDOW_SIZE = 3  # Number of points for velocity calculation

# ──────────────────────────────
# Initialize I2C and Serial
# ──────────────────────────────
i2c = busio.I2C(board.SCL, board.SDA)
ser = serial.Serial('/dev/ttyAMA0', 115200, timeout=1)

# ──────────────────────────────
# Initialize GPS (if available)
# ──────────────────────────────
gps = None
if USE_GPS_MODULE and GPS_UART_PORT:
    try:
        # Uncomment and modify based on your GPS module:
        # import adafruit_gps
        # gps = adafruit_gps.GPS(GPS_UART_PORT, debug=False)
        # gps.send_command(b'PMTK314,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0')
        print("GPS module initialized")
    except Exception as e:
        print(f"GPS initialization failed: {e}")
        gps = None

# ──────────────────────────────
# Initialize Sensors
# ──────────────────────────────
ina = adafruit_ina228.INA228(i2c)
mag = adafruit_lis3mdl.LIS3MDL(i2c)
prox = adafruit_vcnl4040.VCNL4040(i2c)

# ──────────────────────────────
# Coordinate Functions
# ──────────────────────────────
def get_coordinates():
    """Get GPS coordinates (real or simulated)"""
    
    if USE_GPS_MODULE and gps:
        try:
            # Read from real GPS module
            gps.update()
            if gps.has_fix:
                return {
                    'lat': float(gps.latitude),
                    'lon': float(gps.longitude), 
                    'alt': float(gps.altitude_m) if gps.altitude_m else 0.0,
                    'satellites': gps.satellites,
                    'fix_quality': gps.fix_quality
                }
        except Exception as e:
            print(f"GPS read error: {e}")
    
    # Simulated satellite coordinates (for testing without GPS)
    current_time = time.time()
    
    # Simulate orbital parameters for a satellite
    orbital_period = 5400  # 90 minutes in seconds
    angle = (current_time % orbital_period) / orbital_period * 2 * math.pi
    
    # Base coordinates (adjust for your region)
    base_lat = 40.0    # Latitude center
    base_lon = -74.0   # Longitude center  
    base_alt = 400000  # 400 km altitude
    
    # Simulate orbital movement
    lat = base_lat + 15.0 * math.sin(angle)
    lon = base_lon + 25.0 * math.cos(angle) 
    alt = base_alt + 2000 * math.sin(angle * 2)
    
    return {
        'lat': lat,
        'lon': lon,
        'alt': alt,
        'satellites': 8,  # Simulated
        'fix_quality': 1  # Simulated good fix
    }

def calculate_velocity():
    """Calculate velocity from position history"""
    if len(position_history) < 2:
        return 0.0, 0.0, 0.0, 0.0
    
    # Get current and previous position
    current = position_history[-1]
    previous = position_history[0]
    
    # Time difference
    dt = current['time'] - previous['time']
    if dt <= 0:
        return 0.0, 0.0, 0.0, 0.0
    
    # Convert coordinate differences to meters
    # Approximate conversion (for more accuracy use proper geodetic formulas)
    lat_to_meters = 111320  # meters per degree latitude
    lon_to_meters = 111320 * math.cos(math.radians(current['lat']))
    
    # Position changes in meters
    delta_lat_m = (current['lat'] - previous['lat']) * lat_to_meters
    delta_lon_m = (current['lon'] - previous['lon']) * lon_to_meters
    delta_alt_m = current['alt'] - previous['alt']
    
    # Velocity components (m/s)
    vel_north = delta_lat_m / dt  # North-South velocity
    vel_east = delta_lon_m / dt   # East-West velocity
    vel_up = delta_alt_m / dt     # Vertical velocity
    
    # Total speed
    speed = math.sqrt(vel_north*2 + vel_east2 + vel_up*2)
    
    return vel_north, vel_east, vel_up, speed

def detect_position_anomaly(speed):
    """Detect position jumps that might indicate coordinate spoofing"""
    # Typical satellite speeds: 7-8 km/s
    MAX_EXPECTED_SPEED = 10000  # 10 km/s threshold
    
    if speed > MAX_EXPECTED_SPEED:
        return 1  # Anomaly detected
    return 0

# ──────────────────────────────
# Send header with coordinate features added
# ──────────────────────────────
header = (
    "Timestamp,Satellite ID,Shunt Voltage (V),Current (A),Power (W),"
    "Mag X (uT),Mag Y (uT),Mag Z (uT),Proximity,Ambient Light (Lux),"
    "Magnetic Magnitude (uT),Power Density,Light-Proximity Ratio,Power-Magnetic Ratio,"
    "Latitude,Longitude,Altitude (m),GPS Satellites,GPS Fix Quality,"
    "Velocity North (m/s),Velocity East (m/s),Velocity Up (m/s),Speed (m/s),Position Anomaly\n"
)

try:
    ser.write(header.encode('utf-8'))
    print("Header with coordinate features sent to serial output.")
except Exception as e:
    print("Failed to send header:", e)

print("Sending sensor data with coordinate and velocity features.\n")

try:
    while True:
        current_time = time.time()
        
        # ─────────────
        # Read INA228
        # ─────────────
        try:
            shunt_voltage = ina.shunt_voltage
            current = ina.current
            power = ina.power
        except Exception as e:
            print("INA228 read failed:", e)
            shunt_voltage = current = power = 0.0
        
        # ─────────────
        # Read LIS3MDL
        # ─────────────
        try:
            mag_x, mag_y, mag_z = mag.magnetic
        except Exception as e:
            print("Magnetometer read failed:", e)
            mag_x = mag_y = mag_z = 0.0
        
        # ─────────────
        # Read VCNL4040
        # ─────────────
        try:
            proximity = prox.proximity
            lux = prox.lux
        except Exception as e:
            print("VCNL4040 read failed:", e)
            proximity = 0
            lux = 0.0
        
        # ─────────────
        # Get Coordinates
        # ─────────────
        coords = get_coordinates()
        
        # Add to position history for velocity calculation
        position_record = {
            'time': current_time,
            'lat': coords['lat'],
            'lon': coords['lon'],
            'alt': coords['alt']
        }
        position_history.append(position_record)
        
        # Keep only recent positions (sliding window)
        if len(position_history) > WINDOW_SIZE:
            position_history.pop(0)
        
        # ─────────────
        # Calculate Velocity Features
        # ─────────────
        vel_north, vel_east, vel_up, speed = calculate_velocity()
        
        # ─────────────
        # Detect Position Anomalies (for coordinate spoofing detection)
        # ─────────────
        position_anomaly = detect_position_anomaly(speed)
        
        # ─────────────
        # Linear Combination Features (FIXED SYNTAX ERROR)
        # ─────────────
        
        # Magnetic field magnitude (3D vector magnitude) - CORRECTED
        mag_magnitude = math.sqrt(mag_x*2 + mag_y2 + mag_z*2)
        
        # Power density (power per unit current)
        power_density = power / abs(current) if abs(current) > 0.001 else 0.0
        
        # Light to proximity ratio (environmental context)
        lux_proximity_ratio = lux / proximity if proximity > 0 else 0.0
        
        # Power to magnetic field ratio (system activity vs magnetic environment)
        power_mag_ratio = abs(power) / mag_magnitude if mag_magnitude > 0 else 0.0
        
        # ─────────────
        # Satellite label
        # ─────────────
        satellite_label = "Satellite1 or Satellite2"
        
        # ─────────────
        # Format and send data with coordinate features
        # ─────────────
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = (
            f"{timestamp},{satellite1_label},{shunt_voltage:.6f},{current:.3f},{power:.3f},"
            f"{mag_x:.2f},{mag_y:.2f},{mag_z:.2f},{proximity},{lux:.2f},"
            f"{mag_magnitude:.2f},{power_density:.3f},{lux_proximity_ratio:.3f},{power_mag_ratio:.6f},"
            f"{coords['lat']:.8f},{coords['lon']:.8f},{coords['alt']:.2f},"
            f"{coords['satellites']},{coords['fix_quality']},"
            f"{vel_north:.4f},{vel_east:.4f},{vel_up:.4f},{speed:.4f},{position_anomaly}\n"
        )
        
        try:
            ser.write(line.encode('utf-8'))
        except Exception as e:
            print("Serial write failed:", e)
        
        print("Sent:", line.strip())
        
        # Alert for position anomalies
        if position_anomaly:
            print(f"⚠  POSITION ANOMALY DETECTED! Speed: {speed:.2f} m/s")
        
        time.sleep(2)
        
except KeyboardInterrupt:
    print("\nLogging stopped.")
    ser.close()
except Exception as e:
    print(f"Error: {e}")
    ser.close()