def calculate_position(latitude, longitude, altitude):
    # Replace with actual position calculation logic based on your GNSS data format
    return latitude, longitude, altitude

def detect_disruption_from_txt_file(gnss_data_file):
    try:
        strongest_satellite = None
        weakest_satellite = None
        max_strength = -float('inf')
        min_strength = float('inf')
        
        with open(gnss_data_file, 'r') as file:
            for line_num, line in enumerate(file, start=1):
                if line.startswith('#') or line.strip() == '':
                    continue
                
                data_fields = line.split(',')  # Adjust delimiter based on your data format
                
                # Check if line has enough fields
                if len(data_fields) < 30:  # Adjust based on the expected number of fields
                    print(f"Skipping line {line_num}: {line.strip()} - Insufficient data fields")
                    continue
                
                try:
                    # Example: Extract latitude, longitude, altitude
                    latitude = float(data_fields[1])
                    longitude = float(data_fields[9])
                    altitude = float(data_fields[17])
                    
                    # Example: Extract satellite ID and signal strength
                    satellite_id = data_fields[5]
                    signal_strength = float(data_fields[13])
                    
                    # Determine strongest and weakest satellites
                    if signal_strength > max_strength:
                        max_strength = signal_strength
                        strongest_satellite = (satellite_id, latitude, longitude, altitude)
                    
                    if signal_strength < min_strength:
                        min_strength = signal_strength
                        weakest_satellite = (satellite_id, latitude, longitude, altitude)
                
                except ValueError as ve:
                    print(f"Skipping line {line_num}: {line.strip()} - Value error: {ve}")
                    continue
        
        # Calculate positions for strongest and weakest satellites
        if strongest_satellite and weakest_satellite:
            strong_lat, strong_lon, strong_alt = calculate_position(*strongest_satellite[1:])
            weak_lat, weak_lon, weak_alt = calculate_position(*weakest_satellite[1:])
            
            # Calculate differences in location
            lat_diff = abs(strong_lat - weak_lat)
            lon_diff = abs(strong_lon - weak_lon)
            alt_diff = abs(strong_alt - weak_alt)
            
            # Define thresholds for disruption detection based on expected differences
            # Example thresholds; adjust these based on your data analysis
            lat_threshold = 2000  # Increase or decrease as needed
            lon_threshold = 2000  # Increase or decrease as needed
            alt_threshold = 2000  # Increase or decrease as needed
            
            # Check if differences exceed thresholds
            if lat_diff > lat_threshold or lon_diff > lon_threshold or alt_diff > alt_threshold:
                print("Found disruption in GNSS data based on location differences.")
            else:
                print("No disruption detected in GNSS data based on location differences.")
        
    except FileNotFoundError:
        print(f"File not found: {gnss_data_file}")
    
    except Exception as e:
        print(f"Error occurred: {e}")

# Example usage:
gnss_data_file = "C:\\Users\\בר\\OneDrive\\שולחן העבודה\\מדמח\\שנה ג\\רובוטים אוטונומים\\Finish_Project\\GNSS_Navigation_System\\data\\sample\\Walking.txt"
detect_disruption_from_txt_file(gnss_data_file)
