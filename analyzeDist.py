import csv

def calculate_average_position(satellites):
    if not satellites:
        return None, None, None
    avg_lat = sum(s[1] for s in satellites) / len(satellites)
    avg_lon = sum(s[2] for s in satellites) / len(satellites)
    avg_alt = sum(s[3] for s in satellites) / len(satellites)
    return avg_lat, avg_lon, avg_alt

def detect_disruption_from_txt_file(gnss_data_file, num_satellites=3):
    try:
        strongest_satellites = []
        weakest_satellites = []

        with open(gnss_data_file, 'r') as file:
            reader = csv.reader(file)
            for line_num, data_fields in enumerate(reader, start=1):
                if line_num == 1 or len(data_fields) < 18:  # Adjust based on the expected number of fields
                    continue
                
                try:
                    # Example: Extract latitude, longitude, altitude
                    latitude = float(data_fields[1])
                    longitude = float(data_fields[9])
                    altitude = float(data_fields[17])
                    
                    # Example: Extract satellite ID and signal strength
                    satellite_id = data_fields[5]
                    signal_strength = float(data_fields[13])
                    
                    # Add satellite to the strongest list if it qualifies
                    if len(strongest_satellites) < num_satellites:
                        strongest_satellites.append((satellite_id, latitude, longitude, altitude, signal_strength))
                        strongest_satellites.sort(key=lambda x: x[4], reverse=True)
                    elif signal_strength > strongest_satellites[-1][4]:
                        strongest_satellites[-1] = (satellite_id, latitude, longitude, altitude, signal_strength)
                        strongest_satellites.sort(key=lambda x: x[4], reverse=True)
                    
                    # Add satellite to the weakest list if it qualifies
                    if len(weakest_satellites) < num_satellites:
                        weakest_satellites.append((satellite_id, latitude, longitude, altitude, signal_strength))
                        weakest_satellites.sort(key=lambda x: x[4])
                    elif signal_strength < weakest_satellites[-1][4]:
                        weakest_satellites[-1] = (satellite_id, latitude, longitude, altitude, signal_strength)
                        weakest_satellites.sort(key=lambda x: x[4])
                
                except ValueError as ve:
                    print(f"Skipping line {line_num}: {data_fields} - Value error: {ve}")
                    continue
        
        # Calculate average positions for strongest and weakest satellites
        strong_lat, strong_lon, strong_alt = calculate_average_position(strongest_satellites)
        weak_lat, weak_lon, weak_alt = calculate_average_position(weakest_satellites)
        
        # Calculate differences in location
        lat_diff = abs(strong_lat - weak_lat)
        lon_diff = abs(strong_lon - weak_lon)
        alt_diff = abs(strong_alt - weak_alt)
        
        # Define thresholds for disruption detection based on expected differences
        # Example thresholds; adjust these based on your data analysis
        lat_threshold = 500  # Increase or decrease as needed
        lon_threshold = 500  # Increase or decrease as needed
        alt_threshold = 500  # Increase or decrease as needed
        
        print(f"Latitude Difference: {lat_diff}, Threshold: {lat_threshold}")
        print(f"Longitude Difference: {lon_diff}, Threshold: {lon_threshold}")
        print(f"Altitude Difference: {alt_diff}, Threshold: {alt_threshold}")

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
gnss_data_file = "C:\\Users\\בר\\OneDrive\\שולחן העבודה\\מדמח\\שנה ג\\רובוטים אוטונומים\\Finish_Project\\GNSS_Navigation_System\\data\\sample\\beirut.txt"
detect_disruption_from_txt_file(gnss_data_file, num_satellites=2)
