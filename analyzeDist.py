import csv

class GNSSDisruptionDetector:
    def __init__(self, gnss_data_file, num_satellites=3):
        self.gnss_data_file = gnss_data_file
        self.num_satellites = num_satellites
        self.strongest_satellites = []
        self.weakest_satellites = []
        self.isDistrubt = False

    @staticmethod
    def calculate_average_position(satellites):
        if not satellites:
            return None, None, None
        avg_lat = sum(s[1] for s in satellites) / len(satellites)
        avg_lon = sum(s[2] for s in satellites) / len(satellites)
        avg_alt = sum(s[3] for s in satellites) / len(satellites)
        return avg_lat, avg_lon, avg_alt

    def process_data(self):
        try:
            with open(self.gnss_data_file, 'r') as file:
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
                        if len(self.strongest_satellites) < self.num_satellites:
                            self.strongest_satellites.append((satellite_id, latitude, longitude, altitude, signal_strength))
                            self.strongest_satellites.sort(key=lambda x: x[4], reverse=True)
                        elif signal_strength > self.strongest_satellites[-1][4]:
                            self.strongest_satellites[-1] = (satellite_id, latitude, longitude, altitude, signal_strength)
                            self.strongest_satellites.sort(key=lambda x: x[4], reverse=True)

                        # Add satellite to the weakest list if it qualifies
                        if len(self.weakest_satellites) < self.num_satellites:
                            self.weakest_satellites.append((satellite_id, latitude, longitude, altitude, signal_strength))
                            self.weakest_satellites.sort(key=lambda x: x[4])
                        elif signal_strength < self.weakest_satellites[-1][4]:
                            self.weakest_satellites[-1] = (satellite_id, latitude, longitude, altitude, signal_strength)
                            self.weakest_satellites.sort(key=lambda x: x[4])

                    except ValueError as ve:
                        print(f"Skipping line {line_num}: {data_fields} - Value error: {ve}")
                        continue

            # Calculate average positions for strongest and weakest satellites
            strong_lat, strong_lon, strong_alt = self.calculate_average_position(self.strongest_satellites)
            weak_lat, weak_lon, weak_alt = self.calculate_average_position(self.weakest_satellites)

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
                # print("Found disruption in GNSS data based on location differences.")
                self.isDistrubt = True
            else:
                # print("No disruption detected in GNSS data based on location differences.")
                self.isDistrubt = False

        except FileNotFoundError:
            print(f"File not found: {self.gnss_data_file}")

        except Exception as e:
            print(f"Error occurred: {e}")

