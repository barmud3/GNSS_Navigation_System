import sys, os, csv
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import navpy
from gnssutils import ephemeris_manager
import simplekml
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import subprocess

LIGHTSPEED = 2.99792458e8
ephemeris_data_directory = os.path.join('data')
# Define the folder structure
gnss_log_samples_dir = 'gnss_log_samples'
outcomes_dir = 'outcomes'

# Create folders if they don't exist
os.makedirs(gnss_log_samples_dir, exist_ok=True)
os.makedirs(outcomes_dir, exist_ok=True)

def find_latest_gnss_log():
    try:
        # Resetting the ADB command to find the latest file in /sdcard/Download directory
        adb_command = ["C:/Users/User/Documents/adb/adb.exe", "shell", "ls", "-t", "/sdcard/Download/gnss_log*.txt"]

        # Running the ADB command and capturing the output
        result = subprocess.run(adb_command, capture_output=True, text=True, check=True)

        # Sorting files by creation date and selecting the latest file
        files_list = result.stdout.strip().split('\n')
        latest_file = files_list[0]  # The last file in the sorted list is the newest file

        return latest_file.strip()  # Returning the name of the latest file

    except subprocess.CalledProcessError as e:
        print(f"Error finding latest file: {e}")
        return None

def weighted_least_squares(xs, measured_pseudorange, x0, b0, weights):
    dx = 100 * np.ones(3)
    b = b0
    G = np.ones((measured_pseudorange.size, 4))
    iterations = 0

    while np.linalg.norm(dx) > 1e-3:
        r = np.linalg.norm(xs - x0, axis=1)
        phat = r + b0
        deltaP = measured_pseudorange - phat
        W = np.diag(weights)  # Weight matrix
        G[:, 0:3] = -(xs - x0) / r[:, None]

        # Weighted least squares solution
        sol = np.linalg.inv(G.T @ W @ G) @ G.T @ W @ deltaP
        dx = sol[0:3]
        db = sol[3]
        x0 = x0 + dx
        b0 = b0 + db

    norm_dp = np.linalg.norm(deltaP)
    return x0, b0, norm_dp


def positioning_algorithm(csv_file):
    df = pd.read_csv(csv_file)
    data = []
    df_times = df['GPS time'].unique()
    x0 = np.array([0, 0, 0])
    b0 = 0
    for time in df_times:
        df_gps_time = df[df['GPS time'] == time]
        df_gps_time_sorted = df_gps_time.sort_values(by='SatPRN (ID)')
        xs = df_gps_time_sorted[['Sat.X', 'Sat.Y', 'Sat.Z']].values
        measured_pseudorange = df_gps_time_sorted['Pseudo-Range'].values
        weights = df_gps_time_sorted['CN0'].values  # Use CN0 values as weights
        x_estimate, bias_estimate, norm_dp = weighted_least_squares(xs, measured_pseudorange, x0, b0, weights)
        # Update previous estimates for next iteration
        x0 = x_estimate
        b0 = bias_estimate
        lla = convertXYZtoLLA(x_estimate)
        data.append([time, x_estimate[0], x_estimate[1], x_estimate[2], lla[0], lla[1], lla[2]])

    df_ans = pd.DataFrame(data, columns=["GPS_Unique_Time", "Pos_X", "Pos_Y", "Pos_Z", "Lat", "Lon", "Alt"])
    return df_ans

def convertXYZtoLLA(val):
    return navpy.ecef2lla(val)

def ParseToCSV(input_filepath):
    filename = os.path.splitext(os.path.basename(input_filepath))[0]
    data = []
    fields = ['GPS time', 'SatPRN (ID)', 'Sat.X', 'Sat.Y', 'Sat.Z', 'Pseudo-Range', 'CN0']

    # Open the CSV file and iterate over its rows
    with open(input_filepath) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0][0] == '#':
                if 'Fix' in row[0]:
                    android_fixes = [row[1:]]
                elif 'Raw' in row[0]:
                    measurements = [row[1:]]
            else:
                if row[0] == 'Fix':
                    android_fixes.append(row[1:])
                elif row[0] == 'Raw':
                    measurements.append(row[1:])

    android_fixes = pd.DataFrame(android_fixes[1:], columns=android_fixes[0])
    measurements = pd.DataFrame(measurements[1:], columns=measurements[0])

    # Format satellite IDs
    measurements.loc[measurements['Svid'].str.len() == 1, 'Svid'] = '0' + measurements['Svid']
    measurements.loc[measurements['ConstellationType'] == '1', 'Constellation'] = 'G'
    measurements.loc[measurements['ConstellationType'] == '3', 'Constellation'] = 'R'
    measurements['SvName'] = measurements['Constellation'] + measurements['Svid']

    # Remove all non-GPS measurements
    measurements = measurements.loc[measurements['Constellation'] == 'G']

    # Extract SatPRN (ID) from the data
    satPRN = measurements['SvName'].tolist()
    uniqSatPRN = measurements['SvName'].unique().tolist()

    # Convert columns to numeric representation

    # Filter by C/N0 (Carrier-to-Noise Density Ratio)
    measurements['Cn0DbHz'] = pd.to_numeric(measurements['Cn0DbHz'])  # Ensure Cn0DbHz column is numeric

    measurements['TimeNanos'] = pd.to_numeric(measurements['TimeNanos'])
    measurements['FullBiasNanos'] = pd.to_numeric(measurements['FullBiasNanos'])
    measurements['ReceivedSvTimeNanos'] = pd.to_numeric(measurements['ReceivedSvTimeNanos'])
    measurements['PseudorangeRateMetersPerSecond'] = pd.to_numeric(measurements['PseudorangeRateMetersPerSecond'])
    measurements['ReceivedSvTimeUncertaintyNanos'] = pd.to_numeric(measurements['ReceivedSvTimeUncertaintyNanos'])

    # A few measurement values are not provided by all phones
    # We'll check for them and initialize them with zeros if missing
    if 'BiasNanos' in measurements.columns:
        measurements['BiasNanos'] = pd.to_numeric(measurements['BiasNanos'])
    else:
        measurements['BiasNanos'] = 0
    if 'TimeOffsetNanos' in measurements.columns:
        measurements['TimeOffsetNanos'] = pd.to_numeric(measurements['TimeOffsetNanos'])
    else:
        measurements['TimeOffsetNanos'] = 0

    measurements['GpsTimeNanos'] = measurements['TimeNanos'] - (
                measurements['FullBiasNanos'] - measurements['BiasNanos'])
    gpsepoch = datetime(1980, 1, 6, 0, 0, 0)
    measurements['UnixTime'] = pd.to_datetime(measurements['GpsTimeNanos'], utc=True, origin=gpsepoch)
    measurements['UnixTime'] = measurements['UnixTime']

    # Split data into measurement epochs
    measurements['Epoch'] = 0
    measurements.loc[
        measurements['UnixTime'] - measurements['UnixTime'].shift() > timedelta(milliseconds=200), 'Epoch'] = 1
    measurements['Epoch'] = measurements['Epoch'].cumsum()

    # Extract GPS time from the data
    gpsTime = measurements['UnixTime'].tolist()

    # Calculate pseudorange in seconds
    WEEKSEC = 604800
    measurements['tRxGnssNanos'] = measurements['TimeNanos'] + measurements['TimeOffsetNanos'] - (measurements['FullBiasNanos'].iloc[0] + measurements['BiasNanos'].iloc[0])
    measurements['GpsWeekNumber'] = np.floor(1e-9 * measurements['tRxGnssNanos'] / WEEKSEC)
    measurements['tRxSeconds'] = 1e-9*measurements['tRxGnssNanos'] - WEEKSEC * measurements['GpsWeekNumber']
    measurements['tTxSeconds'] = 1e-9*(measurements['ReceivedSvTimeNanos'] + measurements['TimeOffsetNanos'])
    measurements['prSeconds'] = measurements['tRxSeconds'] - measurements['tTxSeconds']

    # Convert to meters
    measurements['PrM'] = LIGHTSPEED * measurements['prSeconds']
    measurements['PrSigmaM'] = LIGHTSPEED * 1e-9 * measurements['ReceivedSvTimeUncertaintyNanos']
    manager = ephemeris_manager.EphemerisManager(ephemeris_data_directory)
    # Calculate satellite Y,X,Z coordinates
    # loop to go through each timezone of satellites

    for i in range(len(measurements['Epoch'].unique())):
        epoch = i
        num_sats = 0
        while num_sats < 5:
            one_epoch = measurements.loc[
                (measurements['Epoch'] == epoch) & (measurements['prSeconds'] < 0.1)].drop_duplicates(subset='SvName')

            if len(one_epoch) < 2:  # Check if there are at least 2 rows
                epoch += 1
                continue

            timestamp = one_epoch.iloc[1]['UnixTime'].to_pydatetime(warn=False)
            one_epoch.set_index('SvName', inplace=True)
            num_sats = len(one_epoch.index)
            epoch += 1

        if len(one_epoch) >= 2:  # Ensure one_epoch is valid before proceeding
            sats = one_epoch.index.unique().tolist()
            ephemeris = manager.get_ephemeris(timestamp, sats)

        def calculate_satellite_position(ephemeris, transmit_time):
            mu = 3.986005e14
            OmegaDot_e = 7.2921151467e-5
            F = -4.442807633e-10
            sv_position = pd.DataFrame()
            sv_position['sv'] = ephemeris.index
            sv_position.set_index('sv', inplace=True)
            sv_position['t_k'] = transmit_time - ephemeris['t_oe']
            A = ephemeris['sqrtA'].pow(2)
            n_0 = np.sqrt(mu / A.pow(3))
            n = n_0 + ephemeris['deltaN']
            M_k = ephemeris['M_0'] + n * sv_position['t_k']
            E_k = M_k
            err = pd.Series(data=[1] * len(sv_position.index))
            i = 0
            while err.abs().min() > 1e-8 and i < 10:
                new_vals = M_k + ephemeris['e'] * np.sin(E_k)
                err = new_vals - E_k
                E_k = new_vals
                i += 1

            sinE_k = np.sin(E_k)
            cosE_k = np.cos(E_k)
            delT_r = F * ephemeris['e'].pow(ephemeris['sqrtA']) * sinE_k
            delT_oc = transmit_time - ephemeris['t_oc']
            sv_position['delT_sv'] = ephemeris['SVclockBias'] + ephemeris['SVclockDrift'] * delT_oc + ephemeris[
                'SVclockDriftRate'] * delT_oc.pow(2)

            v_k = np.arctan2(np.sqrt(1 - ephemeris['e'].pow(2)) * sinE_k, (cosE_k - ephemeris['e']))

            Phi_k = v_k + ephemeris['omega']

            sin2Phi_k = np.sin(2 * Phi_k)
            cos2Phi_k = np.cos(2 * Phi_k)

            du_k = ephemeris['C_us'] * sin2Phi_k + ephemeris['C_uc'] * cos2Phi_k
            dr_k = ephemeris['C_rs'] * sin2Phi_k + ephemeris['C_rc'] * cos2Phi_k
            di_k = ephemeris['C_is'] * sin2Phi_k + ephemeris['C_ic'] * cos2Phi_k

            u_k = Phi_k + du_k

            r_k = A * (1 - ephemeris['e'] * np.cos(E_k)) + dr_k

            i_k = ephemeris['i_0'] + di_k + ephemeris['IDOT'] * sv_position['t_k']

            x_k_prime = r_k * np.cos(u_k)
            y_k_prime = r_k * np.sin(u_k)

            Omega_k = ephemeris['Omega_0'] + (ephemeris['OmegaDot'] - OmegaDot_e) * sv_position['t_k'] - OmegaDot_e * \
                      ephemeris['t_oe']

            sv_position['x_k'] = x_k_prime * np.cos(Omega_k) - y_k_prime * np.cos(i_k) * np.sin(Omega_k)
            sv_position['y_k'] = x_k_prime * np.sin(Omega_k) + y_k_prime * np.cos(i_k) * np.cos(Omega_k)
            sv_position['z_k'] = y_k_prime * np.sin(i_k)
            return sv_position

        sv_position = calculate_satellite_position(ephemeris, one_epoch['tTxSeconds'])

        Yco = sv_position['y_k'].tolist()
        Xco = sv_position['x_k'].tolist()
        Zco = sv_position['z_k'].tolist()

        # Calculate CN0 values
        epoch = i
        num_sats = 0
        while num_sats < 5:
            one_epoch = measurements.loc[
                (measurements['Epoch'] == epoch) & (measurements['prSeconds'] < 0.1)].drop_duplicates(subset='SvName')

            # Check if one_epoch is empty
            if one_epoch.empty:
                epoch += 1
                continue

            timestamp = one_epoch.iloc[0]['UnixTime'].to_pydatetime(warn=False)
            one_epoch.set_index('SvName', inplace=True)
            num_sats = len(one_epoch.index)
            epoch += 1

        CN0 = one_epoch['Cn0DbHz'].tolist()
        pseudo_range = (one_epoch['PrM'] + LIGHTSPEED * sv_position['delT_sv']).to_numpy()

    # saving all the above data into csv file
        for i in range(len(Yco)):
            gpsTime[i] = timestamp
            row = [gpsTime[i], satPRN[i], Xco[i], Yco[i], Zco[i], pseudo_range[i], CN0[i]]
            data.append(row)

    output_csv_filepath = os.path.join(outcomes_dir, f"{filename}.csv")
    # Write data to CSV file
    with open(output_csv_filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write the header
        writer.writerow(fields)

        # Write the data
        writer.writerows(data)
    return


def original_gnss_to_position(input_filepath):
    ParseToCSV(input_filepath)
    filename = os.path.splitext(os.path.basename(input_filepath))[0]

    input_fpath = os.path.join(outcomes_dir, filename + '.csv')

    # Open the CSV file
    csvfile = open(input_fpath, newline='')

    positional_df = positioning_algorithm(csvfile)

    print("Positional Algo succeeded, creating CSV and KML files.")
    existing_df = pd.read_csv(input_fpath)
    existing_df = pd.concat([existing_df, positional_df], axis=1)
    existing_df.to_csv(input_fpath, index=False)

    # Create a KML object
    kml = simplekml.Kml()

    df_filtered = moving_average_filter(existing_df)

    # Accumulate coordinates for the LineString
    coords = []

    # Iterate over the data
    for index, row in df_filtered.iterrows():
        gps_time = row['GPS_Unique_Time']

        if 0 < row['Alt'] < 1000:

            coords.append((row['Lon'], row['Lat'], row['Alt']))

            # Create a point placemark
            pnt = kml.newpoint(name=str(row['GPS_Unique_Time']), coords=[(row['Lon'], row['Lat'], row['Alt'])])

            # Add time information to the placemark
            gps_times = pd.to_datetime(gps_time)
            # Debug print to check the altitude before filtering
            # print(f"Processing row {index}: Alt={row['Alt']} GPSTime:{gps_time}")
            if not pd.isna(gps_times):
                pnt.timestamp.when = gps_times.strftime('%Y-%m-%dT%H:%M:%SZ')

    # Create a LineString for the path
    linestring = kml.newlinestring(name="Path", description="GPS Path")
    linestring.coords = coords
    linestring.altitudemode = simplekml.AltitudeMode.relativetoground  # Adjust altitude mode as needed

    linestring.style.linestyle.color = simplekml.Color.red  # Change color to red
    linestring.style.linestyle.width = 3  # Change width if needed

    # Specify the path for saving the KML file
    # output_path = os.path.join(filename + '.kml')
    output_kml_filepath = os.path.join(outcomes_dir,filename + '.kml')
    # Save the KML file
    kml.save(output_kml_filepath)


# Added for mor accuracy creating the kml.
def moving_average_filter(df, window_size=5):
    # Ensure that Alt values are non-negative before applying the filter
    # df = df[df['Alt'] >= 0].copy()

    df['Pos_X'] = df['Pos_X'].rolling(window=window_size, min_periods=1).mean()
    df['Pos_Y'] = df['Pos_Y'].rolling(window=window_size, min_periods=1).mean()
    df['Pos_Z'] = df['Pos_Z'].rolling(window=window_size, min_periods=1).mean()
    df['Lat'] = df['Lat'].rolling(window=window_size, min_periods=1).mean()
    df['Lon'] = df['Lon'].rolling(window=window_size, min_periods=1).mean()
    df['Alt'] = df['Alt'].rolling(window=window_size, min_periods=1).mean()

    rolling_avg_alt = df['Alt'].rolling(window=window_size, min_periods=1).mean()

    # Calculate the absolute difference from rolling average
    diff_from_avg = np.abs(df['Alt'] - rolling_avg_alt)

    # Replace values in 'Alt' column with -100000 where difference is large
    df.loc[diff_from_avg > 50, 'Alt'] = -100000

    return df


def main():
    while True:
        # Hide the main tkinter window
        root = Tk()
        root.withdraw()
        root.focus_force()
        root.attributes('-topmost', True)

        # Show an "Open" dialog box and return the path to the selected file
        input_filepath = askopenfilename(title="Select GNSS Log File", initialdir=gnss_log_samples_dir, filetypes=[("Txt files", "*.txt")])
        root.destroy()
        if not input_filepath:
            print("No file selected. Exiting...")
            sys.exit()

        original_gnss_to_position(input_filepath)

        again = input("Do you want to continue for another run? (1 for yes, 2 for no): ").strip()
        if again == '2':
            break

if __name__ == "__main__":
    main()