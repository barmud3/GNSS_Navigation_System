import sys, os, csv
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import navpy
from gnssutils import ephemeris_manager
import simplekml
import subprocess
from analyzeDist import GNSSDisruptionDetector
import warnings

SPEEDOFLIGHT = 2.99792458e8
ephemeris_data_dir = os.path.join('data')
# Define the folder structure
gnss_samples_dir = 'gnss_log_samples'
outcomes_dir = 'outcomes'

# Create folders if they don't exist
os.makedirs(gnss_samples_dir, exist_ok=True)
os.makedirs(outcomes_dir, exist_ok=True)


def weightedLeastSquares(xs, measured_pseudorange, x0, b0, weights):
    dX = 100 * np.ones(3)
    b = b0
    G = np.ones((measured_pseudorange.size, 4))

    while np.linalg.norm(dX) > 1e-3:
        r = np.linalg.norm(xs - x0, axis=1)
        phat = r + b0
        delta_p = measured_pseudorange - phat
        W = np.diag(weights)  # Weight matrix
        G[:, 0:3] = -(xs - x0) / r[:, None]

        # Weighted least squares solution
        solution = np.linalg.inv(G.T @ W @ G) @ G.T @ W @ delta_p
        dX = solution[0:3]
        dB = solution[3]
        x0 = x0 + dX
        b0 = b0 + dB

    norm_dp = np.linalg.norm(delta_p)
    return x0, b0, norm_dp


def positioningAlgorithmDistrub(csv_file):
    df = pd.read_csv(csv_file)
    data = []
    df_times = df['GPS time'].unique()
    x0 = np.array([0, 0, 0])
    b0 = 0
    counter = 0

    for time in df_times:
        df_gps_time = df[df['GPS time'] == time]
        df_gps_time_sorted = df_gps_time.sort_values(by='CN0', ascending=False)

        # Split the data into two groups
        split_the_index = len(df_gps_time_sorted) // 2
        high_cn0_group = df_gps_time_sorted.iloc[:split_the_index]
        low_cn0_group = df_gps_time_sorted.iloc[split_the_index:]

        for name_group, data_group in [("High_CN0", high_cn0_group), ("Low_CN0", low_cn0_group)]:
            xs = data_group[['Sat.X', 'Sat.Y', 'Sat.Z']].values
            measured_pseudorange = data_group['Pseudo-Range'].values
            weights = data_group['CN0'].values
            x_estimate, bias_estimate, norm_dp = weightedLeastSquares(xs, measured_pseudorange, x0, b0, weights)

            # Update previous estimates for next iteration
            x0 = x_estimate
            b0 = bias_estimate

            lla = convertXYZtoLLA(x_estimate)
            data.append([time, name_group, x_estimate[0], x_estimate[1], x_estimate[2], lla[0], lla[1], lla[2]])

        counter += 1

    print(counter)
    df_ans = pd.DataFrame(data, columns=["GPS_Unique_Time", "Group", "Pos_X", "Pos_Y", "Pos_Z", "Lat", "Lon", "Alt"])
    return df_ans


def positioningAlgorithmUndistrub(csv_file):
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
        x_estimate, bias_estimate, norm_dp = weightedLeastSquares(xs, measured_pseudorange, x0, b0, weights)
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
                    measur = [row[1:]]
            else:
                if row[0] == 'Fix':
                    android_fixes.append(row[1:])
                elif row[0] == 'Raw':
                    measur.append(row[1:])

    android_fixes = pd.DataFrame(android_fixes[1:], columns=android_fixes[0])
    measur = pd.DataFrame(measur[1:], columns=measur[0])

    # Format satellite IDs
    measur.loc[measur['Svid'].str.len() == 1, 'Svid'] = '0' + measur['Svid']
    measur.loc[measur['ConstellationType'] == '1', 'Constellation'] = 'G'
    measur.loc[measur['ConstellationType'] == '3', 'Constellation'] = 'R'
    measur['SvName'] = measur['Constellation'] + measur['Svid']

    # Remove all non-GPS measurements
    measur = measur.loc[measur['Constellation'] == 'G']

    # Extract SatPRN (ID) from the data
    satPRN = measur['SvName'].tolist()
    uniqSatPRN = measur['SvName'].unique().tolist()

    # Convert columns to numeric representation

    # Filter by C/N0 (Carrier-to-Noise Density Ratio)
    measur['Cn0DbHz'] = pd.to_numeric(measur['Cn0DbHz'])  # Ensure Cn0DbHz column is numeric

    measur['TimeNanos'] = pd.to_numeric(measur['TimeNanos'])
    measur['FullBiasNanos'] = pd.to_numeric(measur['FullBiasNanos'])
    measur['ReceivedSvTimeNanos'] = pd.to_numeric(measur['ReceivedSvTimeNanos'])
    measur['PseudorangeRateMetersPerSecond'] = pd.to_numeric(measur['PseudorangeRateMetersPerSecond'])
    measur['ReceivedSvTimeUncertaintyNanos'] = pd.to_numeric(measur['ReceivedSvTimeUncertaintyNanos'])

    # A few measurement values are not provided by all phones
    # We'll check for them and initialize them with zeros if missing
    if 'BiasNanos' in measur.columns:
        measur['BiasNanos'] = pd.to_numeric(measur['BiasNanos'])
    else:
        measur['BiasNanos'] = 0
    if 'TimeOffsetNanos' in measur.columns:
        measur['TimeOffsetNanos'] = pd.to_numeric(measur['TimeOffsetNanos'])
    else:
        measur['TimeOffsetNanos'] = 0

    measur['GpsTimeNanos'] = measur['TimeNanos'] - (
                measur['FullBiasNanos'] - measur['BiasNanos'])
    gps_epoch = datetime(1980, 1, 6, 0, 0, 0)
    measur['UnixTime'] = pd.to_datetime(measur['GpsTimeNanos'], utc=True, origin=gps_epoch)
    measur['UnixTime'] = measur['UnixTime']

    # Split data into measurement epochs
    measur['Epoch'] = 0
    measur.loc[
        measur['UnixTime'] - measur['UnixTime'].shift() > timedelta(milliseconds=200), 'Epoch'] = 1
    measur['Epoch'] = measur['Epoch'].cumsum()

    # Extract GPS time from the data
    gps_time = measur['UnixTime'].tolist()

    # Calculate pseudorange in seconds
    WEEKSEC = 604800
    measur['tRxGnssNanos'] = measur['TimeNanos'] + measur['TimeOffsetNanos'] - (measur['FullBiasNanos'].iloc[0] + measur['BiasNanos'].iloc[0])
    measur['GpsWeekNumber'] = np.floor(1e-9 * measur['tRxGnssNanos'] / WEEKSEC)
    measur['tRxSeconds'] = 1e-9*measur['tRxGnssNanos'] - WEEKSEC * measur['GpsWeekNumber']
    measur['tTxSeconds'] = 1e-9*(measur['ReceivedSvTimeNanos'] + measur['TimeOffsetNanos'])
    measur['prSeconds'] = measur['tRxSeconds'] - measur['tTxSeconds']

    # Convert to meters
    measur['PrM'] = SPEEDOFLIGHT * measur['prSeconds']
    measur['PrSigmaM'] = SPEEDOFLIGHT * 1e-9 * measur['ReceivedSvTimeUncertaintyNanos']
    manager = ephemeris_manager.EphemerisManager(ephemeris_data_dir)
    # Calculate satellite Y,X,Z coordinates
    # loop to go through each timezone of satellites

    for i in range(len(measur['Epoch'].unique())):
        epoch = i
        num_sats = 0
        while num_sats < 5:
            one_epoch = measur.loc[
                (measur['Epoch'] == epoch) & (measur['prSeconds'] < 0.1)].drop_duplicates(subset='SvName')

            if len(one_epoch) < 2:  # Check if there are at least 2 rows
                epoch += 1
                continue

            times_tamp = one_epoch.iloc[1]['UnixTime'].to_pydatetime(warn=False)
            one_epoch.set_index('SvName', inplace=True)
            num_sats = len(one_epoch.index)
            epoch += 1

        if len(one_epoch) >= 2:  # Ensure one_epoch is valid before proceeding
            sats = one_epoch.index.unique().tolist()
            ephemeris = manager.get_ephemeris(times_tamp, sats)

        def findSatellitePosition(ephem, transmit_time):
            mu = 3.986005e14
            OmegaDot_e = 7.2921151467e-5
            F = -4.442807633e-10
            satell_position = pd.DataFrame()
            satell_position['sv'] = ephem.index
            satell_position.set_index('sv', inplace=True)
            satell_position['t_k'] = transmit_time - ephem['t_oe']
            A = ephem['sqrtA'].pow(2)
            n_0 = np.sqrt(mu / A.pow(3))
            n = n_0 + ephem['deltaN']
            M_k = ephem['M_0'] + n * satell_position['t_k']
            E_k = M_k
            err = pd.Series(data=[1] * len(satell_position.index))
            i = 0
            while err.abs().min() > 1e-8 and i < 10:
                new_vals = M_k + ephem['e'] * np.sin(E_k)
                err = new_vals - E_k
                E_k = new_vals
                i += 1

            sinE_k = np.sin(E_k)
            cosE_k = np.cos(E_k)
            delT_r = F * ephem['e'].pow(ephem['sqrtA']) * sinE_k
            delT_oc = transmit_time - ephem['t_oc']
            satell_position['delT_sv'] = ephem['SVclockBias'] + ephem['SVclockDrift'] * delT_oc + ephem[
                'SVclockDriftRate'] * delT_oc.pow(2)

            v_k = np.arctan2(np.sqrt(1 - ephem['e'].pow(2)) * sinE_k, (cosE_k - ephem['e']))

            Phi_k = v_k + ephem['omega']

            sin2Phi_k = np.sin(2 * Phi_k)
            cos2Phi_k = np.cos(2 * Phi_k)

            du_k = ephem['C_us'] * sin2Phi_k + ephem['C_uc'] * cos2Phi_k
            dr_k = ephem['C_rs'] * sin2Phi_k + ephem['C_rc'] * cos2Phi_k
            di_k = ephem['C_is'] * sin2Phi_k + ephem['C_ic'] * cos2Phi_k

            u_k = Phi_k + du_k

            r_k = A * (1 - ephem['e'] * np.cos(E_k)) + dr_k

            i_k = ephem['i_0'] + di_k + ephem['IDOT'] * satell_position['t_k']

            x_k_prime = r_k * np.cos(u_k)
            y_k_prime = r_k * np.sin(u_k)

            Omega_k = ephem['Omega_0'] + (ephem['OmegaDot'] - OmegaDot_e) * satell_position['t_k'] - OmegaDot_e * \
                      ephem['t_oe']

            satell_position['x_k'] = x_k_prime * np.cos(Omega_k) - y_k_prime * np.cos(i_k) * np.sin(Omega_k)
            satell_position['y_k'] = x_k_prime * np.sin(Omega_k) + y_k_prime * np.cos(i_k) * np.cos(Omega_k)
            satell_position['z_k'] = y_k_prime * np.sin(i_k)
            return satell_position

        sv_position = findSatellitePosition(ephemeris, one_epoch['tTxSeconds'])

        Yco = sv_position['y_k'].tolist()
        Xco = sv_position['x_k'].tolist()
        Zco = sv_position['z_k'].tolist()

        # Calculate CN0 values
        epoch = i
        num_sats = 0
        while num_sats < 5:
            one_epoch = measur.loc[
                (measur['Epoch'] == epoch) & (measur['prSeconds'] < 0.1)].drop_duplicates(subset='SvName')

            # Check if one_epoch is empty
            if one_epoch.empty:
                epoch += 1
                continue

            times_tamp = one_epoch.iloc[0]['UnixTime'].to_pydatetime(warn=False)
            one_epoch.set_index('SvName', inplace=True)
            num_sats = len(one_epoch.index)
            epoch += 1

        CN0 = one_epoch['Cn0DbHz'].tolist()
        pseudo_range = (one_epoch['PrM'] + SPEEDOFLIGHT * sv_position['delT_sv']).to_numpy()

    # saving all the above data into csv file
        for i in range(len(Yco)):
            gps_time[i] = times_tamp
            row = [gps_time[i], satPRN[i], Xco[i], Yco[i], Zco[i], pseudo_range[i], CN0[i]]
            data.append(row)

    output_csv_path = os.path.join(outcomes_dir, f"{filename}.csv")
    # Write data to CSV file
    with open(output_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write the header
        writer.writerow(fields)

        # Write the data
        writer.writerows(data)
    return


def originalGnssToPosition(input_filepath):
    ParseToCSV(input_filepath)
    file_name = os.path.splitext(os.path.basename(input_filepath))[0]

    input_path = os.path.join(outcomes_dir, file_name + '.csv')

    # Open the CSV file
    csv_file = open(input_path, newline='')
    if detector.isDistrubt:
        positional_df = positioningAlgorithmDistrub(csv_file)
    else:
        positional_df = positioningAlgorithmUndistrub(csv_file)

    print("Positional Algo succeeded, creating CSV and KML files.")
    existing_df = pd.read_csv(input_path)
    existing_df = pd.concat([existing_df, positional_df], axis=1)
    existing_df.to_csv(input_path, index=False)

    # Create a KML object
    kml = simplekml.Kml()

    df_filtered = movingAverageFilter(existing_df)

    # Accumulate coordinates for the LineString
    coordinates = []

    # Iterate over the data
    for index, row in df_filtered.iterrows():
        gps_time = row['GPS_Unique_Time']

        if 0 < row['Alt'] < 1000:

            coordinates.append((row['Lon'], row['Lat'], row['Alt']))

            # Create a point place-mark
            pnt = kml.newpoint(name=str(row['GPS_Unique_Time']), coords=[(row['Lon'], row['Lat'], row['Alt'])])

            # Add time information to the place-mark
            times_in_gps = pd.to_datetime(gps_time)
            # Debug print to check the altitude before filtering
            # print(f"Processing row {index}: Alt={row['Alt']} GPSTime:{gps_time}")
            if not pd.isna(times_in_gps):
                pnt.timestamp.when = times_in_gps.strftime('%Y-%m-%dT%H:%M:%SZ')

    # Create a LineString for the path
    linestring = kml.newlinestring(name="Path", description="GPS Path")
    linestring.coords = coordinates
    linestring.altitudemode = simplekml.AltitudeMode.relativetoground  # Adjust altitude mode as needed

    linestring.style.linestyle.color = simplekml.Color.red  # Change color to red
    linestring.style.linestyle.width = 3  # Change width if needed

    # Specify the path for saving the KML file
    output_kml_path = os.path.join(outcomes_dir, file_name + '.kml')
    # Save the KML file
    kml.save(output_kml_path)


# Added for mor accuracy creating the kml.
def movingAverageFilter(df, window_size=5):
    # Ensure that Alt values are non-negative before applying the filter

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


the_data_gnss_file = "C:\\Users\\בר\\OneDrive\\שולחן העבודה\\מדמח\\שנה ג\\רובוטים " \
                 "אוטונומים\\Finish_Project\\GNSS_Navigation_System\\data\\sample\\Driving2.txt "
detector = GNSSDisruptionDetector(the_data_gnss_file, num_satellites=2)


def main():

    detector.process_data()
    # Filter out the specific warning
    warnings.filterwarnings("ignore", message="In a future version of pandas all arguments of DataFrame.drop except "
                                              "for the argument 'labels' will be keyword-only")
    originalGnssToPosition(the_data_gnss_file)


if __name__ == "__main__":
    main()
