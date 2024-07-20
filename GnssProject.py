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
from numpy.linalg import inv, norm, LinAlgError

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

        # Apply scoring to the satellites
        df_gps_time = score_satellites(df_gps_time)

        for constellation in df_gps_time['Constellation'].unique():
            df_constellation = df_gps_time[df_gps_time['Constellation'] == constellation]
            df_constellation_sorted = df_constellation.sort_values(by='Satellite_Score', ascending=False)
            
            # Split into high and low Satellite_Score groups
            split_index = len(df_constellation_sorted) // 2
            high_score_group = df_constellation_sorted.iloc[:split_index]
            low_score_group = df_constellation_sorted.iloc[split_index:]

            for name_group, data_group in [("High_Score", high_score_group), ("Low_Score", low_score_group)]:
                xs = data_group[['Sat.X', 'Sat.Y', 'Sat.Z']].values
                measured_pseudorange = data_group['Pseudo-Range'].values
                weights = data_group['Satellite_Score'].values
                num_satellites = len(xs)

                if num_satellites < 4:
                    print(f"Skipping group {name_group} due to insufficient satellites (need at least 4, got {num_satellites})")
                    continue

                try:
                    x_estimate, bias_estimate, norm_dp = weightedLeastSquares(xs, measured_pseudorange, x0, b0, weights)
                    x0 = x_estimate
                    b0 = bias_estimate
                    lla = convertXYZtoLLA(x_estimate)
                    
                    # Append data without including Constellation
                    data.append([time, name_group, num_satellites, x_estimate[0], x_estimate[1], x_estimate[2], lla[0], lla[1], lla[2]])
                except Exception as e:
                    print(f"Error in group {name_group}:{str(e)} ")
                
                counter += 1

    # Create DataFrame without Constellation column
    df_ans = pd.DataFrame(data, columns=["GPS_Unique_Time", "Group", "Num_Satellites", "Pos_X", "Pos_Y", "Pos_Z", "Lat", "Lon", "Alt"])
    
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


def score_satellites(df_gps_time):
    # Normalize CN0 values
    df_gps_time['CN0_norm'] = (df_gps_time['CN0'] - df_gps_time['CN0'].min()) / (df_gps_time['CN0'].max() - df_gps_time['CN0'].min())
    
    # Calculate pseudorange rate (if available)
    if 'PseudorangeRateMetersPerSecond' in df_gps_time.columns:
        df_gps_time['PR_rate_norm'] = np.abs(df_gps_time['PseudorangeRateMetersPerSecond'])
        df_gps_time['PR_rate_norm'] = (df_gps_time['PR_rate_norm'] - df_gps_time['PR_rate_norm'].min()) / (df_gps_time['PR_rate_norm'].max() - df_gps_time['PR_rate_norm'].min())
    else:
        df_gps_time['PR_rate_norm'] = 1  # Default value if not available
    

    constellation_counts = df_gps_time['Constellation'].value_counts()
    expected_counts = {'G': 8, 'R': 6, 'E': 6, 'C': 6}  # Adjust these values based on typical visibility
    
    df_gps_time['Constellation_score'] = df_gps_time['Constellation'].map(
        {const: min(1, count / expected_counts.get(const, 1)) for const, count in constellation_counts.items()}
    )
    
    # Calculate composite score
    df_gps_time['Satellite_Score'] = (
        df_gps_time['CN0_norm'] * 0.6 +
        df_gps_time['PR_rate_norm'] * 0.2 +
        df_gps_time['Constellation_score'] * 0.2
    )
    
    return df_gps_time

def ParseToCSV(input_filepath):
    filename = os.path.splitext(os.path.basename(input_filepath))[0]
    data = []
    fields = ['GPS time', 'SatPRN (ID)', 'Sat.X', 'Sat.Y', 'Sat.Z', 'Pseudo-Range', 'CN0', 'PseudorangeRateMetersPerSecond', 'Constellation']

    # Open the CSV file and iterate over its rows
    with open(input_filepath) as csvfile:
        reader = csv.reader(csvfile)
        android_fixes = []
        measur = []
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
    constellation_map = {
        '1': 'G',  # GPS
        '3': 'R',  # GLONASS
        '5': 'C',  # BeiDou
        '6': 'E'   # Galileo
    }
    measur['Constellation'] = measur['ConstellationType'].map(constellation_map)
    measur['SvName'] = measur['Constellation'] + measur['Svid']

    if(detector.isDistrubt):
        measur = measur.loc[measur['Constellation'].isin(['G', 'R', 'E', 'C'])]
    else:
        measur = measur.loc[measur['Constellation'].isin(['G', 'R', 'C'])]

    # Convert columns to numeric representation
    numeric_columns = ['Cn0DbHz', 'TimeNanos', 'FullBiasNanos', 'ReceivedSvTimeNanos', 
                       'PseudorangeRateMetersPerSecond', 'ReceivedSvTimeUncertaintyNanos']
    

    for col in numeric_columns:
        measur[col] = pd.to_numeric(measur[col])

    # Handle optional columns
    for col in ['BiasNanos', 'TimeOffsetNanos']:
        if col in measur.columns:
            measur[col] = pd.to_numeric(measur[col])
        else:
            measur[col] = 0

    # Filter by C/N0 (Carrier-to-Noise Density Ratio)
    if not detector.isDistrubt:  # Make sure 'detector' is defined and accessible
        min_cn0_threshold = 30  # CN0 threshold
        measur = measur[measur['Cn0DbHz'] >= min_cn0_threshold]

    # Calculate GPS time
    measur['GpsTimeNanos'] = measur['TimeNanos'] - (measur['FullBiasNanos'] - measur['BiasNanos'])
    gps_epoch = datetime(1980, 1, 6, 0, 0, 0)
    measur['UnixTime'] = pd.to_datetime(measur['GpsTimeNanos'], utc=True, origin=gps_epoch)

    # Split data into measurement epochs
    measur['Epoch'] = (measur['UnixTime'] - measur['UnixTime'].shift() > timedelta(milliseconds=200)).cumsum()

    # Calculate pseudorange
    WEEKSEC = 604800
    measur['tRxGnssNanos'] = measur['TimeNanos'] + measur['TimeOffsetNanos'] - (measur['FullBiasNanos'].iloc[0] + measur['BiasNanos'].iloc[0])
    measur['GpsWeekNumber'] = np.floor(1e-9 * measur['tRxGnssNanos'] / WEEKSEC)
    measur['tRxSeconds'] = 1e-9 * measur['tRxGnssNanos'] - WEEKSEC * measur['GpsWeekNumber']
    measur['tTxSeconds'] = 1e-9 * (measur['ReceivedSvTimeNanos'] + measur['TimeOffsetNanos'])
    measur['prSeconds'] = measur['tRxSeconds'] - measur['tTxSeconds']
    measur['PrM'] = SPEEDOFLIGHT * measur['prSeconds']
    measur['PrSigmaM'] = SPEEDOFLIGHT * 1e-9 * measur['ReceivedSvTimeUncertaintyNanos']

    manager = ephemeris_manager.EphemerisManager(ephemeris_data_dir)

    def findSatellitePosition(ephem, transmit_time):
        mu = 3.986005e14
        OmegaDot_e = 7.2921151467e-5
        F = -4.442807633e-10
        
        satell_position = pd.DataFrame(index=ephem.index)
        satell_position['t_k'] = transmit_time - ephem['t_oe']
        A = ephem['sqrtA'].pow(2)
        n_0 = np.sqrt(mu / A.pow(3))
        n = n_0 + ephem['deltaN']
        M_k = ephem['M_0'] + n * satell_position['t_k']
        E_k = M_k
        
        for _ in range(10):
            new_E_k = M_k + ephem['e'] * np.sin(E_k)
            if (np.abs(new_E_k - E_k) < 1e-8).all():
                break
            E_k = new_E_k

        sinE_k, cosE_k = np.sin(E_k), np.cos(E_k)
        v_k = np.arctan2(np.sqrt(1 - ephem['e'].pow(2)) * sinE_k, (cosE_k - ephem['e']))
        Phi_k = v_k + ephem['omega']
        sin2Phi_k, cos2Phi_k = np.sin(2 * Phi_k), np.cos(2 * Phi_k)

        du_k = ephem['C_us'] * sin2Phi_k + ephem['C_uc'] * cos2Phi_k
        dr_k = ephem['C_rs'] * sin2Phi_k + ephem['C_rc'] * cos2Phi_k
        di_k = ephem['C_is'] * sin2Phi_k + ephem['C_ic'] * cos2Phi_k

        u_k = Phi_k + du_k
        r_k = A * (1 - ephem['e'] * np.cos(E_k)) + dr_k
        i_k = ephem['i_0'] + di_k + ephem['IDOT'] * satell_position['t_k']

        x_k_prime = r_k * np.cos(u_k)
        y_k_prime = r_k * np.sin(u_k)

        Omega_k = ephem['Omega_0'] + (ephem['OmegaDot'] - OmegaDot_e) * satell_position['t_k'] - OmegaDot_e * ephem['t_oe']
        
        satell_position['x_k'] = x_k_prime * np.cos(Omega_k) - y_k_prime * np.cos(i_k) * np.sin(Omega_k)
        satell_position['y_k'] = x_k_prime * np.sin(Omega_k) + y_k_prime * np.cos(i_k) * np.cos(Omega_k)
        satell_position['z_k'] = y_k_prime * np.sin(i_k)
        
        delT_r = F * ephem['e'] * ephem['sqrtA'] * sinE_k
        delT_oc = transmit_time - ephem['t_oc']
        satell_position['delT_sv'] = ephem['SVclockBias'] + ephem['SVclockDrift'] * delT_oc + ephem['SVclockDriftRate'] * delT_oc.pow(2) + delT_r
        
        return satell_position

    # Process each epoch
    for epoch in measur['Epoch'].unique():
        one_epoch = measur.loc[
            (measur['Epoch'] == epoch) & (measur['prSeconds'] < 0.1)
        ].drop_duplicates(subset='SvName')

        if len(one_epoch) < 5:
            continue

        times_tamp = one_epoch.iloc[0]['UnixTime'].to_pydatetime(warn=False)
        one_epoch.set_index('SvName', inplace=True)
        sats = one_epoch.index.unique().tolist()
        ephemeris = manager.get_ephemeris(times_tamp, sats)

        sv_position = findSatellitePosition(ephemeris, one_epoch['tTxSeconds'])

        # Ensure alignment by using the same index
        aligned_data = pd.concat([
            one_epoch,
            sv_position[['x_k', 'y_k', 'z_k', 'delT_sv']]
        ], axis=1, join='inner')

        # Calculate pseudo_range with aligned data
        aligned_data['pseudo_range'] = aligned_data['PrM'] + SPEEDOFLIGHT * aligned_data['delT_sv']

        # Append aligned data to the output list
        for _, row in aligned_data.iterrows():
            data.append([
                times_tamp,
                row.name,  # SvName (satPRN)
                row['x_k'],
                row['y_k'],
                row['z_k'],
                row['pseudo_range'],
                row['Cn0DbHz'],
                row['PseudorangeRateMetersPerSecond'],
                row.name[0]  # Constellation (first character of SvName)
            ])

    # Write data to CSV file
    output_csv_path = os.path.join(outcomes_dir, f"{filename}.csv")
    with open(output_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(fields)
        writer.writerows(data)

    return output_csv_path


def originalGnssToPosition(input_filepath):
    ParseToCSV(input_filepath)
    file_name = os.path.splitext(os.path.basename(input_filepath))[0]
    input_path = os.path.join(outcomes_dir, file_name + '.csv')

    if detector.isDistrubt:
        positional_df = positioningAlgorithmDistrub(input_path)
    else:
        positional_df = positioningAlgorithmUndistrub(input_path)

    print("Positional Algo succeeded, creating CSV and KML files.")
    existing_df = pd.read_csv(input_path)
    existing_df = pd.concat([existing_df, positional_df], axis=1)
    existing_df.to_csv(input_path, index=False)

    kml = simplekml.Kml()
    df_filtered = movingAverageFilter(existing_df)

    # Define styles for each constellation group
    styles = {
        "G_High_Score": simplekml.Style(),
        "G_Low_Score": simplekml.Style(),
        "E_High_Score": simplekml.Style(),
        "E_Low_Score": simplekml.Style(),
        "R_High_Score": simplekml.Style(),
        "R_Low_Score": simplekml.Style(),
        "C_High_Score": simplekml.Style(),
        "C_Low_Score": simplekml.Style(),
    }

    styles["G_High_Score"].iconstyle.color = simplekml.Color.green
    styles["G_Low_Score"].iconstyle.color = simplekml.Color.red
    styles["E_High_Score"].iconstyle.color = simplekml.Color.blue
    styles["E_Low_Score"].iconstyle.color = simplekml.Color.orange
    styles["R_High_Score"].iconstyle.color = simplekml.Color.purple
    styles["R_Low_Score"].iconstyle.color = simplekml.Color.yellow
    styles["C_High_Score"].iconstyle.color = simplekml.Color.cyan
    styles["C_Low_Score"].iconstyle.color = simplekml.Color.magenta

    for key in styles.keys():
        styles[key].iconstyle.scale = 1

    # Accumulate coordinates for the LineString
    coordinates = []

    # Iterate over the data
    for index, row in df_filtered.iterrows():
        gps_time = row['GPS_Unique_Time']

        if 0 < row['Alt'] < 1000:
            coordinates.append((row['Lon'], row['Lat'], row['Alt']))

            # Create a point place-mark
            pnt = kml.newpoint(name=str(row['GPS_Unique_Time']), coords=[(row['Lon'], row['Lat'], row['Alt'])])

            # Determine the style based on constellation and group
            if(detector.isDistrubt) :
                style_key = f"{row['Constellation']}_{row['Group']}"
                if style_key in styles:
                    pnt.style = styles[style_key]

            # Add time information to the place-mark
            times_in_gps = pd.to_datetime(gps_time)
            if not pd.isna(times_in_gps):
                pnt.timestamp.when = times_in_gps.strftime('%Y-%m-%dT%H:%M:%SZ')

    # Create a LineString for the path
    linestring = kml.newlinestring(name="Path", description="GPS Path")
    linestring.coords = coordinates
    linestring.altitudemode = simplekml.AltitudeMode.relativetoground  # Adjust altitude mode as needed

    linestring.style.linestyle.color = simplekml.Color.blue  # Change color to blue
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
                 "אוטונומים\\Finish_Project\\GNSS_Navigation_System\\data\\sample\\cairo.txt "
detector = GNSSDisruptionDetector(the_data_gnss_file, num_satellites=2)


def main():

    detector.process_data()
    # Filter out the specific warning
    warnings.filterwarnings("ignore", message="In a future version of pandas all arguments of DataFrame.drop except "
                                              "for the argument 'labels' will be keyword-only")
    originalGnssToPosition(the_data_gnss_file)
    print("Is data distrubt :" , detector.isDistrubt)


if __name__ == "__main__":
    main()
