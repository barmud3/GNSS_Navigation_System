import sys, os, csv
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import navpy
from gnssutils import EphemerisManager
import simplekml
from filterpy.kalman import KalmanFilter

WEEKSEC = 604800
LIGHTSPEED = 2.99792458e8

parent_directory = os.getcwd()
ephemeris_data_directory = os.path.join(parent_directory, 'data')
sys.path.insert(0, parent_directory)
# Get path to sample file in data directory, which is located in the parent directory of this notebook
input_filepath = os.path.join(parent_directory, 'data', 'sample', 'Driving.txt')

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

android_fixes = pd.DataFrame(android_fixes[1:], columns = android_fixes[0])
measurements = pd.DataFrame(measurements[1:], columns = measurements[0])


# Format satellite IDs
measurements.loc[measurements['Svid'].str.len() == 1, 'Svid'] = '0' + measurements['Svid']
measurements.loc[measurements['ConstellationType'] == '1', 'Constellation'] = 'G'
measurements.loc[measurements['ConstellationType'] == '3', 'Constellation'] = 'R'

measurements['SatPRN (ID)'] = measurements['Constellation'] + measurements['Svid']

# Remove all non-GPS measurements
measurements = measurements.loc[measurements['Constellation'] == 'G']

# Convert columns to numeric representation
measurements['Cn0DbHz'] = pd.to_numeric(measurements['Cn0DbHz'])
measurements['TimeNanos'] = pd.to_numeric(measurements['TimeNanos'])
measurements['FullBiasNanos'] = pd.to_numeric(measurements['FullBiasNanos'])
measurements['ReceivedSvTimeNanos']  = pd.to_numeric(measurements['ReceivedSvTimeNanos'])
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

measurements['GpsTimeNanos'] = measurements['TimeNanos'] - (measurements['FullBiasNanos'] - measurements['BiasNanos'])
gpsepoch = datetime(1980, 1, 6, 0, 0, 0)
measurements['UnixTime'] = pd.to_datetime(measurements['GpsTimeNanos'], utc = True, origin=gpsepoch)
measurements['UnixTime'] = measurements['UnixTime']

# Split data into measurement epochs
measurements['Epoch'] = 0
measurements.loc[measurements['UnixTime'] - measurements['UnixTime'].shift() > timedelta(milliseconds=200), 'Epoch'] = 1
measurements['Epoch'] = measurements['Epoch'].cumsum()

# This should account for rollovers since it uses a week number specific to each measurement
measurements['tRxGnssNanos'] = measurements['TimeNanos'] + measurements['TimeOffsetNanos'] - (measurements['FullBiasNanos'].iloc[0] + measurements['BiasNanos'].iloc[0])
measurements['GpsWeekNumber'] = np.floor(1e-9 * measurements['tRxGnssNanos'] / WEEKSEC)
measurements['tRxSeconds'] = 1e-9*measurements['tRxGnssNanos'] - WEEKSEC * measurements['GpsWeekNumber']
measurements['tTxSeconds'] = 1e-9*(measurements['ReceivedSvTimeNanos'] + measurements['TimeOffsetNanos'])
# Calculate pseudorange in seconds
measurements['prSeconds'] = measurements['tRxSeconds'] - measurements['tTxSeconds']
# Convert to meters
measurements['PrM'] = LIGHTSPEED * measurements['prSeconds']
measurements['PrSigmaM'] = LIGHTSPEED * 1e-9 * measurements['ReceivedSvTimeUncertaintyNanos']

manager = EphemerisManager(ephemeris_data_directory)

epoch = 0
num_sats = 0
while num_sats < 5:
    one_epoch = measurements.loc[(measurements['Epoch'] == epoch) & (measurements['prSeconds'] < 0.1)].drop_duplicates(subset='SatPRN (ID)')
    timestamp = one_epoch.iloc[0]['UnixTime'].to_pydatetime(warn=False)
    one_epoch.set_index('SatPRN (ID)', inplace=True)
    num_sats = len(one_epoch.index)
    epoch += 1

sats = one_epoch.index.unique().tolist()
ephemeris_df = manager.get_ephemeris(timestamp, sats)
ephemeris = pd.DataFrame(ephemeris_df)
# ephemeris_df = pd.DataFrame(ephemeris)

# # Specify the output CSV file path
# output_csv_path = 'path_to_output_csv_file.csv'

# # Export ephemeris data to CSV
# ephemeris_df.to_csv(output_csv_path, index=False)
# exit(1)

# Reorder the columns to include 'SvName' as the first column
one_epoch_selected = one_epoch[['Cn0DbHz', 'UnixTime', 'tTxSeconds']]
one_epoch_selected['tTxSeconds'] = pd.to_timedelta(one_epoch_selected['tTxSeconds'], unit='s')
one_epoch_selected['GPS time'] = one_epoch_selected.apply(lambda row: row['UnixTime'] + row['tTxSeconds'], axis=1)
# Drop the original 'UnixTime' and 'tTxSeconds' columns
one_epoch_selected.drop(['UnixTime', 'tTxSeconds'], axis=1, inplace=True)

def least_squares(xs, measured_pseudorange, x0, b0):
    dx = 100*np.ones(3)
    b = b0
    # set up the G matrix with the right dimensions. We will later replace the first 3 columns
    # note that b here is the clock bias in meters equivalent, so the actual clock bias is b/LIGHTSPEED
    G = np.ones((measured_pseudorange.size, 4))
    iterations = 0
    while np.linalg.norm(dx) > 1e-3:
        # Eq. (2):
        r = np.linalg.norm(xs - x0, axis=1)
        # Eq. (1):
        phat = r + b0
        # Eq. (3):
        deltaP = measured_pseudorange - phat
        G[:, 0:3] = -(xs - x0) / r[:, None]
        # Eq. (4):
        sol = np.linalg.inv(np.transpose(G) @ G) @ np.transpose(G) @ deltaP
        # Eq. (5):
        dx = sol[0:3]
        db = sol[3]
        x0 = x0 + dx
        b0 = b0 + db
    norm_dp = np.linalg.norm(deltaP)
    return x0, b0, norm_dp

def ecef_to_lla(location):
    lla = navpy.ecef2lla(location)  # Specify latitude and longitude units
    latitude = lla[0]
    longitude = lla[1]
    altitude = lla[2]
    return latitude, longitude, altitude

def calculate_satellite_position(ephemeris, transmit_time):
    mu = 3.986005e14
    OmegaDot_e = 7.2921151467e-5
    F = -4.442807633e-10
    
    sv_position = pd.DataFrame(index=ephemeris.index)
    sv_position['t_k'] = transmit_time - ephemeris['t_oe']
    
    A = ephemeris['sqrtA'] ** 2
    n_0 = np.sqrt(mu / A ** 3)
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
    delT_r = F * ephemeris['e'] * A * sinE_k
    delT_oc = transmit_time - ephemeris['t_oc']
    
    sv_position['delT_sv'] = (
        ephemeris['SVclockBias']
        + ephemeris['SVclockDrift'] * delT_oc
        + ephemeris['SVclockDriftRate'] * delT_oc ** 2
    )
    
    v_k = np.arctan2(np.sqrt(1 - ephemeris['e'] ** 2) * sinE_k, cosE_k - ephemeris['e'])
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
    
    Omega_k = (
        ephemeris['Omega_0']
        + (ephemeris['OmegaDot'] - OmegaDot_e) * sv_position['t_k']
        - OmegaDot_e * ephemeris['t_oe']
    )
    
    sv_position['x_k'] = (
        x_k_prime * np.cos(Omega_k) - y_k_prime * np.cos(i_k) * np.sin(Omega_k)
    )
    sv_position['y_k'] = (
        x_k_prime * np.sin(Omega_k) + y_k_prime * np.cos(i_k) * np.cos(Omega_k)
    )
    sv_position['z_k'] = y_k_prime * np.sin(i_k)
    
    return sv_position[['x_k', 'y_k', 'z_k']]

def plot_positions(kml, name, coords, color):
    ls = kml.newlinestring(name=name)
    ls.coords = coords
    ls.extrude = 1
    ls.altitudemode = simplekml.AltitudeMode.relativetoground
    ls.style.linestyle.width = 2
    ls.style.linestyle.color = color

kml = simplekml.Kml()
coords = []
output_filepath = os.path.join(parent_directory, 'output', 'pos.csv')
output_df = pd.DataFrame(columns = ['Latitude', 'Longitude', 'Altitude'])

# Initialize Kalman filter for disruption detection
kf = KalmanFilter(dim_x=4, dim_z=2)
kf.x = np.array([0., 0., 0., 0.]) # initial state
kf.F = np.array([[1., 1., 0., 0.], 
                 [0., 1., 0., 0.],
                 [0., 0., 1., 1.],
                 [0., 0., 0., 1.]]) # state transition matrix
kf.H = np.array([[1., 0., 0., 0.], 
                 [0., 0., 1., 0.]]) # measurement function
kf.P *= 1000. # covariance matrix
kf.R = 5 # measurement noise

for epoch in range(1, measurements['Epoch'].max()+1):
    one_epoch = measurements.loc[(measurements['Epoch'] == epoch) & (measurements['prSeconds'] < 0.1)].drop_duplicates(subset='SatPRN (ID)')
    timestamp = one_epoch.iloc[0]['UnixTime'].to_pydatetime(warn=False)
    one_epoch.set_index('SatPRN (ID)', inplace=True)
    ephemeris = manager.get_ephemeris(timestamp, one_epoch.index.unique().tolist())
    sat_positions = calculate_satellite_position(ephemeris, one_epoch['tTxSeconds'])
    x_0 = np.array([0, 0, 0])
    b_0 = 0
    location, clockbias, error = least_squares(sat_positions.values, one_epoch['PrM'].values, x_0, b_0)
    
    
    # Use Kalman filter to detect disruptions
    kf.predict()
    kf.update([location[0], location[1]])
    
    # Simple threshold-based detection (you might want to adjust this)
    if error > 100 or np.abs(kf.y).sum() > 20: # Example threshold for disruption
        print(f"Disruption detected at epoch {epoch}")
        continue
    
    latitude, longitude, altitude = ecef_to_lla(location)
    coords.append((longitude, latitude, altitude))
    output_df = output_df.append({'Latitude': latitude, 'Longitude': longitude, 'Altitude': altitude}, ignore_index=True)

plot_positions(kml, 'Flight Path', coords, 'ff0000ff')
kml_filepath = os.path.join(parent_directory, 'output', 'path.kml')
kml.save(kml_filepath)

output_df.to_csv(output_filepath, index=False)

print('Position calculation and KML file generation completed.')
