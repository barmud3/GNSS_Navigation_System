import sys, os, csv
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import navpy
from gnssutils import EphemerisManager
import simplekml

WEEKSEC = 604800
LIGHTSPEED = 2.99792458e8
DISRUPTION_THRESHOLD = 50  # Threshold in meters for disruption detection

parent_directory = os.getcwd()
ephemeris_data_directory = os.path.join(parent_directory, 'data')
sys.path.insert(0, parent_directory)
# Get path to sample file in data directory, which is located in the parent directory of this notebook
input_filepath = os.path.join(parent_directory, 'data', 'sample', 'beirut.txt')

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
# Conver to meters
measurements['PrM'] = LIGHTSPEED * measurements['prSeconds']
measurements['PrSigmaM'] = LIGHTSPEED * 1e-9 * measurements['ReceivedSvTimeUncertaintyNanos']

manager = EphemerisManager(ephemeris_data_directory)

epoch = 0
num_sats = 0
while num_sats < 5 :
    one_epoch = measurements.loc[(measurements['Epoch'] == epoch) & (measurements['prSeconds'] < 0.1)].drop_duplicates(subset='SatPRN (ID)')
    timestamp = one_epoch.iloc[0]['UnixTime'].to_pydatetime(warn=False)
    one_epoch.set_index('SatPRN (ID)', inplace=True)
    num_sats = len(one_epoch.index)
    epoch += 1

sats = one_epoch.index.unique().tolist()
ephemeris = manager.get_ephemeris(timestamp, sats)

# Reorder the columns to include 'SvName' as the first column
one_epoch_selected = one_epoch[['Cn0DbHz', 'UnixTime', 'tTxSeconds']]
one_epoch_selected['tTxSeconds'] = pd.to_timedelta(one_epoch_selected['tTxSeconds'], unit='s')
one_epoch_selected['GPS time'] = one_epoch_selected.apply(lambda row: row['UnixTime'] + row['tTxSeconds'], axis=1)
# Drop the original 'UnixTime' and 'tTxSeconds' columns
one_epoch_selected.drop(['UnixTime', 'tTxSeconds'], axis=1, inplace=True)

# Required columns for ephemeris data
required_columns = ['t_oe', 'sqrtA', 'deltaN', 'M_0', 'e', 'w', 'C_us', 'C_uc', 'C_rs', 'C_rc', 'i_0', 'iDot', 'C_is', 'C_ic', 'omega', 'OmegaDot']

# Check if all required columns are present in ephemeris
missing_columns = [col for col in required_columns if col not in ephemeris.columns]
if missing_columns:
    raise KeyError(f"Missing columns in ephemeris data: {missing_columns}")

def least_squares(xs, measured_pseudorange, x0, b0):
    dx = 100*np.ones(3)
    b = b0
    # set up the G matrix with the right dimensions. We will later replace the first 3 columns
    # note that b here is the clock bias in meters equivalent, so the actual clock bias is b/LIGHTSPEED
    G = np.ones((measured_pseudorange.size, 4))
    iterations = 0
    while np.linalg.norm(dx) > 1e-3:
        r = np.linalg.norm(xs - x0, axis=1)
        phat = r + b0
        deltaP = measured_pseudorange - phat
        G[:, 0:3] = -(xs - x0) / r[:, None]
        sol = np.linalg.inv(np.transpose(G) @ G) @ np.transpose(G) @ deltaP
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
    sv_position = pd.DataFrame()
    sv_position['sv']= ephemeris.index
    sv_position.set_index('sv', inplace=True)
    sv_position['t_k'] = transmit_time - ephemeris['t_oe']
    A = ephemeris['sqrtA']**2
    n_0 = np.sqrt(mu / A**3)
    n = n_0 + ephemeris['deltaN']
    M_k = ephemeris['M_0'] + n * sv_position['t_k']
    E_k = M_k
    err = np.ones_like(E_k)
    i = 0
    while np.max(np.abs(err)) > 1e-10 and i < 1000:
        new_vals = M_k + ephemeris['e'] * np.sin(E_k)
        err = new_vals - E_k
        E_k = new_vals
        i = i + 1
    v_k = np.arctan2(np.sqrt(1 - ephemeris['e']**2) * np.sin(E_k), np.cos(E_k) - ephemeris['e'])
    Phi_k = v_k + ephemeris['w']
    sv_position['u_k'] = Phi_k + ephemeris['C_us'] * np.sin(2*Phi_k) + ephemeris['C_uc'] * np.cos(2*Phi_k)
    sv_position['r_k'] = A * (1 - ephemeris['e']*np.cos(E_k)) + ephemeris['C_rs'] * np.sin(2*Phi_k) + ephemeris['C_rc'] * np.cos(2*Phi_k)
    sv_position['i_k'] = ephemeris['i_0'] + ephemeris['iDot'] * sv_position['t_k'] + ephemeris['C_is'] * np.sin(2*Phi_k) + ephemeris['C_ic'] * np.cos(2*Phi_k)
    sv_position['x_k'] = sv_position['r_k'] * np.cos(sv_position['u_k'])
    sv_position['y_k'] = sv_position['r_k'] * np.sin(sv_position['u_k'])
    sv_position['Omega_k'] = ephemeris['OMEGA'] + (ephemeris['OMEGA_DOT'] - OmegaDot_e) * sv_position['t_k'] - OmegaDot_e * ephemeris['t_oe']
    sv_position['x_kdot'] = -sv_position['y_k'] * np.cos(sv_position['i_k']) * (ephemeris['OMEGA_DOT'] - OmegaDot_e) - sv_position['r_k'] * (ephemeris['iDot']*np.cos(sv_position['u_k']) - 2 * np.pi / (ephemeris['A'] ** 1.5) * np.sin(sv_position['u_k']))
    sv_position['y_kdot'] = sv_position['x_k'] * np.cos(sv_position['i_k']) * (ephemeris['OMEGA_DOT'] - OmegaDot_e) - sv_position['r_k'] * (ephemeris['iDot']*np.sin(sv_position['u_k']) + 2 * np.pi / (ephemeris['A'] ** 1.5) * np.cos(sv_position['u_k']))
    sv_position['z_kdot'] = sv_position['r_k'] * ephemeris['iDot'] * np.cos(sv_position['u_k']) + sv_position['r_k'] * (2 * np.pi / (ephemeris['A'] ** 1.5)) * np.sin(sv_position['u_k'])
    sv_position['ECEF_x'] = sv_position['x_k'] * np.cos(sv_position['Omega_k']) - sv_position['y_k'] * np.cos(sv_position['i_k']) * np.sin(sv_position['Omega_k'])
    sv_position['ECEF_y'] = sv_position['x_k'] * np.sin(sv_position['Omega_k']) + sv_position['y_k'] * np.cos(sv_position['i_k']) * np.cos(sv_position['Omega_k'])
    sv_position['ECEF_z'] = sv_position['y_k'] * np.sin(sv_position['i_k'])
    return sv_position

def calculate_positions(one_epoch, ephemeris):
    transmit_time = one_epoch['tTxSeconds'] + one_epoch['tTxSeconds']
    sv_position = calculate_satellite_position(ephemeris, transmit_time)
    sv_position = sv_position.loc[sv_position.index.intersection(one_epoch.index)]
    measured_pseudorange = one_epoch['PrM'].to_numpy()
    xs = sv_position[['ECEF_x', 'ECEF_y', 'ECEF_z']].to_numpy()
    x0 = np.zeros(3)
    b0 = 0
    position, bias, error = least_squares(xs, measured_pseudorange, x0, b0)
    return position, bias, error

# Calculate positions using primary satellites
position_primary, bias_primary, error_primary = calculate_positions(one_epoch, ephemeris)
latitude_primary, longitude_primary, altitude_primary = ecef_to_lla(position_primary)

# Select weaker satellites
weaker_sats = one_epoch[one_epoch['Cn0DbHz'] < one_epoch['Cn0DbHz'].mean()]

# Check if there are enough weaker satellites for a position fix
if len(weaker_sats) >= 5:
    weaker_ephemeris = manager.get_ephemeris(timestamp, weaker_sats.index)
    position_weaker, bias_weaker, error_weaker = calculate_positions(weaker_sats, weaker_ephemeris)
    latitude_weaker, longitude_weaker, altitude_weaker = ecef_to_lla(position_weaker)
    
    # Calculate disruption based on the difference between primary and weaker satellite positions
    disruption_distance = np.linalg.norm(np.array([latitude_primary - latitude_weaker, longitude_primary - longitude_weaker]) * np.array([111320, 40075000 / 360]))  # Convert degrees to meters for lat/lon
    
    disruption_detected = disruption_distance > DISRUPTION_THRESHOLD
else:
    disruption_detected = False

# Create KML file for the GNSS data with disruption indication
kml = simplekml.Kml()
if disruption_detected:
    kml.newpoint(name="Disruption Detected", coords=[(longitude_primary, latitude_primary, altitude_primary)], description=f"Position deviation: {disruption_distance:.2f} meters")
else:
    kml.newpoint(name="No Disruption", coords=[(longitude_primary, latitude_primary, altitude_primary)])

output_kml_filepath = os.path.join(parent_directory, 'output', 'gnss_disruption.kml')
kml.save(output_kml_filepath)
