import sys, os, csv
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import navpy
from gnssutils import EphemerisManager
import simplekml
import asyncio
import websockets

WEEKSEC = 604800
LIGHTSPEED = 2.99792458e8

parent_directory = os.getcwd()
ephemeris_data_directory = os.path.join(parent_directory, 'data')
sys.path.insert(0, parent_directory)

# Function to process incoming GNSS data
def process_realtime_gnss_data(data):
    global gnss_data_queue
    measurements = pd.DataFrame(data['measurements'])
    android_fixes = pd.DataFrame(data['fixes'])

    # Format satellite IDs
    measurements.loc[measurements['Svid'].str.len() == 1, 'Svid'] = '0' + measurements['Svid']
    measurements.loc[measurements['ConstellationType'] == '1', 'Constellation'] = 'G'
    measurements.loc[measurements['ConstellationType'] == '3', 'Constellation'] = 'R'
    measurements['SatPRN (ID)'] = measurements['Constellation'] + measurements['Svid']
    measurements = measurements.loc[measurements['Constellation'] == 'G']

    # Convert columns to numeric representation
    measurements['Cn0DbHz'] = pd.to_numeric(measurements['Cn0DbHz'])
    measurements['TimeNanos'] = pd.to_numeric(measurements['TimeNanos'])
    measurements['FullBiasNanos'] = pd.to_numeric(measurements['FullBiasNanos'])
    measurements['ReceivedSvTimeNanos'] = pd.to_numeric(measurements['ReceivedSvTimeNanos'])
    measurements['PseudorangeRateMetersPerSecond'] = pd.to_numeric(measurements['PseudorangeRateMetersPerSecond'])
    measurements['ReceivedSvTimeUncertaintyNanos'] = pd.to_numeric(measurements['ReceivedSvTimeUncertaintyNanos'])

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
    measurements['UnixTime'] = pd.to_datetime(measurements['GpsTimeNanos'], utc=True, origin=gpsepoch)

    measurements['Epoch'] = 0
    measurements.loc[measurements['UnixTime'] - measurements['UnixTime'].shift() > timedelta(milliseconds=200), 'Epoch'] = 1
    measurements['Epoch'] = measurements['Epoch'].cumsum()

    measurements['tRxGnssNanos'] = measurements['TimeNanos'] + measurements['TimeOffsetNanos'] - (measurements['FullBiasNanos'].iloc[0] + measurements['BiasNanos'].iloc[0])
    measurements['GpsWeekNumber'] = np.floor(1e-9 * measurements['tRxGnssNanos'] / WEEKSEC)
    measurements['tRxSeconds'] = 1e-9 * measurements['tRxGnssNanos'] - WEEKSEC * measurements['GpsWeekNumber']
    measurements['tTxSeconds'] = 1e-9 * (measurements['ReceivedSvTimeNanos'] + measurements['TimeOffsetNanos'])
    measurements['prSeconds'] = measurements['tRxSeconds'] - measurements['tTxSeconds']
    measurements['PrM'] = LIGHTSPEED * measurements['prSeconds']
    measurements['PrSigmaM'] = LIGHTSPEED * 1e-9 * measurements['ReceivedSvTimeUncertaintyNanos']

    gnss_data_queue.append(measurements)

# Define a global variable to store incoming GNSS data
gnss_data_queue = []

async def main():
    global gnss_data_queue
    manager = EphemerisManager(ephemeris_data_directory)

    while True:
        if gnss_data_queue:
            measurements = gnss_data_queue.pop(0)
            epoch = 0
            num_sats = 0
            while num_sats < 5:
                one_epoch = measurements.loc[(measurements['Epoch'] == epoch) & (measurements['prSeconds'] < 0.1)].drop_duplicates(subset='SatPRN (ID)')
                if len(one_epoch) == 0:
                    break
                timestamp = one_epoch.iloc[0]['UnixTime'].to_pydatetime(warn=False)
                one_epoch.set_index('SatPRN (ID)', inplace=True)
                num_sats = len(one_epoch.index)
                epoch += 1

            if len(one_epoch) > 0:
                sats = one_epoch.index.unique().tolist()
                ephemeris = manager.get_ephemeris(timestamp, sats)
                transmit_time = one_epoch['tTxSeconds']
                sv_positions = calculate_satellite_position(ephemeris, transmit_time)
                measurement_errors = one_epoch[['PrM', 'PrSigmaM']].to_numpy()

                positions = sv_positions[['x_k', 'y_k', 'z_k']].to_numpy()
                x0 = np.array([0, 0, 0])
                b0 = 0

                measured_location, b0, norm_dp = least_squares(positions, measurement_errors[:, 0], x0, b0)
                lat, lon, alt = ecef_to_lla(measured_location)

                kml = simplekml.Kml()
                kml.newpoint(name='Calculated Position', coords=[(lon, lat, alt)])
                kml.save(os.path.join(parent_directory, 'calculated_position.kml'))

        await asyncio.sleep(1)

def least_squares(xs, measured_pseudorange, x0, b0):
    dx = 100*np.ones(3)
    b = b0
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
    lla = navpy.ecef2lla(location, latlon_unit='deg')  # Specify latitude and longitude units
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
    A = ephemeris['sqrtA'].pow(2)
    n_0 = np.sqrt(mu / A.pow(3))
    n = n_0 + ephemeris['deltaN']
    M_k = ephemeris['M_0'] + n * sv_position['t_k']
    E_k = M_k
    err = pd.Series(data=[1]*len(sv_position.index))
    i = 0
    while err.abs().min() > 1e-8 and i < 10:
        new_vals = M_k + ephemeris['e']*np.sin(E_k)
        err = new_vals - E_k
        E_k = new_vals
        i += 1

    sinE_k = np.sin(E_k)
    cosE_k = np.cos(E_k)
    delT_r = F * ephemeris['e'].pow(ephemeris['sqrtA']) * sinE_k
    delT_oc = transmit_time - ephemeris['t_oc']
    sv_position['delT_sv'] = ephemeris['SVclockBias'] + ephemeris['SVclockDrift'] * delT_oc + ephemeris['SVclockDriftRate'] * delT_oc.pow(2)

    v_k = np.arctan2(np.sqrt(1-ephemeris['e'].pow(2))*sinE_k,(cosE_k - ephemeris['e']))

    Phi_k = v_k + ephemeris['omega']

    sin2Phi_k = np.sin(2*Phi_k)
    cos2Phi_k = np.cos(2*Phi_k)

    du_k = ephemeris['C_uc']*cos2Phi_k + ephemeris['C_us']*sin2Phi_k
    dr_k = ephemeris['C_rc']*cos2Phi_k + ephemeris['C_rs']*sin2Phi_k
    di_k = ephemeris['C_ic']*cos2Phi_k + ephemeris['C_is']*sin2Phi_k

    u_k = Phi_k + du_k
    r_k = A * (1 - ephemeris['e'] * np.cos(E_k)) + dr_k
    i_k = ephemeris['i_0'] + ephemeris['IDOT']*sv_position['t_k'] + di_k
    x_k = r_k * np.cos(u_k)
    y_k = r_k * np.sin(u_k)
    e_k = OmegaDot_e * sv_position['t_k']
    omeg_k = ephemeris['omega_0'] + e_k
    sinik = np.sin(i_k)
    cosik = np.cos(i_k)
    sinok = np.sin(omeg_k)
    cosok = np.cos(omeg_k)

    sv_position['x_k'] = x_k * cosok - y_k * cosik * sinok
    sv_position['y_k'] = x_k * sinok + y_k * cosik * cosok
    sv_position['z_k'] = y_k * sinik

    return sv_position

if __name__ == '__main__':
    asyncio.run(main())
