"""
Ephemeris Manager for GNSS Satellite Systems

This script provides a class for managing and retrieving ephemeris data
for various Global Navigation Satellite Systems (GNSS). It supports GPS,
GLONASS, Galileo, and BeiDou constellations.

The EphemerisManager class can:
- Download ephemeris data from NASA and IGS servers
- Parse and process ephemeris files
- Retrieve ephemeris data for specific satellites at a given timestamp
- Handle leap seconds
"""

import os
import gzip
import shutil
import ftplib
import georinex
import numpy as np
import pandas as pd
import unlzw3
from datetime import datetime, timezone
from ftplib import FTP_TLS, FTP

class EphemerisManager:
    def __init__(self, data_directory=os.path.join(os.getcwd(), 'data', 'ephemeris')):
        self.data_directory = data_directory
        nasa_dir = os.path.join(data_directory, 'nasa')
        igs_dir = os.path.join(data_directory, 'igs')
        os.makedirs(nasa_dir, exist_ok=True)
        os.makedirs(igs_dir, exist_ok=True)
        self.data = None
        self.leapseconds = None

    def get_ephemeris(self, timestamp, satellites):
        systems = EphemerisManager.get_constellations(satellites)
        if not isinstance(self.data, pd.DataFrame):
            self.load_data(timestamp, systems)
        data = self.data
        if satellites:
            data = data.loc[data['sv'].isin(satellites)]
        data = data.loc[data['time'] < timestamp]
        data = data.sort_values('time').groupby('sv').last().drop(['index'], axis=1)
        data['Leap Seconds'] = self.leapseconds
        return data

    def get_leapseconds(self, timestamp):
        return self.leapseconds

    def load_data(self, timestamp, constellations=None):
        filepaths = EphemerisManager.get_filepaths(timestamp)
        data_list = []
        timestamp_age = datetime.now(timezone.utc) - timestamp
        if constellations is None:
            for fileinfo in filepaths.values():
                data = self.get_ephemeris_dataframe(fileinfo)
                data_list.append(data)
        else:
            legacy_systems_only = False
            if timestamp_age.days > 0:
                if legacy_systems_only:
                    data_list.append(self.get_ephemeris_dataframe(filepaths['nasa_daily_gps']))
                    if 'R' in constellations:
                        data_list.append(self.get_ephemeris_dataframe(filepaths['nasa_daily_glonass']))
                else:
                    data_list.append(self.get_ephemeris_dataframe(filepaths['nasa_daily_combined']))
            else:
                data_list.append(self.get_ephemeris_dataframe(filepaths['nasa_daily_gps']))
                if not legacy_systems_only:
                    data_list.append(self.get_ephemeris_dataframe(filepaths['bkg_daily_combined']))

        data = pd.concat(data_list, ignore_index=True)
        data.reset_index(inplace=True)
        data.sort_values('time', inplace=True, ignore_index=True)
        self.data = data

    def get_ephemeris_dataframe(self, fileinfo, constellations=None):
        filepath = fileinfo['filepath']
        url = fileinfo['url']
        directory = os.path.split(filepath)[0]
        filename = os.path.split(filepath)[1]
        if url == 'igs.bkg.bund.de':
            dest_filepath = os.path.join(self.data_directory, 'igs', filename)
        else:
            dest_filepath = os.path.join(self.data_directory, 'nasa', filename)
        decompressed_filename = os.path.splitext(dest_filepath)[0]
        if not os.path.isfile(decompressed_filename):
            if url == 'gdc.cddis.eosdis.nasa.gov':
                secure = True
            else:
                secure = False
            try:
                self.retrieve_file(url, directory, filename, dest_filepath, secure)
                self.decompress_file(dest_filepath)
            except ftplib.error_perm as err:
                print('ftp error')
                return pd.DataFrame()
        if not self.leapseconds:
            self.leapseconds = EphemerisManager.load_leapseconds(decompressed_filename)
        data = georinex.load(decompressed_filename, use=['G', 'R', 'E', 'C']).to_dataframe()
        data.dropna(how='all', inplace=True)
        data.reset_index(inplace=True)
        data['source'] = decompressed_filename
        WEEKSEC = 604800
        data['t_oc'] = pd.to_numeric(data['time'] - datetime(1980, 1, 6, 0, 0, 0))
        data['t_oc'] = 1e-9 * data['t_oc'] - WEEKSEC * np.floor(1e-9 * data['t_oc'] / WEEKSEC)
        data['time'] = data['time'].dt.tz_localize('UTC')
        data.rename(columns={'M0': 'M_0', 'Eccentricity': 'e', 'Toe': 't_oe', 'DeltaN': 'deltaN', 'Cuc': 'C_uc', 'Cus': 'C_us',
                             'Cic': 'C_ic', 'Crc': 'C_rc', 'Cis': 'C_is', 'Crs': 'C_rs', 'Io': 'i_0', 'Omega0': 'Omega_0'}, inplace=True)
        
        return data

    @staticmethod
    def get_filetype(timestamp):
        if timestamp >= datetime(2020, 12, 1, 0, 0, 0, tzinfo=timezone.utc):
            extension = '.gz'
        else:
            extension = '.Z'
        return extension

    @staticmethod
    def load_leapseconds(filename):
        with open(filename) as f:
            for line in f:
                if 'LEAP SECONDS' in line:
                    return int(line.split()[0])
                if 'END OF HEADER' in line:
                    return None

    @staticmethod
    def get_constellations(satellites):
        if isinstance(satellites, list):
            systems = {sat[0] for sat in satellites}
            return systems
        return None

    def retrieve_file(self, url, directory, filename, dest_filepath, secure=False):
        print('Retrieving ' + directory + '/' + filename + ' from ' + url)
        ftp = self.connect(url, secure)
        src_filepath = directory + '/' + filename
        try:
            with open(dest_filepath, 'wb') as handle:
                ftp.retrbinary('RETR ' + src_filepath, handle.write)
        except ftplib.error_perm as err:
            print('Failed to retrieve ' + src_filepath + ' from ' + url)
            print(err)
            os.remove(dest_filepath)
            raise ftplib.error_perm

    def decompress_file(self, filepath):
        extension = os.path.splitext(filepath)[1]
        decompressed_path = os.path.splitext(filepath)[0]
        if extension == '.gz':
            with gzip.open(filepath, 'rb') as f_in:
                with open(decompressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        elif extension == '.Z':
            with open(filepath, 'rb') as f_in:
                with open(decompressed_path, 'wb') as f_out:
                    f_out.write(unlzw3.unlzw(f_in.read()))
        os.remove(filepath)

    def connect(self, url, secure):
        if secure:
            ftp = FTP_TLS(url)
            ftp.login()
            ftp.prot_p()
        else:
            ftp = FTP(url)
            ftp.login()
        return ftp

    def listdir(self, url, directory, secure):
        ftp = self.connect(url, secure)
        dirlist = ftp.nlst(directory)
        dirlist = [x for x in dirlist]
        print(dirlist)

    @staticmethod
    def get_filepaths(timestamp):
        timetuple = timestamp.timetuple()
        extension = EphemerisManager.get_filetype(timestamp)
        filepaths = {}

        directory = 'gnss/data/daily/' + str(timetuple.tm_year) + '/brdc/'
        filename = 'BRDC00IGS_R_' + str(timetuple.tm_year) + str(timetuple.tm_yday).zfill(3) + '0000_01D_MN.rnx.gz'
        filepaths['nasa_daily_combined'] = {'filepath': directory + filename, 'url': 'gdc.cddis.eosdis.nasa.gov'}

        filename = 'brdc' + str(timetuple.tm_yday).zfill(3) + '0.' + str(timetuple.tm_year)[-2:] + 'n' + extension
        filepaths['nasa_daily_gps'] = {'filepath': directory + filename, 'url': 'gdc.cddis.eosdis.nasa.gov'}

        filename = 'brdc' + str(timetuple.tm_yday).zfill(3) + '0.' + str(timetuple.tm_year)[-2:] + 'g' + extension
        filepaths['nasa_daily_glonass'] = {'filepath': directory + filename, 'url': 'gdc.cddis.eosdis.nasa.gov'}

        directory = '/IGS/BRDC/' + str(timetuple.tm_year) + '/' + str(timetuple.tm_yday).zfill(3) + '/'
        filename = 'BRDC00WRD_S_' + str(timetuple.tm_year) + str(timetuple.tm_yday).zfill(3) + '0000_01D_MN.rnx.gz'
        filepaths['bkg_daily_combined'] = {'filepath': directory + filename, 'url': 'igs-ftp.bkg.bund.de'}

        return filepaths

if __name__ == '__main__':
    repo = EphemerisManager()
    target_time = datetime(2024, 1, 9, 12, 0, 0, tzinfo=timezone.utc)