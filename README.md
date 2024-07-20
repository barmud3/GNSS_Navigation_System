# GNSS Navigation System

## Project Overview

This GNSS Navigation System is an advanced tool for processing and analyzing Global Navigation Satellite System (GNSS) data. It focuses on distinguishing between real and potentially fake location data using a sophisticated scoring and grouping mechanism.

## Key Features

1. **Multi-Constellation Support**: Processes data from various GNSS constellations including GPS (G), GLONASS (R), Galileo (E), and BeiDou (C).

2. **Data Analysis and Scoring**: 
   - Sorts data by GPS time and constellation.
   - Implements a scoring system based on:
     - CN0 (Carrier-to-Noise density ratio) normalization
     - Pseudorange rate normalization
     - Constellation-specific scoring

3. **High/Low Score Grouping**: 
   - For each constellation at a specific GPS time, creates two groups:
     - High Score Group
     - Low Score Group
   - Potentially creates up to 8 groups (2 per constellation) when sufficient satellites are available.

4. **KML Output**: 
   - Generates a KML file for visual representation.
   - Distinguishes between points representing real and potentially fake locations.

5. **Ephemeris Management**: 
   - Utilizes `ephemeris_manager.py` for handling ephemeris data across multiple GNSS constellations.

## Ephemeris Manager (`ephemeris_manager.py`)

The Ephemeris Manager is a crucial component that handles ephemeris data for various GNSS satellite systems. Key functionalities include:

- Downloading ephemeris data from NASA and IGS servers
- Parsing and processing ephemeris files
- Retrieving ephemeris data for specific satellites at given timestamps
- Handling leap seconds

### Features:
- Support for GPS, GLONASS, Galileo, and BeiDou constellations
- Efficient data retrieval and caching mechanisms
- Conversion of ephemeris data into usable DataFrame format
- Leap second management

## How It Works

1. **Data Input**: The system takes raw GNSS data as input.
2. **Data Processing**: 
   - Sorts the data by GPS time and constellation.
   - Calculates scores for each satellite based on multiple factors.
3. **Grouping**: 
   - Divides satellites into high and low score groups for each constellation.
   - Requires a minimum of 4 satellites per constellation for grouping.
4. **Position Calculation**: 
   - Uses weighted least squares algorithm for position estimation.
   - Applies different algorithms for disturbed and undisturbed scenarios.
5. **Output Generation**:
   - Creates a CSV file with processed data.
   - Generates a KML file for visual representation of the trajectory, distinguishing between likely real and fake locations.

## Usage

Go to folder data -> sample -> add your own gnss raw data file (txt file).
In GnssProject at varible `the_data_gnss_file` change cairo.txt to the file name you add at sample folder.
run python GnssProject.py

## Given Data

at folder data -> sample : 
cairo.txt : distrub gnss data
Driving.txt , Fixed.txt , Walking.txt : undistrub gnss data

## Examples
### cairo.txt
#### cairo.kml:
![image](https://github.com/user-attachments/assets/d9de18a4-1666-4936-9377-dcafddaad23e)
#### cairo.csv:
![image](https://github.com/user-attachments/assets/9565e98c-aceb-4e30-b8e1-eb2f0424a8c5)

### Driving.txt
#### driving.kml:
![image](https://github.com/user-attachments/assets/9ac5812f-4f2c-44db-ad87-d19b0d1fa308)
#### driving.csv:
![image](https://github.com/user-attachments/assets/e3f19bbe-e694-46d0-991a-8a9e08321d45)



## Dependencies

- Python 3.x
- Libraries: pandas, numpy, simplekml, georinex.
