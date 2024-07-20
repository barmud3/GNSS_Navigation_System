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

[Include instructions on how to run the script, required inputs, and how to interpret the outputs]

## Dependencies

- Python 3.x
- Libraries: pandas, numpy, simplekml, georinex.
