# Navigation System Based on Raw GNSS Measurements

## Description

This project focuses on developing a sophisticated navigation system that utilizes raw Global Navigation Satellite System (GNSS) measurements. The primary objectives include:

1. **Real-time Position Editing**: Implementing an efficient and accurate algebraic approach to dynamically adjust and refine real-time positional data.

2. **Satellite Filtering**: Incorporating the ability to filter satellites based on several criteria, including:
    - **Constellation**: Filtering satellites by their specific GNSS constellation (e.g., GPS, GLONASS, Galileo, BeiDou).
    - **Otzana**: Identifying and filtering satellites based on Otzana, a metric related to satellite health and reliability.
    - **False Satellites**: Detecting and excluding erroneous satellite signals to ensure data accuracy.

3. **Disruption Handling**: Managing and mitigating disruptions, particularly those identified as "Cairo + Beirut" scenarios, which refer to specific types of interference or anomalies that can affect GNSS signals.

4. **Disturbance Identification and Management**: Implementing an algorithm to detect disturbances in GNSS signals and effectively address them to maintain the reliability of the navigation data.

## Review of Other Projects

### 1. RTKLIB
- **Description**: RTKLIB is an open-source program library for standard and precise positioning with GNSS. It provides various positioning modes, including single, differential, RTK, and PPP (Precise Point Positioning).
- **Highlights**: Known for its versatility and support for a wide range of GNSS data formats and correction sources. RTKLIB is a robust tool but can be complex to configure and use effectively.

### 2. GPSTk (GPS Toolkit)
- **Description**: GPSTk is an open-source library that provides a suite of tools for the processing of GNSS data. It is designed for both academic and professional use.
- **Highlights**: Offers extensive functionality for GNSS data manipulation and analysis. It is highly modular, allowing users to integrate specific functionalities into their applications. However, it may require significant effort to integrate with modern real-time applications.

### 3. Google GNSS Logger
- **Description**: Google GNSS Logger is an Android app designed to log raw GNSS measurements from Android devices.
- **Highlights**: While primarily a data logging tool, it provides valuable insights into the raw GNSS data available from modern smartphones. It is useful for research and development but lacks built-in advanced processing capabilities.

### 4. RTK-GPS
- **Description**: RTK-GPS is a project aimed at providing precise positioning using Real-Time Kinematic (RTK) processing of GNSS data.
- **Highlights**: Focuses on achieving high-precision positioning. It is effective for applications requiring centimeter-level accuracy but typically requires a base station setup and may not handle disruptions and false signals as effectively without additional customization.

This project aims to build on the strengths of these existing projects while addressing specific challenges related to real-time position editing, satellite filtering, and disruption management, ultimately contributing to the advancement of GNSS-based navigation systems.
