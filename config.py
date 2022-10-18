import numpy as np

# To save memory and computing power, this script allows users to exclude all
# data before a given time t_0.

t_0 = 135.0 # [s]
t_end = np.Inf # [s]


# ============================ Sensor parameters ============================ #

# analogue accelerometers
a_x_sens = 0.020 # Sensitivity [V/gee]
a_x_offset = 2.53 # Offset [V]. Nominal value: 2.5 V
a_x_max = 100 # Sensor limit in the x-direction [gee]

a_y_sens = 0.040 # Sensitivity [V/gee]
a_y_offset = 2.50 # Offset [V]. Nominal value: 2.5 V
a_y_max = 50 # Sensor limit in the y-direction [gee]


# External and internal temperature sensors
temp_ext_gain = 15 # As a fraction, not in dB!
temp_ext_offset = 0.0715 # Offset after gain [V]
temp_int_gain = 5.3 # As a fraction, not in dB!
temp_int_offset = 0.0715 # Offset after gain [V]


# NTC
R_fixed = 1e4 # [ohm]
R_ref = 1e4 # [ohm]
A_1 = 3.354016e-3 # [–]
B_1 = 2.569850e-4 # [K^(-1)]
C_1 = 2.620131e-6 # [K^(-2)]
D_1 = 6.383091e-8 # [K^(-3)]


# IMU
a_x_imu_sens = 7.32e-4 # Sensitivity [gee/LSB]
a_y_imu_sens = 7.32e-4 # Sensitivity [gee/LSB]
a_z_imu_sens = 7.32e-4 # Sensitivity [gee/LSB]
a_x_imu_offset = 0 # Offset [gee]
a_y_imu_offset = 0 # Offset [gee]
a_z_imu_offset = 0 # Offset [gee]

ang_vel_x_sens = 0.07 # Sensitivity [dps/LSB]
ang_vel_y_sens = 0.07 # Sensitivity [dps/LSB]
ang_vel_z_sens = 0.07 # Sensitivity [dps/LSB]
ang_vel_x_offset = 0 # Offset [dps]
ang_vel_y_offset = 0 # Offset [dps]
ang_vel_z_offset = 0 # Offset [dps]

mag_x_sens = 1.4e-4 # Sensitivity [gauss/LSB]
mag_y_sens = 1.4e-4 # Sensitivity [gauss/LSB]
mag_z_sens = 1.4e-4 # Sensitivity [gauss/LSB]
mag_x_offset = 0 # Offset [gauss]
mag_y_offset = 0 # Offset [gauss]
mag_z_offset = 0 # Offset [gauss]


# Power sensor
voltage_sensor_gain = 6.064 # As a fraction, not in dB!
R_current_sensor = 2e-3 # [ohm]


# ============================= Channel set-up ============================== #

# The dictionaries below serves several purposes:
#
#    1) Limit which channels are loaded (to ensure greater computational
#       performance)
#
#    2) Identify the data streams with the correct sensors
#
#    3) Simplify and/or clarify channel names. Among other things, DeweSoft
#       appends the data unit to the channel names (i.e. 'Time (s)' rather
#       than 'Time'). Some channel names are also rather cryptic
#
# It is cruical that the lefmost channel names corresponds exactly to the
# column names used in the raw data file!!!


analogue_channels = {
    'A6 (-)': 'raw_pressure',
    'A5 (-)': 'raw_light',
    'A7 (-)': 'raw_voltage_analogue',
    'A1 (-)': 'raw_temp_ext',
    'A0 (-)': 'raw_temp_int',
    'A3 (-)': 'raw_a_x',
    'A4 (-)': 'raw_a_y',
    'A2 (-)': 'raw_mag',
    'internal_temp (K)': 'internal_temp',
    'external_temp (K)': 'external_temp',
    'analog_ax (m/s^2)': 'ax',
    'analog_ay (m/s^2)': 'ay',
    'analogue_voltage (V)': 'voltage',
    'pressure (Pa)': 'pressure',
    'magnetometer (-)': 'mag',
    'photometer (-)': 'photometer',
    }

temp_array_channels = {
    'array_temp0 (-)': 'raw_temp_array_0',
    'array_tenp1 (-)': 'raw_temp_array_1',
    'array_temp2 (-)': 'raw_temp_array_2',
    'array_temp3 (-)': 'raw_temp_array_3',
    'array_temp4 (-)': 'raw_temp_array_4',
    'array_temp5 (-)': 'raw_temp_array_5',
    'array_temp6 (-)': 'raw_temp_array_6',
    'array_temp7 (-)': 'raw_temp_array_7',
    'array_temp8 (-)': 'raw_temp_array_8',
    'array_temp9 (-)': 'raw_temp_array_9',
    'array_temp_0 (-)': 'temp_array_0',
    'array_temp_1 (-)': 'temp_array_1',
    'array_temp_2 (-)': 'temp_array_2',
    'array_temp_3 (-)': 'temp_array_3',
    'array_temp_4 (-)': 'temp_array_4',
    'array_temp_5 (-)': 'temp_array_5',
    'array_temp_6 (-)': 'temp_array_6',
    'array_temp_7 (-)': 'temp_array_7',
    'array_temp_8 (-)': 'temp_array_8',
    'array_temp_9 (-)': 'temp_array_9',
    }

power_sensor_channels = {
    'array_voltage (-)': 'raw_array_voltage',
    'array_current (-)': 'raw_array_current',
    'voltage_array (V)': 'array_voltage',
    'current_array (A)': 'array_current',
    }

gps_channels = {
    'gps_satellites (-)': 'raw_satellites',
    'gps_long (-)': 'raw_long',
    'gps_lat (-)': 'raw_lat',
    'gps_altitude (-)': 'raw_height',
    'gps_speed (-)': 'raw_speed',
    "speed (m/s)": "speed",
    "lat (deg)": 'lat',
    "long (deg)": 'long',
    "altitude (m)": 'height',
    }

imu_channels = {
    'P0/gps_imu/imu_ax (-)': 'raw_a_x_imu',
    'P0/gps_imu/imu_ay (-)': 'raw_a_y_imu',
    'P0/gps_imu/imu_az (-)': 'raw_a_z_imu',
    'P0/gps_imu/imu_gx (-)': 'raw_ang_vel_x',
    'P0/gps_imu/imu_gy (-)': 'raw_ang_vel_y',
    'P0/gps_imu/imu_gz (-)': 'raw_ang_vel_z',
    'P0/gps_imu/imu_mx (-)': 'raw_mag_x',
    'P0/gps_imu/imu_my (-)': 'raw_mag_y',
    'P0/gps_imu/imu_mz (-)': 'raw_mag_z',
    "Formula 12/imu_ax (m/s^2)": "imu_ax",
    "Formula 12/imu_ay (m/s^2)": "imu_ay",
    "Formula 12/imu_az (m/s^2)": "imu_az",
    "Formula 12/imu_gx (rad/s)": "imu_gx",
    "Formula 12/imu_gy (rad/s)": "imu_gy",
    "Formula 12/imu_gz (rad/s)": "imu_gz",
    "Formula 12/imu_mx (T)": "imu_mx",
    "Formula 12/imu_my (T)": "imu_my",
    "Formula 12/imu_mz (T)": "imu_mz",
    }

misc_channels = {
    'Time (s)': 't',
    'main_counter (-)': 'framecounter',
    }




# Create one large dictionary of all channels to be imported

channels = {
    **analogue_channels,
    **temp_array_channels,
    **power_sensor_channels,
    **gps_channels,
    **imu_channels,
    **misc_channels
    }


# Replace dictionaries with list of new channel names. This makes it easy to
# write code for, say, all the temp-array sensors, as the entire list can be
# reached by using "temp_array_channels" or, alternatively
# "x in temp_array_channels". See further down in the code for examples.

analogue_channels = list(analogue_channels.values())
temp_array_channels = list(temp_array_channels.values())
power_sensor_channels = list(power_sensor_channels.values())
gps_channels = list(gps_channels.values())
imu_channels = list(imu_channels.values())
misc_channels = list(misc_channels.values())