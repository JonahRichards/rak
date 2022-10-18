# -*- coding: utf-8 -*-
'''
This script refines raw Dewesoft data from an Education student rocket. The
different data streams are separated and converted to physical units, stored,
smoothed, processed and presented graphically for preliminary analysis.

The code presented below is meant to serve as a starting point for customized
data analysis, and should be expanded upon and edited as needed.

Andøya Space Education

Created on Tue Aug 3 2021 at 16:20:00.
Last modified [dd.mm.yyyy]: 05.10.2022
@author: bjarne.ådnanes.bergtun
The tkinter code is based on a script by odd-einar.cedervall.nervik
'''

import tkinter as tk # GUI
import tkinter.filedialog as fd # file dialogs
import os # OS-specific directory manipulation
from os import path # common file path manipulations
import dask.dataframe as dd # import of data
import matplotlib.pyplot as plt # plotting
import numpy as np # maths
import pandas as pd # data handling


# Setup of imported libraries

pd.options.mode.chained_assignment = None


# =========================================================================== #
# ========================= User defined parameters ========================= #

# Logical switches.
# If using Spyder, you can avoid needing to load the data every time by going
# to "Run > Configure per file" and activating the option "Run in console's
# namespace instead of an empty one".

load_data = True
sanitize_gps = True # Use satellite number to clean GPS-data during loading?
convert_data = True
process_data = True # Relevant calculations for your scientific case
create_plots = True
show_plots = True
export_plots = True # Note that plots cannot be exported unless created!
export_data = False
export_kml = False


# To save memory and computing power, this script allows users to exclude all
# data before a given time t_0.

t_0 = 135.5 # [s]
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
    'A6 (-)': 'pressure',
    'A5 (-)': 'light',
    'A7 (-)': 'voltage_analogue',
    'A1 (-)': 'temp_ext',
    'A0 (-)': 'temp_int',
    'A3 (-)': 'a_x',
    'A4 (-)': 'a_y',
    'A2 (-)': 'mag',
    }

temp_array_channels = {
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
    'array_voltage (-)': 'voltage',
    'array_current (-)': 'current',
    }

gps_channels = {
    'gps_satellites (-)': 'satellites',
    'gps_long (-)': 'long',
    'gps_lat (-)': 'lat',
    'gps_altitude (-)': 'height',
    'gps_speed (-)': 'speed',
    "lat (deg)": 'la',
    "long (deg)": 'lo',
    "altitude (m)": 'h',
    }

imu_channels = {
    'P0/gps_imu/imu_ax (-)': 'a_x_imu',
    'P0/gps_imu/imu_ay (-)': 'a_y_imu',
    'P0/gps_imu/imu_az (-)': 'a_z_imu',
    'P0/gps_imu/imu_gx (-)': 'ang_vel_x',
    'P0/gps_imu/imu_gy (-)': 'ang_vel_y',
    'P0/gps_imu/imu_gz (-)': 'ang_vel_z',
    'P0/gps_imu/imu_mx (-)': 'mag_x',
    'P0/gps_imu/imu_my (-)': 'mag_y',
    'P0/gps_imu/imu_mz (-)': 'mag_z',
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



# =========================================================================== #
# ============================== Load CSV data ============================== #

if load_data:

    # First a root window is created and put on top of all other windows.

    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)

    # On top of the root window, a filedialog is opened to get the CSV file
    # from the file explorer.

    data_file = "C:/Users/jonah/Downloads/pyIGRF/pyIGRF/release/Raks_tm_readout.csv "

    '''fd.askopenfilename(
        title = 'Select rocket data to import',
        filetypes = (('CSV files','.csv'),('All files','.*')),
        parent = root,
        )
    '''
    print('\nLoading data from t =',t_0,'s to',t_end,'s.')
    print('\nFile path:\n',data_file,'\n')

    # Save some file paths for later.

    parent_file_name, parent_file_extension = path.splitext(
        path.basename(data_file)
        )
    working_directory = ''

    # Use dask to load the file, saving only the lines with t >= t_0 into
    # memory, and only the colons listed in the channels dictionary defined
    # above.

    raw_data = dd.read_csv(
        data_file,
        usecols = channels.keys(),
        sep = ',',
        assume_missing = True,
        encoding = 'utf-8',
        )
    raw_data = raw_data[raw_data['Time (s)'] >= t_0]
    raw_data = raw_data[raw_data['Time (s)'] < t_end]
    raw_data = raw_data.compute()

    # Simplify channel names according to the user-defined dictionary.

    raw_data.rename(
        columns = channels,
        inplace = True,
        )

    # Sanitize GPS-data
    if sanitize_gps:
        mask = raw_data['satellites'] < 3
        mask2 = raw_data['satellites'] == 3
        mask3 = raw_data['speed'] > 1e6 # cm/s
        raw_data.loc[mask, ['lat', 'long', 'height', 'speed']] = np.nan
        raw_data.loc[mask2, ['height', 'speed']] = np.nan
        raw_data.loc[mask3, ['speed']] = np.nan
    
    
    # ======================== De-multiplex data ======================== #
    
    def isolate_dataseries(data_label, unique_time_label=False):
        time_label = 't'
        if unique_time_label:
            time_label += '_' + data_label
        data = pd.DataFrame()
        data[[time_label, data_label]] = raw_data[['t', data_label]].dropna(thresh=2)
        # data = data.dropna(subset=[time_label]) # In case we somehow have a NaN in the time-axis
        # To ease the identification of lost data, we want to store a nan-value
        # between data points where there *should* have been data. In order to
        # identify these points, we need to calculate how often the sensor data
        # should come. The code below is an automated solution; a less error-prone
        # but more manual method would be to calculate the expected measurement
        # frequency from the frame format specification
        dt = min(data[time_label].diff().dropna())
        # No time measurement is exact, but if the interval between each data point
        # is more than 1.5 the minimal update interval, we have missing data
        mask = np.abs(data[time_label].diff() - dt) / dt >= 0.5
        nan_data = data[mask]
        nan_data[time_label] -= dt
        nan_data.index -= 1
        nan_column = np.full(len(nan_data[data_label]), np.nan)
        nan_data[data_label] = nan_column
        data = pd.concat([data, nan_data]).sort_values(by=[time_label])
        data.reset_index(drop=True, inplace=True)
        return data
    
    def isolate_dataframe(frame, unique_time_label=False):
        # Step 1: Create list of separated data series
        # Step 2: Combine the data series into one dataframe
        # Step 3: Profit!
        if unique_time_label:
            dataseries = [isolate_dataseries(x, unique_time_label=True) for x in frame]
            new_frame = pd.concat(dataseries, axis=1, join='inner')
        else:
            new_frame = pd.DataFrame({'t':[]}) # Empty time-frame
            for x in frame:
                if x != 't':
                    dataseries = isolate_dataseries(x)
                    new_frame = pd.merge(new_frame, dataseries, how='outer')
            new_frame = new_frame.sort_values('t', ignore_index=True)
        return new_frame
    
    print('De-multiplexing ...')
    
    analogue = isolate_dataframe(analogue_channels)
    temp_array = isolate_dataframe(temp_array_channels)
    power_sensor = isolate_dataframe(power_sensor_channels)
    gps = isolate_dataframe(gps_channels)
    imu = isolate_dataframe(imu_channels, unique_time_label=True)
    misc = isolate_dataframe(misc_channels)
    

# =========================================================================== #
# ============================== Convert data =============================== #

if convert_data:

    print('Converting data to sensible units ...')

    # Physical constants

    T_0 = 273.15 # 0 celsius degrees in kelvin


    # Other useful constants
    U_main = 5.0
    U_array = 3.3
    wordlength_main = 8
    wordlength_array = 12
    
    R_rel = R_fixed/R_ref


    # Conversion formulas

    def volt(bit_value, wordlength=wordlength_main, U=U_main):
        Z = 2**wordlength - 1
        return U*bit_value/Z
    
    def analogue_voltage(U): # unit: volts
        return U*3.2 # gain taken from the encoder documentation

    def volt_to_pressure(U): # unit: kPa
        return (200*U+95)/9

    def linear_temp(U, gain, offset): # unit: celsius degrees
        return 100*(U/gain-offset)

    def volt_to_acceleration(U, sensitivity, offset): # unit: gee
        return (U-offset)/sensitivity

    def phototransistor(U):
        # Not implemented!
        return U

    def magnetometer(U):
        # Not implemented!
        return U

    def NTC(U): # unit: celsius degrees
        divisor = U_array-U
        divisor[divisor<=0] = np.nan # avoids division by zero
        R = R_rel*U/divisor
        R[R<=0] = np.nan # avoids complex logarithms
        ln_R = np.log(R)
        T = 1/(A_1 + B_1*ln_R + C_1*ln_R**2 + D_1*ln_R**3)
        T -= T_0 # convert to celsius degrees
        return T

    def array_voltage(U, gain): # unit: volts
        return U*gain

    def array_current(U, R_current): # unit: ampere
        return U/(100*R_current)

    def imu_1D(bit_value, sensitivity, offset):
        """

        Parameters
        ----------
        bit_value : int
            Bit value to be converted.

        sensitivity : float
            Sensitivity of the sensor. Typical values can be found in the data sheet, but accurate values need to be determined experimentally.

        offset : float, optional
            Offset in physical units. The default is zero. Should be set to whatever imu_1D gives (with default parameters) when the measurment *should* be zero.

        Returns
        -------
        float
            Converted value(s). The unit depends on the sensor:
                accelerometer: gee, i.e. 9.81 m/s^2
                gyroscope: degrees per second
                magnetometer: gauss

        """
        return sensitivity*bit_value-offset

    def gps_degrees(angle): # unit: degrees
        return angle*1e-7

    def gps_height(height): # unit: meters
        return height*1e-2

    def gps_velocity(velocity): # unit: meters per second
        return velocity*1e-2



    # Convert raw channels to volt
    
    for x in analogue_channels:
        analogue[x] = volt(analogue[x])
    
    for x in power_sensor_channels:
        power_sensor[x] = volt(
            power_sensor[x],
            wordlength = wordlength_array,
            U = U_array,
            )

    for x in temp_array_channels:
        temp_array[x] = volt(
            temp_array[x],
            wordlength = wordlength_array,
            U = U_array,
            )



    # Convert data to physical units

    analogue['pressure'] = volt_to_pressure(analogue['pressure'])
    analogue['a_x'] = volt_to_acceleration(
        analogue['a_x'],
        a_x_sens,
        a_x_offset,
        )
    analogue['a_y'] = volt_to_acceleration(
        analogue['a_y'],
        a_y_sens,
        a_y_offset,
        )
    analogue['temp_int'] = linear_temp(
        analogue['temp_int'],
        temp_int_gain,
        temp_ext_offset,
        )
    analogue['temp_ext'] = linear_temp(
        analogue['temp_ext'],
        temp_ext_gain,
        temp_int_offset,
        )
    analogue['light'] = phototransistor(analogue['light'])
    analogue['mag'] = magnetometer(analogue['mag'])
    analogue['voltage_analogue'] = analogue_voltage(analogue['voltage_analogue'])
   
    
    power_sensor['voltage'] = array_voltage(
        power_sensor['voltage'],
        voltage_sensor_gain,
        )
    power_sensor['current'] = array_current(
        power_sensor['current'],
        R_current_sensor,
        )


    gps['lat'] = gps_degrees(gps['lat'])
    gps['long'] = gps_degrees(gps['long'])
    gps['height'] = gps_height(gps['height'])
    gps['speed'] = gps_velocity(gps['speed'])


    imu['a_x_imu'] = imu_1D(imu['a_x_imu'], a_x_imu_sens, a_x_imu_offset)
    imu['a_y_imu'] = imu_1D(imu['a_y_imu'], a_y_imu_sens, a_y_imu_offset)
    imu['a_z_imu'] = imu_1D(imu['a_z_imu'], a_z_imu_sens, a_z_imu_offset)

    imu['ang_vel_x'] = imu_1D(imu['ang_vel_x'], ang_vel_x_sens, ang_vel_x_offset)
    imu['ang_vel_y'] = imu_1D(imu['ang_vel_y'], ang_vel_y_sens, ang_vel_y_offset)
    imu['ang_vel_z'] = imu_1D(imu['ang_vel_z'], ang_vel_z_sens, ang_vel_z_offset)

    imu['mag_x'] = imu_1D(imu['mag_x'], mag_x_sens, mag_x_offset)
    imu['mag_y'] = imu_1D(imu['mag_y'], mag_y_sens, mag_y_offset)
    imu['mag_z'] = imu_1D(imu['mag_z'], mag_z_sens, mag_z_offset)


    for x in temp_array_channels:
        temp_array[x] = NTC(temp_array[x])


# =========================================================================== #
# ============================= Processes data ============================== #

if process_data:

    print('Calculating useful stuff ...')

    # smoothing function

    def smooth(data, r_tol=0.02):
        """
        Smooths a data series by requiring that the change between one datapoint and the next is below a certain relative treshold (the default is 2 % of the total range of the values in the dataseries). If this is not the case, the value is replaced by NaN. This gives a 'rougher' output than a more sophisticated smoothing algorithm (see for example statsmodels.nonparametric.filterers_lowess.lowess from the statmodels module), but has the advantage of being very quick. If more sophisticated methods are needed, this algorithm can be used to trow out obviously erroneous data before the data is sent through a more traditional smoothing algorithm.

        Parameters
        ----------
        data : pandas.DataSeries
            The data series to be smoothed

        r_tol : float, optional
            Tolerated change between one datapoint and the next, relative to the full range of values in DATA. The default is 0.02.

        Returns
        -------
        data_smooth : pandas.DataSeries
            smoothed data series.

        """
        valid_data = data[data.notna()]
        if len(valid_data) == 0:
            data_range = np.inf
        else:
            data_range = np.ptp(valid_data)
        tol = r_tol*data_range
        data_smooth = data.copy()
        data_interpol = data_smooth.interpolate()
        data_smooth[np.abs(data_interpol.diff()) > tol] = np.nan
        data_smooth = data_smooth.interpolate()
        return data_smooth


    times = list(analogue[["t", "mag"]].dropna()["t"])
    vals = list(analogue[["t", "mag"]].dropna()["mag"])

    diffs = [times[i] - times[i-1] for i in range(1, len(times))]
    avg = sum(diffs)/len(diffs)

    for i in range(0, len(times)):
        if i == len(times)-1:
            break
        diff = times[i+1] - times[i]
        if diff > 1.5 * avg:
            times.insert(i+1, times[i] + diff/2)
            vals.insert(i+1, (vals[i] + vals[i+1])/2)

    



    import matplotlib.pyplot as plt
    t = np.arange(256)
    sp = np.fft.fft(np.sin(t))
    freq = np.fft.fftfreq(t.shape[-1])
    plt.plot(freq, sp.real, freq, sp.imag)
    plt.show()


    print()



    #imu["mag_mag"] = [np.sqrt(imu["imu_mx"][i]**2 + imu["imu_my"][i]**2 + imu["imu_mz"][i]**2) for i, v in imu.iterrows()]

    np.fft.fftfreq()
    np.fft.helper


    print()

    '''
    imusub = imu.iloc[::5, :][["t_imu_mx", "imu_mx", "imu_my", "imu_mz"]].reset_index()
    gpssub = gps[["t", "la", "lo", "h"]]
    magdata = imusub.merge(gpssub, left_index=True, right_index=True)

    magdata.drop(columns=["index"], inplace=True)

    

    .to_csv("mag_data.csv", index=False)

    '''

    '''
    imu["imu_ac"] = [gx**2*0.015 for gx in imu["imu_gx"]]
    imu["imu_ay_cor"] = [imu["imu_ay"][i] - imu["imu_ac"][i] for i, v in imu.iterrows()]
    '''

    '''

    df = imu[["t_imu_gx", "imu_gx"]].dropna(0).reset_index()

    thetas = []
    times = []
    theta = 3 * np.pi / 2
    for i in range(len(df)):
        thetas.append(theta)
        times.append(df["t_imu_gx"][i])
        dt = 0
        if i > 0:
            dt = df["t_imu_gx"][i] - df["t_imu_gx"][i-1]
        else:
            print("nay")
            dt = 0.09
        dtheta = df["imu_gx"][i] * dt
        theta = (theta + dtheta) % (2 * np.pi)

    df["t"] = times
    df["th"] = thetas


    print()

    '''

    '''
    [row for i, row in imu.iterrows() if round(row["t_imu_gx"], 2) == round(172.483, 2)]
    [row for i, row in imu.iterrows() if round(row["t_imu_ay"], 2) == round(172.483)]

    [r for r in [r[1] for r in imu.iterrows()][-1100:-1000] if r["imu_ay"] == min(imu[-1100:-1000]["imu_ay"])]
    [row for i, row in imu.iterrows() if round(row["t_imu_ax"], 2) == 172.48]

    imu["imu_thetax"] = [sum([(imu["t_imu_gx"][i]-imu["t_imu_gx"][max(i-1, 0)])*v for i, v in enumerate([imu["imu_gx"][max(i - 1, 0)] if imu["imu_gx"][max(i - 1, 0)] == imu["imu_gx"][max(i - 1, 0)] else imu["imu_gx"][max(i - 2, 0)] for i, t in enumerate(imu["t_imu_ax"])])][0:j])%2*np.pi for j in range(len(imu))]
    print()
    '''



# =========================================================================== #
# =========================== Prepare for export ============================ #

export = export_data or export_kml or export_plots

if export and working_directory == '':
    working_directory = fd.askdirectory(
        title = 'Choose output folder',
        parent = root
        )
    plot_directory = path.join(working_directory, 'Plots')
    data_directory = path.join(working_directory, 'Processed data')


# =========================================================================== #
# ================================ Plot data ================================ #

if create_plots:

    print('Plotting ...')

    plt.ioff() # Prevent figures from showing unless calling plt.show()
    plt.style.use('seaborn') # plotting style.
    plt.rcParams['legend.frameon'] = 'True' # Fill the background of legends.

    if export_plots and not path.exists(plot_directory):
        os.mkdir(plot_directory)
    
    # ==================== Custom plotting functions ==================== #

    # Custom parameters

    standard_linewidth = 0.5


    # First some auxillary functions containing some often-needed lines of
    # code for custom plots.


    # Standard settings for plt.figure()
    # Create a figure, or ready an already existing figure for new data.
    # Returns the window title for easier export with finalize_figure().

    def create_figure(name):
        name = name + ' [' + parent_file_name + ']'
        plt.figure(name, clear=True)
        return name


    # Standard plotting function.

    def plot_data(x, y, data=analogue, data_set=''):
        x_smooth = x + '_smooth'
        y_smooth = y + '_smooth'
        if x_smooth in data.columns:
            if data_set=='':
                data_set = 'compare'
        else:
            x_smooth = x
        if y_smooth in data.columns:
            if data_set=='':
                data_set = 'compare'
        else:
            y_smooth = y
            
        if data_set=='compare':
            raw, = plt.plot(
                data[x],
                data[y],
                'r-',
                linewidth = standard_linewidth,
                )
            smoothed, = plt.plot(
                data[x_smooth],
                data[y_smooth],
                'b-',
                linewidth = standard_linewidth,
                )
            plots = [smoothed, raw]
            labels = ['Smoothed data', 'Raw data']
            
        elif data_set=='raw_only' or data_set=='':
            plots, = plt.plot(
                data[x],
                data[y],
                'b-',
                linewidth = standard_linewidth,
                )
            labels = 'Raw data'
            
        elif data_set=='smooth_only':
            plots, = plt.plot(
                data[x_smooth],
                data[y_smooth],
                'b-',
                linewidth = standard_linewidth,
                )
            labels = 'Smoothed data'
            
        else:
            print('The given data_set argument is invalid.')
            
        return plots, labels


    # Create legend. Standard settings for plt.legend()

    def make_legend(plots, plot_labels):
        plt.legend(
            plots,
            plot_labels,
            facecolor = 'white',
            framealpha = 1
            )


    # Standard layout, and if-statements to show and/or export the figure as
    # necessary.

    def finalize_figure(figure_name):
        plt.tight_layout()
        if export_plots:
            file_formats = ['png', 'pdf'] # pdf needed for vector graphics.
            for ext in file_formats:
                file_name = figure_name + '.' + ext
                file_name = path.join(plot_directory, file_name)
                plt.savefig(
                    file_name,
                    format = ext,
                    dpi = 600
                    )
        if show_plots:
            plt.draw()
            plt.show()


    # This is a single function providing a simple interface for standard
    # graphs, as well as serving as an example of how the auxillary functions
    # above might be utilized.

    def plot_graph(figure_name, x, y, x_label, y_label, data_bank=''):
        figure_name = create_figure(figure_name)
        data_plots, data_labels = plot_data(
            x,
            y,
            data = data_bank
            )
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        make_legend(data_plots, data_labels)
        finalize_figure(figure_name)


    # analog magnetometer
    figure_name = create_figure('Magnetic field (y)')
    plot_data(
        't',
        'mag',
        data_set="raw_only"
        )
    plt.ylim(0, U_main)
    plt.xlabel('$t$ [s]')
    plt.ylabel('$M_y$ [V]')
    finalize_figure(figure_name)

    # imu gx
    figure_name = create_figure('Gyro Rotation (x)')
    plot_data(
        't_imu_gx',
        'imu_gx',
        data=imu,
        data_set="raw_only"
    )
    #plt.ylim(0, U_main)
    plt.xlabel('$t$ [s]')
    plt.ylabel('$g_y$ [rad/s]')
    finalize_figure(figure_name)

    # imu m magnitude
    figure_name = create_figure('Magnetic Field Magnitude')
    plot_data(
        't_imu_mx',
        'mag_mag',
        data=imu,
        data_set="raw_only"
    )
    # plt.ylim(0, U_main)
    plt.xlabel('$t$ [s]')
    plt.ylabel('$m$ [T]')
    finalize_figure(figure_name)

    # imu mag components
    figure_name = create_figure('Magnetic Field components')
    make_legend(plot_data(
        't_imu_mx',
        'imu_mx',
        data=imu,
        data_set="raw_only"
    ))
    make_legend(plot_data(
        't_imu_my',
        'imu_my',
        data=imu,
        data_set="raw_only"
    ))
    make_legend(plot_data(
        't_imu_mz',
        'imu_mz',
        data=imu,
        data_set="raw_only"
    ))
    # plt.ylim(0, U_main)
    plt.xlabel('$t$ [s]')
    plt.ylabel('$m$ [T]')

    finalize_figure(figure_name)


    # ========================= Specific plots ========================== #

    

    # ======================== Analogue sensors ========================= #
    
    # Pressure

    '''
    figure_name = create_figure('Pressure')
    plots, labels = plot_data(
        't',
        'pressure',
        )
    plt.xlabel('$t$ [s]')
    plt.ylabel('Pressure [kPa]')
    make_legend(plots, labels)
    finalize_figure(figure_name)
    
    
    # Magnetic field (y)
       

    # Acceleration (y)
    
    figure_name = create_figure('Acceleration (y)')
    plots, labels = plot_data(
        't',
        'a_y',
        )
    plt.ylim(-a_y_max, a_y_max)
    plt.xlabel('$t$ [s]')
    plt.ylabel('$a_y$ [gee]')
    make_legend(plots, labels)
    finalize_figure(figure_name)
    
    
    # Acceleration (x)
    
    figure_name = create_figure('Acceleration (x)')
    plots, labels = plot_data(
        't',
        'a_x',
        )
    plt.ylim(-a_x_max, a_x_max)
    plt.xlabel('$t$ [s]')
    plt.ylabel('$a_x$ [gee]')
    make_legend(plots, labels)
    finalize_figure(figure_name)
    
    
    # Light sensor
    
    figure_name = create_figure('Light sensor')
    plot_data(
        't',
        'light',
        )
    plt.ylim(0, U_main)
    plt.xlabel('$t$ [s]')
    plt.ylabel('Brightness [V]')
    finalize_figure(figure_name)


    # Internal temperature

    figure_name = create_figure('Internal temperature')
    plots, labels = plot_data(
        't',
        'temp_int',
        )
    plt.xlabel('$t$ [s]')
    plt.ylabel(u'Temperature [\N{DEGREE SIGN}C]')
    make_legend(plots, labels)
    finalize_figure(figure_name)


    # External temperature

    figure_name = create_figure('External temperature')
    plots, labels = plot_data(
        't',
        'temp_ext',
        )
    plt.xlabel('$t$ [s]')
    plt.ylabel(u'Temperature [\N{DEGREE SIGN}C]')
    make_legend(plots, labels)
    finalize_figure(figure_name)



    # ======================== Temperature array ======================== #
    
    # Fancy plot

    plt.rcParams['axes.grid'] = False # disables the grid. Remember to turn it back on again after this figure!
    background_color = '#EAEAF2' # same color as the seaborn-theme. Not the most elegant solution
    figure_name = 'Temperature array'
    fig = plt.figure(
        figure_name,
        facecolor = background_color,
        clear = True,
        )
    ax = plt.axes(facecolor = background_color)
    sensor_IDs = np.arange(10)
    im = ax.pcolormesh(
        temp_array['t'],
        sensor_IDs,
        temp_array[temp_array_channels].T,
        shading = 'nearest',
        cmap = 'hot',
        )
    plt.xlabel('$t$ [s]')
    plt.ylabel('Sensor #')
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.set_title(u'\N{DEGREE SIGN}C')
    finalize_figure(figure_name)
    plt.rcParams['axes.grid'] = True # enables the grid again for later plots


    # Sanity check of fancy plot (used to verify data integrity)
    
    figure_name = create_figure('Temperature array (sanity check)')
    x = temp_array['t']
    for i in temp_array_channels:
        y = temp_array[i]
        plt.plot(x, y, label=i)
    plt.xlabel('$t$ [s]')
    plt.ylabel(u'Temperature [\N{DEGREE SIGN}C]')
    plt.legend(
        facecolor = 'white',
        framealpha = 1,
        )
    finalize_figure(figure_name)
  
    
    
    # ========================== Power sensor =========================== #
    
    # Battery voltage

    figure_name = create_figure('Battery voltage')
    analogue_plot, = plt.plot(
        analogue['t'],
        analogue['voltage_analogue'],
        '-',
        linewidth = standard_linewidth,
        )
    digital_plot, = plt.plot(
        power_sensor['t'],
        power_sensor['voltage'],
        '-',
        linewidth = standard_linewidth,
        )
    plots = [analogue_plot, digital_plot]
    labels = ['analogue', 'digital']
    plt.xlabel('$t$ [s]')
    plt.ylabel('[V]')
    make_legend(plots, labels)
    finalize_figure(figure_name)
    
    
    # Current

    figure_name = create_figure('Current')
    plt.plot(
        power_sensor['t'],
        power_sensor['current'],
        '-',
        linewidth = standard_linewidth,
        )
    plt.xlabel('$t$ [s]')
    plt.ylabel('Current [A]')
    finalize_figure(figure_name)
    
    
    
    # =============================== GPS =============================== #
    
    # Speed (from GPS)

    figure_name = create_figure('GPS speed')
    plot_data(
        't',
        'speed',
        data = gps,
        )
    plt.xlabel('$t$ [s]')
    plt.ylabel('Speed [m/s]')
    finalize_figure(figure_name)
    
    
    # Height (from GPS)

    figure_name = create_figure('GPS height (smoothed)')
    plot_data(
        't',
        'height',
        data = gps,
        data_set = 'smooth_only',
        )
    plt.xlabel('$t$ [s]')
    plt.ylabel('Altitude [m]')
    finalize_figure(figure_name)
    
    
    
    # =============================== IMU =============================== #
    
    # The IMU is a bit special, in that each channel is stored at different
    # time slots. Hence, we need to use unique time-variables for each channel.
    
    # IMU magnetometer
    
    figure_name = create_figure('IMU mag')
    x_plot, = plt.plot(
        imu['t_mag_x'],
        imu['mag_x'],
        '-',
        linewidth = standard_linewidth,
        )
    y_plot, = plt.plot(
        imu['t_mag_y'],
        imu['mag_y'],
        '-',
        linewidth = standard_linewidth,
        )
    z_plot, = plt.plot(
        imu['t_mag_z'],
        imu['mag_z'],
        '-',
        linewidth = standard_linewidth,
        )
    plots = [x_plot, y_plot, z_plot]
    labels = ['$M_x$', '$M_y$', '$M_z$']
    plt.xlabel('$t$ [s]')
    plt.ylabel('Magnetic fieldstrength [gauss]')
    make_legend(plots, labels)
    finalize_figure(figure_name)
    
    
    # IMU gyroscope
    
    figure_name = create_figure('IMU gyro')
    x_plot, = plt.plot(
        imu['t_ang_vel_x'],
        imu['ang_vel_x'],
        '-',
        linewidth = standard_linewidth,
        )
    y_plot, = plt.plot(
        imu['t_ang_vel_y'],
        imu['ang_vel_y'],
        '-',
        linewidth = standard_linewidth,
        )
    z_plot, = plt.plot(
        imu['t_ang_vel_z'],
        imu['ang_vel_z'],
        '-',
        linewidth = standard_linewidth,
        )
    plots = [x_plot, y_plot, z_plot]
    labels = ['$x$', '$y$', '$z$']
    plt.xlabel('$t$ [s]')
    plt.ylabel('Angular velocity [degrees per second]')
    make_legend(plots, labels)
    finalize_figure(figure_name)


    # IMU accelerometer
    
    figure_name = create_figure('IMU accelerometer')
    x_plot, = plt.plot(
        imu['t_a_x_imu'],
        imu['a_x_imu'],
        '-',
        linewidth = standard_linewidth,
        )
    y_plot, = plt.plot(
        imu['t_a_y_imu'],
        imu['a_y_imu'],
        '-',
        linewidth = standard_linewidth,
        )
    z_plot, = plt.plot(
        imu['t_a_z_imu'],
        imu['a_z_imu'],
        '-',
        linewidth = standard_linewidth,
        )
    plots = [x_plot, y_plot, z_plot]
    labels = ['$a_x$', '$a_y$', '$a_z$']
    plt.xlabel('$t$ [s]')
    plt.ylabel('Acceleration [gee]')
    make_legend(plots, labels)
    finalize_figure(figure_name)
    
    
    
    # ========================== Miscellaneous ========================== #
    
    # Frame counter

    figure_name = create_figure('Frame counter')
    plot_data(
        't',
        'framecounter',
        data = misc,
        )
    plt.xlabel('$t$ [s]')
    plt.ylabel('Frame number')
    finalize_figure(figure_name)
    '''


# =========================================================================== #
# ========================== Export processed data ========================== #

# Much like before, a filedialog is opened, this time to allow the user to
# specify the name and storage location of the processed data.
# The processed data is stored using Pandas' .to_csv()

if export_data:
    print('Exporting ...')
    
    if not path.exists(data_directory):
        os.mkdir(data_directory)
    
    def export_frame(frame, frame_name):
        data_file = parent_file_name + '_' + frame_name + '.csv'
        data_file = path.join(data_directory, data_file)
        frame.to_csv(data_file, sep = ';', decimal = ',', index = False)

    export_frame(analogue, 'analogue')
    export_frame(gps, 'GPS')
    export_frame(imu, 'IMU')
    export_frame(temp_array, 'temp_array')
    export_frame(power_sensor, 'power_sensor')
    export_frame(misc, 'misc')


# Create and export a kml-file which can be opened in Google Earth.

if export_kml:
    print('Exporting kml ...')

    # kml-coordinates needs to be in degrees for longitude and latitude, and
    # meters for the height. Hence, we will take our data from processed_data:
    notna_indices = (
        gps['lat_smooth'].notna() &
        gps['long_smooth'].notna() &
        gps['height_smooth'].notna()
        )
    kml_lat = gps['lat_smooth'][notna_indices].copy().to_numpy()
    kml_long = gps['long_smooth'][notna_indices].copy().to_numpy()
    kml_height = gps['height_smooth'][notna_indices].copy().to_numpy()

    # To avoid having to install a kml-library, we will instead (ab)use a numpy
    # array and savetxt()-function to save our kml file.
    # Unfortunately, this means that we need to hard-code the kml-file ...
    kml_header = (
'''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://earth.google.com/kml/2.2">
<Document>
<name>Paths</name>
<description>Paths based on GPS/GNSS coordinates.</description>
<Style id="yellowLineGreenPoly">
<LineStyle>
<color>7f00ffff</color>
<width>4</width>
</LineStyle>
<PolyStyle>
<color>7f00ff00</color>
</PolyStyle>
</Style>
<Placemark>
<name>Student rocket path</name>
<description>Student rocket path, according to its onboard GPS</description>
<styleUrl>#yellowLineGreenPoly</styleUrl>
<LineString>
<extrude>0</extrude>
<tessellate>0</tessellate>
<altitudeMode>absolute</altitudeMode>
<coordinates>''')

    kml_body = np.array([kml_long, kml_lat, kml_height]).transpose()

    kml_footer = (
'''</coordinates>
</LineString>
</Placemark>
</Document>
</kml>''')

    data_file = parent_file_name + '.kml'
    data_file = path.join(working_directory, data_file)

    np.savetxt(
        data_file,
        kml_body,
        fmt = '%.6f',
        delimiter = ',',
        header = kml_header,
        footer = kml_footer,
        comments = '',
        )
