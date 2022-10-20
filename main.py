import datetime

import numpy as np
import pandas
import pandas as pd
import scipy
import matplotlib.pyplot as plt

#import utilities
import tkinter as tk


root = tk.Tk()
root.withdraw()
root.wm_attributes('-topmost', 1)


def read(name: str) -> pandas.DataFrame:
    file_name = f"{root}\Processed Data\Raks_tm_readout_{name}.csv"
    data = pandas.read_csv(file_name)
    return data


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
    tol = r_tol * data_range
    data_smooth = data.copy()
    data_interpol = data_smooth.interpolate()
    data_smooth[np.abs(data_interpol.diff()) > tol] = np.nan
    data_smooth = data_smooth.interpolate()
    return data_smooth


def calc_ang_vel_mag_fft():
    SP = 128

    imu_t = read("imu")["t_imu_gx"]
    mag = read("analogue")[["t", "mag"]][::4].reset_index()

    mag_t = mag["t"]
    mag_v = smooth(mag["mag"])

    # Rebase mag data equilibrium at zero
    avg_mag = sum(mag_v)/len(mag_v)
    mag_v = mag_v.map(lambda x: x - avg_mag)

    mag_freq = pd.Series()

    #plt.plot(mag_t, mag_v)
    #plt.show()

    for t in imu_t[250:600]:
        print(datetime.datetime.now().time())

        i = next(i for i, v in enumerate(mag_t) if v >= t)

        min_i = max(0, int(i-SP/2))
        max_i = min(len(mag_v), int(i+SP/2))

        times = mag_t[min_i:max_i]
        points = mag_v[min_i:max_i]

        coeffs = np.abs(np.fft.fft(points))
        freq_num = coeffs.argmax()

        spacing = (mag_t[max_i] - mag_t[min_i])/SP
        freqs = np.fft.fftfreq(SP, d=spacing)
        freq = freqs[freq_num]*2*np.pi

        plt.plot(times, points)
        plt.show()

        print()
        mag_freq.append(freq)


def calc_ang_vel_mag_fit():
    SP = 512

    def fun(x, A, w, p):
        return A * np.sin(w * x**2 - p)

    imu_t = read("imu")["t_imu_gx"]
    mag = read("analogue")[["t", "mag"]][::1].reset_index()

    mag_t = mag["t"]
    mag_v = smooth(mag["mag"])

    # Rebase mag data equilibrium at zero
    avg_mag = sum(mag_v)/len(mag_v)
    mag_v = mag_v.map(lambda x: x - avg_mag)

    #mag_freq = pd.Series()

    for t in imu_t[250:600]:
        print(datetime.datetime.now().time())

        i = next(i for i, v in enumerate(mag_t) if v >= t)

        min_i = max(0, int(i-SP/2))
        max_i = min(len(mag_v), int(i+SP/2))

        times = mag_t[min_i:max_i]
        values = mag_v[min_i:max_i]

        test = scipy.optimize.curve_fit(fun, times, values)[0]

        plt.plot(times, values)
        plt.show()

        print()
        print()


def calc_ang_vel_mag_zer():
    SP = 512

    def fun(x, A, w, p):
        return A * np.sin(w * x**2 - p)

    imu_t = read("imu")["t_imu_gx"]
    mag = read("analogue")[["t", "mag"]][::1].reset_index()

    mag_t = mag["t"]
    mag_v = smooth(mag["mag"])

    # Rebase mag data equilibrium at zero
    avg_mag = sum(mag_v)/len(mag_v)
    mag_v = mag_v.map(lambda x: x - avg_mag)

    #mag_freq = pd.Series()

    for t in imu_t[250:600]:
        print(datetime.datetime.now().time())

        i = next(i for i, v in enumerate(mag_t) if v >= t)

        min_i = max(0, int(i-SP/2))
        max_i = min(len(mag_v), int(i+SP/2))

        times = mag_t[min_i:max_i]
        values = mag_v[min_i:max_i]

        test = scipy.optimize.curve_fit(fun, times, values)[0]

        plt.plot(times, values)
        plt.show()

        print()
        print()


def test():
    imu = read("imu")

    t = imu["t_imu_ay"]
    y = imu["imu_ay"]
    z = imu["imu_az"]

    plt.plot(t, y)
    plt.plot(t, z)
    plt.show()


def test_2():
    analogue = read("analogue")

    t = analogue["t"]
    y = smooth(analogue["photometer"])
    x = smooth(analogue["ax"])

    plt.plot(t, y)
    #plt.plot(t, x)
    plt.show()


def main():
    test_2()




if __name__ == "__main__":
    main()