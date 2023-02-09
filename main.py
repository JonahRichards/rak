import datetime

import numpy as np
import pandas
import pandas as pd
import scipy
from scipy.signal import savgol_filter
from scipy.signal import argrelextrema
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# import utilities
import tkinter as tk

root = tk.Tk()
root.withdraw()
root.wm_attributes('-topmost', 1)


def read(name: str) -> pandas.DataFrame:
    file_name = f"{root}\Processed Data\Raks_tm_readout_{name}.csv"
    data = pandas.read_csv(file_name)
    return data


def smooth(data, r_tol=0.005):
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
    avg_mag = sum(mag_v) / len(mag_v)
    mag_v = mag_v.map(lambda x: x - avg_mag)

    mag_freq = pd.Series()

    # plt.plot(mag_t, mag_v)
    # plt.show()

    for t in imu_t[250:600]:
        print(datetime.datetime.now().time())

        i = next(i for i, v in enumerate(mag_t) if v >= t)

        min_i = max(0, int(i - SP / 2))
        max_i = min(len(mag_v), int(i + SP / 2))

        times = mag_t[min_i:max_i]
        points = mag_v[min_i:max_i]

        coeffs = np.abs(np.fft.fft(points))
        freq_num = coeffs.argmax()

        spacing = (mag_t[max_i] - mag_t[min_i]) / SP
        freqs = np.fft.fftfreq(SP, d=spacing)
        freq = freqs[freq_num] * 2 * np.pi

        plt.plot(times, points)
        plt.show()

        print()
        mag_freq.append(freq)


def calc_ang_vel_mag_fit():
    # APPROXIMATES A MODEL THE MAG SIGNAL SHOULD OBEY OVER SMALL RANGES
    def model(x, a, b, c, d, e):
        return np.sin(a / 2 * x ** 2 + b) * (c * x + d) + e

    # READ IN ANALOGUE DATA AND IMU TIMEPOINTS TO FIND ANG VEL FOR
    mag = read("analogue")[["t", "mag"]][::1].reset_index()
    imu = read("imu")[["t_imu_gx", "imu_gx", "t_imu_mx"]][::1].reset_index()

    imu_t = imu["t_imu_mx"]
    mag_t = mag["t"]
    mag_v = smooth(mag["mag"])

    # REBASE MAG DATA AT ZERO. NOT EXACT. THERE REMAINS A NON-NEGLIGIBLE AND TIME VARYING OFFSET
    mag_v = mag_v.map(lambda x: x - 2.50980401039124)

    # FIND ALL INDICES WHERE MAG READS ZERO
    zero_indices = []
    zero_toggle = False
    min_i = 0
    max_i = 0
    jump = 0
    for i, v in enumerate(mag_v):
        if jump > 0:                # Avoids double counting by skipping ahead a bit when a zero is found
            jump -= 1
            continue
        if not zero_toggle:         # If no zero has been found yet, look for one
            if abs(v) <= 0.02:
                min_i = i
                zero_toggle = True
        else:                       # If zero has been found, continue until mag doesn't read zero
            if v != 0.0:
                max_i = i - 1       # When mag reads non zero, save avg index of zero interval
                zero_toggle = False
                zero_indices.append(int(np.floor((max_i + min_i) / 2)))  #
                jump = 50

    zero_indices = zero_indices[::2]

    times = [mag_t[i] for i in zero_indices]
    '''
    # PLOT COMPUTED ZERO POINTS ON THE MAG DATA
    plt.plot(mag_t, mag_v)
    plt.plot(times, [0]*len(times), 'ro')
    plt.show()
    '''

    # GENERATE DICT OF APPROXIMATE FREQUENCIES FOR DIFFERENT TIMES. TO BE USED TO GENERATE BOUNDS FOR REGRESSION
    approx_freq = {}
    for i in range(1, len(zero_indices)):
        start = mag_t[zero_indices[i-1]]
        end = mag_t[zero_indices[i]]

        period = (end - start)
        freq = 2 * np.pi / period

        approx_freq[(start + end) / 2] = freq

    imu_g_t = imu["t_imu_gx"]
    imu_g = imu["imu_gx"]

    # PLOT THIS ESTIMATE AGAINST THE IMU ANG VEL
    plt.plot(imu_g_t, smooth(imu_g))
    plt.plot(approx_freq.keys(), smooth(pd.Series(approx_freq.values())))
    plt.legend(["IMU Angular Velocity", "Magnetometer Computed Angular Velocity"])
    plt.show()

    # COMPUTE ANG VEL FOR EACH IMU TIME POINT
    for t in imu_t[700:]:
        print(datetime.datetime.now().time())

        i = next(i for i, v in enumerate(mag_t) if v >= t)
        max_i = mag_t.tolist().index(float(next(mag_t[i] for i in zero_indices if mag_t[i] >= t)))
        min_i = zero_indices[[mag_t[i] for i in zero_indices].index(mag_t[max_i])-1]

        times = mag_t[min_i:max_i]
        values = mag_v[min_i:max_i]

        # times = times.map(lambda x: x - t)

        a, b, c, d, e = scipy.optimize.curve_fit(f=model,
                                                 xdata=times,
                                                 ydata=values,
                                                 p0=[0, 30, 0, max(values), 0],
                                                 # bounds=([], []),
                                                 check_finite=True)[0]

        f = a * t + b

        plt.plot(times, values)
        plt.plot(times, model(times, a, b, c, d, e))
        plt.show()

        print()
        print()


def calc_ang_vel_mag_zer():
    SP = 512

    def fun(x, A, w, p):
        return A * np.sin(w * x ** 2 - p)

    imu_t = read("imu")["t_imu_gx"]
    mag = read("analogue")[["t", "mag"]][::1].reset_index()

    mag_t = mag["t"]
    mag_v = smooth(mag["mag"])

    # Rebase mag data equilibrium at zero
    avg_mag = sum(mag_v) / len(mag_v)
    mag_v = mag_v.map(lambda x: x - avg_mag)

    # mag_freq = pd.Series()

    for t in imu_t[250:600]:
        print(datetime.datetime.now().time())

        i = next(i for i, v in enumerate(mag_t) if v >= t)

        min_i = max(0, int(i - SP / 2))
        max_i = min(len(mag_v), int(i + SP / 2))

        times = mag_t[min_i:max_i]
        values = mag_v[min_i:max_i]

        test = scipy.optimize.curve_fit(fun, times, values)[0]

        plt.plot(times, values)
        plt.show()

        print()
        print()


def calc_acc(ax: str, mi=3950, ma=4650, plot=False) -> dict[str, list]:
    results = {}

    imu = read("imu")

    # RAW DATA
    t = imu[f"t_imu_g{ax}"].tolist()
    raw = list(smooth(imu[f"imu_g{ax}"]))

    if plot:
        plt.plot(t[mi:ma], raw[mi:ma], 'r', label="Y-Accelerometer")
        plt.title("Raw Data")
        plt.legend()
        plt.show()

    # REDUCE NOISE
    red = list(savgol_filter(raw, 30, 2))

    if plot:
        plt.plot(t[mi:ma], red[mi:ma], 'r', label="Y-Accelerometer")
        plt.title("Noise Reduced Data")
        plt.legend()
        plt.show()

    # DETERMINE EQUILIBRIUM
    def model(x, a, b):
        return a * x + b

    S = 25

    eql = []

    for i in range(mi, ma):
        a, b = scipy.optimize.curve_fit(f=model, xdata=t[i-S:i+S], ydata=red[i-S:i+S], check_finite=True)[0]
        eql.append(model(t[i], a, b))

    if plot:
        plt.plot(t[mi:ma], red[mi:ma], 'r', label="Y-Accelerometer")
        plt.plot(t[mi:ma], eql, 'b', label="Estimated Equilibrium Position")
        plt.title("Data with Equilibrium of Oscillation")
        plt.legend()
        plt.show()

    red = red[mi:ma]
    osc = [red[i] - eql[i] for i in range(len(red))]

    if plot:
        plt.plot(t[mi:ma], osc, 'r', label="Force of Gravity")
        plt.title("Isolated Oscillatory Component (Gravity)")
        plt.legend()
        plt.show()

    osc_red = list(savgol_filter(osc, 30, 2))

    mii = list(argrelextrema(np.array(osc_red), np.greater)[0])
    mai = list(argrelextrema(np.array(osc_red), np.less)[0])
    mm = mii + mai
    mm.sort()
    mmv = [osc_red[i] for i in mm]

    if plot:
        plt.plot(t[mi:ma], osc_red, 'r', label="Force of Gravity")
        plt.plot([t[mi + i] for i in mm], mmv, 'bo')
        plt.title("Noise Reduced Oscillatory Component (Gravity)")
        plt.legend()
        plt.show()

    mmv = [abs(v) for v in mmv]
    env = list(interp1d(mm, mmv, kind="cubic")(range(mm[0], mm[-1])))

    if plot:
        plt.plot(t[mi:ma], osc_red, 'r', label="Force of Gravity")
        plt.plot(t[mi + mm[0]:mi + mm[-1]], env, "b", label="Envelope")
        plt.plot(t[mi + mm[0]:mi + mm[-1]], [v * -1 for v in env], "b")
        plt.legend()
        plt.title("Interpolated Envelope for Signal")
        plt.show()

    osc_red = osc_red[mm[0]:mm[-1]]
    nor = [osc_red[i] / env[i] for i in range(len(osc_red))]

    if plot:
        plt.plot(t[mi+ mm[0]:mi + mm[-1]], nor, 'r', label="Coefficient of Force of Gravity")
        plt.title("Normalized Signal")
        plt.legend()
        plt.show()

    match ax:
        case "y":
            th = [np.arcsin(v) * -1 if abs(v) <= 1 else np.nan for v in nor]
        case _:
            th = [np.arccos(v) * -1 if abs(v) <= 1 else np.nan for v in nor]

    zr = min(mii[0], mai[0])
    mi = mi + zr
    ma = mi + max(mii[-1], mai[-1]) - zr
    mii = [v - zr for v in mii]
    mai = [v - zr for v in mai]

    for mai_i in mai:
        try:
            mii_i = next(v for v in mii if v > mai_i)
        except StopIteration:
            continue
        for i in range(mai_i, mii_i):
            th[i] *= -1
            th[i] += np.pi

    for i in range(len(th)):
        if th[i] < 0:
            th[i] += 2 * np.pi

    results["t"] = t[mi:ma]
    results["raw"] = raw[mi:ma]
    results["red"] = red[mm[0]:mm[-1]]
    results["eql"] = eql[mm[0]:mm[-1]]
    results["osc"] = osc[mm[0]:mm[-1]]
    results["env"] = [env[i] for i in range(len(osc_red))]
    results["osc_red"] = osc_red
    results["nor"] = nor
    results["ang"] = th

    if plot:
        scale = [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
        plt.plot(t[mi:ma], th, 'co', label='Angle')
        plt.yticks(scale, ['0', 'π/2', 'π', '3π/2', '2π'])
        plt.title("Computed Roll Angle")
        plt.show()

    t = t[mi:ma]

    t = [t[i] for i in range(len(th)) if th[i] == th[i]]
    th = [v for v in th if v == v]

    g1 = []
    for i in range(10, len(th) - 10):
        dt = t[i+10]-t[i-10]
        dth = th[i+10]-th[i-10]
        if dth < 0:
            dth += 2 * np.pi
        g1.append(dth/dt)

    tg2 = imu["t_imu_gx"]
    g2 = smooth(imu["imu_gx"])

    if plot:
        plt.plot(tg2[mi-300:ma+300], g2[mi-300:ma+300], 'r', label='Raw Gyro Angular Velocity')
        plt.plot(t[10:-10], g1, 'c', label='Time Derivative of Roll Angle')
        plt.title("Gyro Angular Velocity Comparison")
        plt.legend()
        plt.show()

    return results


def test():
    calc_ang_vel_mag_fit()

    imu = read("imu")
    ana = read("analogue")

    it = ana["t"]
    igx = smooth(ana["mag"])

    plt.plot(it, smooth(igx), "r")
    plt.xlabel("Time (s)")
    plt.ylabel("Magnetometer Reading (V)")
    plt.rc('font', size=2)
    plt.show()

    it = imu["t_imu_my"]
    iax = smooth(imu["imu_ax"])
    iay = smooth(imu["imu_ay"])
    iaz = smooth(imu["imu_az"])

    plt.plot(it, iax, label="x")
    plt.plot(it, iay, label="y")
    plt.plot(it, iaz, label="z")
    plt.legend()
    plt.show()

    at = ana["t"]
    ax = smooth(ana["ax"])
    ay = smooth(ana["ay"])

    plt.plot(at, ax, label="x")
    plt.plot(at, ay, label="y")
    plt.legend()
    plt.show()


def equilibrium(t, v, mi, ma, s=50):
    def model(x, a, b):
        return a * x + b

    eql = []

    for i in range(mi, ma):
        a, b = scipy.optimize.curve_fit(f=model, xdata=t[i - s:i + s], ydata=v[i - s:i + s], check_finite=True)[0]
        eql.append(model(t[i], a, b))

    return eql


def gyr_ang_calc(mi=3950, ma=4650, plot=False):
    imu = read("imu")

    # RAW DATA
    tx = imu[f"t_imu_ax"].tolist()
    ty = imu[f"t_imu_ay"].tolist()
    tz = imu[f"t_imu_az"].tolist()
    gx = list(smooth(imu[f"imu_gx"]))
    gy = list(smooth(imu[f"imu_gy"]))
    gz = list(smooth(imu[f"imu_gz"]))

    if plot:
        plt.plot(ty[mi:ma], gy[mi:ma], 'r', label="y-axis")
        plt.plot(tz[mi:ma], gz[mi:ma], 'b', label="z-axis")
        plt.title("Raw Gyroscope Data")
        plt.legend()
        plt.show()

    # REDUCE NOISE
    gy_red = list(savgol_filter(gy, 30, 2))
    gz_red = list(savgol_filter(gz, 30, 2))

    if plot:
        plt.plot(ty[mi:ma], gy_red[mi:ma], 'r', label="y-axis")
        plt.plot(tz[mi:ma], gz_red[mi:ma], 'b', label="z-axis")
        plt.title("Noise Reduced Gyroscope Data")
        plt.legend()
        plt.show()

    # DETERMINE EQUILIBRIUM
    s = 50

    gy_eql = equilibrium(ty, gy_red, mi-2*s, ma+2*s, s=s)
    gz_eql = equilibrium(tz, gz_red, mi-2*s, ma+2*s, s=s)
    gy_eql = equilibrium(ty[mi - 2*s:ma + 2*s], gy_eql, s, len(gy_eql) - s, s=s)
    gz_eql = equilibrium(tz[mi - 2*s:ma + 2*s], gz_eql, s, len(gz_eql) - s, s=s)
    gy_eql = equilibrium(ty[mi-s:ma+s], gy_eql, s, len(gy_eql)-s, s=s)
    gz_eql = equilibrium(tz[mi-s:ma+s], gz_eql, s, len(gz_eql)-s, s=s)

    if plot:
        plt.plot(ty[mi:ma], gy_red[mi:ma], 'r', label="Y-Gyroscope")
        plt.plot(ty[mi:ma], gy_eql, 'g', label="Estimated Equilibrium Position")
        plt.plot(tz[mi:ma], gz_red[mi:ma], 'b', label="Z-Gyroscope")
        plt.plot(tz[mi:ma], gz_eql, 'y', label="Estimated Equilibrium Position")
        plt.title("Gyroscope Data with Equilibrium of Oscillation")
        plt.legend()
        plt.show()

    # ISOLATE OSCILLATORY COMPONENT
    gy_red = gy_red[mi:ma]
    gz_red = gz_red[mi:ma]
    gy_osc = [gy_red[i] - gy_eql[i] for i in range(len(gy_red))]
    gz_osc = [gz_red[i] - gz_eql[i] for i in range(len(gz_red))]

    if plot:
        plt.plot(ty[mi:ma], gy_osc, 'r', label="Y-Gyroscope")
        plt.plot(tz[mi:ma], gz_osc, 'b', label="Z-Gyroscope")
        plt.title("Isolated Oscillatory Component")
        plt.legend()
        plt.show()

    # NOISE REDUCE SIGNAL TO FIND ENVELOPE
    gy_osc_red = list(savgol_filter(gy_osc, 5, 2))
    gz_osc_red = list(savgol_filter(gz_osc, 5, 2))

    if plot:
        plt.plot(ty[mi:ma], gy_osc, 'r', label="Y-Gyroscope")
        plt.plot(tz[mi:ma], gz_osc, 'b', label="Z-Gyroscope")
        plt.plot(ty[mi:ma], gy_osc_red, 'g', label="Y-Gyroscope")
        plt.plot(tz[mi:ma], gz_osc_red, 'y', label="Z-Gyroscope")
        plt.title("Isolated Oscillatory Component")
        plt.legend()
        plt.show()

    # FIND MINIMA AND MAXIMA
    gy_mii = list(argrelextrema(np.array(gy_osc_red), np.greater)[0])
    gy_mai = list(argrelextrema(np.array(gy_osc_red), np.less)[0])
    gz_mii = list(argrelextrema(np.array(gz_osc_red), np.greater)[0])
    gz_mai = list(argrelextrema(np.array(gz_osc_red), np.less)[0])

    gy_mm = gy_mii + gy_mai
    gz_mm = gz_mii + gz_mai

    gy_mm.sort()
    gz_mm.sort()

    gy_mmv = [gy_osc_red[i] for i in gy_mm]
    gz_mmv = [gz_osc_red[i] for i in gz_mm]

    if plot:
        plt.plot(ty[mi:ma], gy_osc_red, 'r', label="Y-Gyroscope")
        plt.plot(tz[mi:ma], gz_osc_red, 'b', label="Z-Gyroscope")
        plt.plot([ty[mi + i] for i in gy_mm], gy_mmv, 'go')
        plt.plot([tz[mi + i] for i in gz_mm], gz_mmv, 'yo')
        plt.title("Noise Reduced Oscillatory Component")
        plt.legend()
        plt.show()

    # INTERPOLATE ENVELOPE FOR THE SIGNAL
    gy_mmv = [abs(v) for v in gy_mmv]
    gz_mmv = [abs(v) for v in gz_mmv]
    gy_env = list(interp1d(gy_mm, gy_mmv, kind="cubic")(range(gy_mm[0], gy_mm[-1])))
    gz_env = list(interp1d(gz_mm, gz_mmv, kind="cubic")(range(gz_mm[0], gz_mm[-1])))

    if plot:
        plt.plot(ty[mi:ma], gy_osc, 'r', label="Y-Signal")
        plt.plot(tz[mi:ma], gz_osc, 'b', label="Z-Signal")
        plt.plot(ty[mi + gy_mm[0]:mi + gy_mm[-1]], gy_env, "g", label="Y-Envelope")
        plt.plot(tz[mi + gz_mm[0]:mi + gz_mm[-1]], gz_env, "y", label="Y-Envelope")
        plt.plot(ty[mi + gy_mm[0]:mi + gy_mm[-1]], [v * -1 for v in gy_env], "g")
        plt.plot(tz[mi + gz_mm[0]:mi + gz_mm[-1]], [v * -1 for v in gz_env], "y")
        plt.legend()
        plt.title("Interpolated Envelope for Signal")
        plt.show()

    # NORMALIZE SIGNAL
    gy_osc_red = gy_osc_red[gy_mm[0]:gy_mm[-1]]
    gz_osc_red = gz_osc_red[gz_mm[0]:gz_mm[-1]]
    gy_nor = [gy_osc_red[i] / gy_env[i] for i in range(len(gy_osc_red))]
    gz_nor = [gz_osc_red[i] / gz_env[i] for i in range(len(gz_osc_red))]

    if plot:
        plt.plot(ty[mi + gy_mm[0]:mi + gy_mm[-1]], gy_nor, 'r', label="Y-Normalized")
        plt.plot(tz[mi + gz_mm[0]:mi + gz_mm[-1]], gz_nor, 'b', label="Z-Normalized")
        plt.title("Normalized Signal")
        plt.legend()
        plt.show()

    # CALCULATE ANGLE FROM INVERSE SIN
    th = [np.arcsin(v) * -1 if abs(v) <= 1 else np.nan for v in gz_nor]

    zr = min(gz_mii[0], gz_mai[0])
    mi = mi + zr
    ma = mi + max(gz_mii[-1], gz_mai[-1]) - zr
    mii = [v - zr for v in gz_mii]
    mai = [v - zr for v in gz_mai]

    for mai_i in mai:
        try:
            mii_i = next(v for v in mii if v > mai_i)
        except StopIteration:
            continue
        for i in range(mai_i, mii_i):
            th[i] *= -1
            th[i] += np.pi

    for i in range(len(th)):
        if th[i] < 0:
            th[i] += 2 * np.pi

    t = tz[mi:ma]

    for i in reversed(range(len(th))):
        if th[i] != th[i]:
            th.pop(i)
            t.pop(i)

    if plot:
        scale = [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
        plt.plot(t, th, 'co', label='Angle')
        plt.yticks(scale, ['0', 'π/2', 'π', '3π/2', '2π'])
        plt.title("Single Channel Computed Roll Angle")
        plt.show()

    th_single = th
    t_single = t

    # NUMERICALLY DIFFERENTIATE FOR ANG VEL

    t = [t[i] for i in range(len(th)) if th[i] == th[i]]
    th = [v for v in th if v == v]

    g1 = []
    INT = 10
    for i in range(INT + 1, len(th) - INT):
        dt = t[i + INT] - t[i - INT - 1]
        dth = th[i + INT] - th[i - INT - 1]
        if dth < 0:
            dth += 2 * np.pi
        g1.append(dth / dt)

    tg2 = imu["t_imu_gx"]
    g2 = smooth(imu["imu_gx"])

    if plot:
        plt.plot(tg2, g2, 'r', label='Raw Gyro Angular Velocity') # [mi - 300:ma + 300]
        plt.plot(t[INT:(-INT - 1)], g1, 'c', label='Time Derivative of Roll Angle')
        plt.title("Single Channel Gyro Angular Velocity Comparison")
        plt.legend()
        plt.show()

    # MULTICHANNEL ANGLE COMPUTATION

    def inter(t1: float, t_list: list[float], v_list: list[float]) -> float:
        i, j = [(i-1, i) for i in range(len(t_list)) if t_list[i] > t1][0]
        frac = (t1 - t_list[i]) / (t_list[j] - t_list[i])
        v1 = v_list[i] + frac * (v_list[j] - v_list[i])
        return v1

    ty = ty[mi + gy_mm[0]:mi + gy_mm[-1]]
    tz = tz[mi + gz_mm[0]:mi + gz_mm[-1]]

    gy_nor = [inter(tz[i], ty, gy_nor) for i in range(len(tz))]

    t = tz

    tan_th = [gy_nor[i] / gz_nor[i] for i in range(len(gy_nor))]

    if plot:
        plt.plot(t, tan_th, 'r', label="YZ-Normalized")
        plt.title("Normalized Signals Merged")
        plt.legend()
        plt.show()

    th = [np.arctan(v) for v in tan_th]

    if plot:
        plt.plot(t, th, 'co', label="YZ-Normalized")
        plt.plot(t, tan_th, 'r', label="YZ-Normalized")
        plt.plot(t, gz_nor, 'b', label="YZ-Normalized")
        plt.title("Normalized Signals Merged")
        plt.legend()
        plt.show()

    for i in range(len(th)):
        if gz_nor[i] < 0:
            th[i] += np.pi
        if th[i] < 0:
            th[i] += 2 * np.pi

    if plot:
        scale = [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
        plt.plot(t, th, 'co', label='Angle')
        plt.yticks(scale, ['0', 'π/2', 'π', '3π/2', '2π'])
        plt.title("Multi Channel Computed Roll Angle")
        plt.show()

    g1 = []
    for i in range(INT + 1, len(th) - INT):
        dt = t[i + INT] - t[i - INT - 1]
        dth = th[i + INT] - th[i - INT - 1]
        if dth < 0:
            dth += 2 * np.pi
        g1.append(dth / dt)

    tg2 = imu["t_imu_gx"]
    g2 = smooth(imu["imu_gx"])

    if plot:
        plt.plot(tg2, g2, 'r', label='Raw Gyro Angular Velocity') # [mi - 300:ma + 300]
        plt.plot(t[INT:(-INT - 1)], g1, 'c', label='Time Derivative of Roll Angle')
        plt.title("Multi Channel Gyro Angular Velocity Comparison")
        plt.legend()
        plt.show()

    results = {}

    results["t"] = t_single
    results["nor"] = gz_nor
    results["ang"] = th_single

    return results


def unmod_ang(th: list):
    offset = 0
    new_th = [th[0]]
    for i in range(1, len(th)):
        if th[i] < th[i - 1]:
            offset += 2 * np.pi
        new_th.append(th[i] + offset)
    return new_th


def roll_angle():
    results = gyr_ang_calc(3200, 5100, True)
    t = results["t"]
    ang = results["ang"]

    ang = unmod_ang(ang)

    ang_red = list(savgol_filter(ang, 300, 3))

    vel = []
    vel_red = []

    for i in range(len(ang)-1):
        vel.append((ang[i+1]-ang[i])/(t[i+1]-t[i]))
        vel_red.append((ang_red[i + 1] - ang_red[i]) / (t[i + 1] - t[i]))

    imu = read("imu")

    imu_t = imu[f"t_imu_gx"].tolist()
    imu_vel = list(smooth(imu[f"imu_gx"]))

    '''
    plt.plot(t[:-1], vel, "co")
    plt.plot(t[:-1], vel_red, "ro")
    plt.plot(imu_t, imu_vel, "g")
    plt.show()
    '''

    for i in range(len(ang_red)):
        while ang_red[i] > 2 * np.pi:
            ang_red[i] -= 2 * np.pi

    results["ang_red"] = ang_red
    del results["nor"]
    results_df = pd.DataFrame(results)
    results_df.to_csv("Output Data\Roll.csv", index=False)


def equilibrium_v2(t, v, s=50):
    def model(x, a, b):
        return a * x + b

    eql = []

    for i in range(0, len(v)):
        mi, ma = i - s, i + s
        if mi < 0:
            mi, ma = 0, 2 * s
        elif ma > len(v) - 1:
            mi, ma = len(v) - 2 * s - 1, len(v) - 1
        a, b = scipy.optimize.curve_fit(f=model, xdata=t[mi:ma], ydata=v[mi:ma], check_finite=True)[0]
        eql.append(model(t[i], a, b))

    return eql


def yaw_angle():
    plot = True

    gps = read("gps")

    t = gps["t"]
    lt = gps["lat"]
    lg = gps["long"]

    if plot:
        plt.plot(t, lt, 'r', label="Latitude")
        # plt.plot(t, lg, 'r', label="Longitude")
        plt.title("GPS Data")
        plt.legend()
        plt.show()

    dlt, dlg = [], []

    for i in range(1, len(t)):
        dlt.append(lt[i] - lt[i-1])
        dlg.append(lg[i] - lg[i-1])

    if plot:
        plt.plot(t[:-1], dlt, 'ro', label="Change in Latitude")
        plt.plot(t[:-1], dlg, 'bo', label="Change in Longitude")
        plt.title("GPS Data")
        plt.legend()
        plt.show()

    tan_th = [dlg[i]/dlt[i]*np.cos(np.radians(lt[i])) for i in range(len(dlg))]

    if plot:
        plt.plot(t[:-1], tan_th, 'ro', label="Ratio of Long/Lat Change")
        plt.title("GPS Data")
        plt.legend()
        plt.show()

    ang = [np.degrees(np.arctan(v)) for v in tan_th]

    if plot:
        plt.plot(t[:-1], ang, 'co', label="Heading (°)")
        plt.title("GPS Data")
        plt.legend()
        plt.show()

    t_nnan = [t[i] for i in range(len(ang)) if ang[i] == ang[i]]
    ang_nnan = [v for v in ang if v == v]

    if plot:
        plt.plot(t_nnan, ang_nnan, 'co', label="Heading (°)")
        plt.title("GPS Data")
        plt.legend()
        plt.show()

    if plot:
        plt.plot(t_nnan, ang_nnan, 'ro', label="Heading (°)")
        plt.plot(t_nnan, smooth(pd.Series(ang_nnan)), 'bo', label="Noise Reduced Heading (°)")
        plt.ylim(-180, 180)
        plt.title("GPS Data")
        plt.legend()
        plt.show()

    ang_red = equilibrium_v2(t_nnan, list(smooth(pd.Series(ang_nnan))), 10)
    for i in range(1, 20):
        ang_red = equilibrium_v2(t_nnan, ang_red, 10)

    ang_red = [ang + 360 if ang < 0 else ang for ang in ang_red]
    ang_nnan = [ang + 360 if ang < 0 else ang for ang in ang_nnan]
    ang_red = [ang * 2 * np.pi / 360 for ang in ang_red]
    ang_nnan = [ang * 2 * np.pi / 360 for ang in ang_nnan]

    if plot:
        plt.plot(t_nnan, ang_nnan, 'ro', label="Heading (rad)")
        plt.plot(t_nnan, ang_red, 'b', label="Noise Reduced Heading (rad)")
        #plt.ylim(-180, 180)
        plt.title("GPS Data")
        plt.legend()
        plt.show()

    results = {}
    results["t"] = t_nnan
    results["ang"] = ang_nnan
    results["ang_red"] = ang_red
    results_df = pd.DataFrame(results)
    results_df.to_csv("Output Data\Yaw.csv", index=False)


def pitch_angle():
    plot = True

    gps = read("gps")

    t = gps["t"]
    lt = gps["lat"]
    lg = gps["long"]
    h = gps["height"]
    s = gps["speed"]

    if plot:
        plt.plot(t, h, 'r', label="Height")
        # plt.plot(t, lg, 'r', label="Longitude")
        plt.title("GPS Data")
        plt.legend()
        plt.show()

    dt, dlt, dlg, dh = [], [], [], []

    for i in range(1, len(t)):
        dt.append(t[i] - t[i-1])
        dlt.append(lt[i] - lt[i-1])
        dlg.append(lg[i] - lg[i-1])
        dh.append(h[i] - h[i-1])

    if plot:
        plt.plot(t[:-1], dh, 'ro', label="Change in Height")
        plt.title("GPS Data")
        plt.legend()
        plt.show()

    r = 6378137

    dy = [dlt[i] / 360 * 2 * np.pi * (r + h[i]) for i in range(len(dlt))]
    dx = [dlg[i] * np.cos(np.radians(lt[i])) / 360 * 2 * np.pi * (r + h[i]) for i in range(len(dlg))]

    if plot:
        plt.plot(t[:-1], dh, 'ro', label="Change in Z")
        plt.plot(t[:-1], dx, 'bo', label="Change in X")
        plt.plot(t[:-1], dy, 'go', label="Change in Y")
        plt.title("GPS Data")
        plt.legend()
        plt.show()

    dd = [np.sqrt(dx[i]**2 + dy[i]**2) for i in range(len(dx))]
    th = [np.arctan(dh[i] / dd[i]) * 360 / 2 / np.pi for i in range(len(dh))]

    if plot:
        plt.plot(t[:-1], th, 'co', label="Elevation (°)")
        plt.title("GPS Data")
        plt.legend()
        plt.show()

    s2 = [np.sqrt(dd[i]**2 + dh[i]**2) / dt[i] for i in range(len(dd))]

    if plot:
        plt.plot(t[:-1], s2, 'co', label="Calculated Speed (m/s)")
        plt.plot(t[:-1], s[:-1], 'ro', label="GPS Speed (m/s)")
        plt.title("GPS Data")
        plt.legend()
        plt.show()

    t_nnan = [t[i] for i in range(len(th)) if th[i] == th[i]]
    th_nnan = [v for v in th if v == v]

    if plot:
        plt.plot(t_nnan, th_nnan, 'co', label="Elevation (°)")
        plt.title("GPS Data")
        plt.legend()
        plt.show()

    th_red = equilibrium_v2(t_nnan, list(smooth(pd.Series(th_nnan))), 5)
    for i in range(1, 10):
        th_red = equilibrium_v2(t_nnan, th_red, 5)

    th_red = [ang * 2 * np.pi / 360 for ang in th_red]
    th_nnan = [ang * 2 * np.pi / 360 for ang in th_nnan]

    if plot:
        plt.plot(t_nnan, th_nnan, 'ro', label="Heading (°)")
        plt.plot(t_nnan, th_red, 'b', label="Noise Reduced Heading (°)")
        plt.title("GPS Data")
        plt.legend()
        plt.show()

    results = {}
    results["t"] = t_nnan
    results["ang"] = th_nnan
    results["ang_red"] = th_red
    results_df = pd.DataFrame(results)
    results_df.to_csv("Output Data\Pitch.csv", index=False)

    results = {}
    results["t"] = t[:-1]
    results["dx"] = dx
    results["dy"] = dy
    results["dz"] = dh
    results["s"] = s2
    results_df = pd.DataFrame(results)
    results_df.to_csv("Output Data\Velocity.csv", index=False)


def interpolate(x2: list, x1: list, y1: list) -> list:
    return list(interp1d(x1, y1, kind="cubic")(x2))


def transform_mag():
    plot = False

    roll = pd.read_csv("Output Data/Roll.csv")
    pitch = pd.read_csv("Output Data/Pitch.csv")
    yaw = pd.read_csv("Output Data/Yaw.csv")

    rt = roll["t"]
    ra = roll["ang_red"]
    pt = pitch["t"]
    pa = pitch["ang_red"]
    yt = yaw["t"]
    ya = yaw["ang_red"]

    if plot:
        plt.plot(rt, ra, 'r', label="Roll Angle")
        plt.plot(pt, pa, 'g', label="Pitch Angle")
        plt.plot(yt, ya, 'b', label="Yaw Angle")
        plt.title("Rotation Data")
        plt.legend()
        plt.show()

    ra = unmod_ang(ra)

    imu = read("imu")

    tx = imu["t_imu_mx"]
    mx = imu["imu_mx"]
    ty = imu["t_imu_my"]
    my = imu["imu_my"]
    tz = imu["t_imu_mz"]
    mz = imu["imu_mz"]

    tx = [t for i, t in enumerate(tx) if mx[i] == mx[i]]
    mx = [m for m in mx if m == m]
    ty = [t for i, t in enumerate(ty) if my[i] == my[i]]
    my = [m for m in my if m == m]
    tz = [t for i, t in enumerate(tz) if mz[i] == mz[i]]
    mz = [m for m in mz if m == m]

    if plot:
        plt.plot(tx, mx, 'r', label="Mag X")
        plt.plot(ty, my, 'g', label="Mag Y")
        plt.plot(tz, mz, 'b', label="Mag Z")
        plt.title("Mag Data")
        plt.legend()
        plt.show()

    my_eq = equilibrium_v2(ty, my, 100)
    mz_eq = equilibrium_v2(tz, mz, 100)
    mx_eq = equilibrium_v2(tx, mx, 100)
    for i in range(1, 4):
        my_eq = equilibrium_v2(ty, my_eq, 100)
        mz_eq = equilibrium_v2(tz, mz_eq, 100)
        mx_eq = equilibrium_v2(tx, mx_eq, 100)

    dmxdt = [(mx_eq[i] - mx_eq[i - 1]) / (tx[i] - tx[i - 1]) for i in range(1, len(mx_eq))][1000:]
    mxzr = mx_eq[1000:][dmxdt.index(max(dmxdt))]

    # mx = [v - mxzr for v in mx]
    my = [v - my_eq[i] for i, v in enumerate(my)]
    mz = [v - mz_eq[i] for i, v in enumerate(mz)]

    if plot:
        plt.plot(tx, mx, 'r', label="Mag X")
        plt.plot(ty, my, 'g', label="Mag Y")
        plt.plot(tz, mz, 'b', label="Mag Z")
        plt.title("Mag Data Rebased")
        plt.legend()
        plt.show()

    tmi = next(i for i in range(len(tx)) if tx[i] > rt[0])
    tma = next(i for i in reversed(range(len(tx))) if tx[i] < rt[len(rt)-1])

    t = tx[tmi:tma]

    mx = mx[tmi:tma]
    my = interpolate(t, ty, my)
    mz = interpolate(t, tz, mz)

    ra = interpolate(t, rt, ra)
    pa = interpolate(t, pt, pa)
    ya = interpolate(t, yt, ya)

    if not plot:
        plt.plot(t, ra, 'ro', label="Roll Angle")
        plt.plot(t, pa, 'go', label="Pitch Angle")
        plt.plot(t, ya, 'bo', label="Yaw Angle")
        plt.title("Rotation Data")
        plt.legend()
        plt.show()

    if plot:
        plt.plot(t, mx, 'r', label="Mag X")
        plt.plot(t, my, 'g', label="Mag Y")
        plt.plot(t, mz, 'b', label="Mag Z")
        plt.title("Mag Data")
        plt.legend()
        plt.show()

    m = [np.array([mx[i], my[i], mz[i]]) for i in range(len(t))]

    mag1 = [np.sqrt(np.dot(v, v)) for v in m]

    if plot:
        plt.plot(t, mx, 'r', label="Mag X")
        plt.plot(t, my, 'g', label="Mag Y")
        plt.plot(t, mz, 'b', label="Mag Z")
        plt.ylim(-0.001, 0.001)
        plt.title("Original Mag Data")
        plt.legend()
        plt.show()

    if plot:
        plt.plot(t, ra, 'r', label="Roll Angle")
        plt.title("Rotation Data")
        plt.legend()
        plt.show()

    for i in range(len(m)):
        th = -ra[i]
        rot_mat = np.array([[1, 0, 0],
                            [0, np.cos(th), -np.sin(th)],
                            [0, np.sin(th), np.cos(th)]])
        m[i] = np.matmul(rot_mat, m[i])

    for i in range(len(m)):
        rot_mat = np.array([[np.cos(-pa[i]), 0, np.sin(-pa[i])],
                            [0, 1, 0],
                            [-np.sin(-pa[i]), 0, np.cos(-pa[i])]])
        m[i] = np.matmul(rot_mat, m[i])

    for i in range(len(m)):
        rot_mat = np.array([[np.cos(-ya[i]), -np.sin(-ya[i]), 0],
                            [np.sin(-ya[i]), np.cos(-ya[i]), 0],
                            [0, 0, 1]])
        m[i] = np.matmul(rot_mat, m[i])

    mag2 = [np.sqrt(np.dot(v, v)) for v in m]

    mx = [v[0] for v in m]
    my = [v[1] for v in m]
    mz = [v[2] for v in m]

    if plot:
        plt.plot(t, mx, 'r', label="Mag X")
        plt.plot(t, my, 'g', label="Mag Y")
        plt.plot(t, mz, 'b', label="Mag Z")
        plt.title("Transformed Mag Data")
        plt.legend()
        plt.show()

    if plot:
        plt.plot(t, mag1, 'r', label="Magnitude Pre-Transformation")
        plt.plot(t, mag2, 'b', label="Magnitude Post-Transformation")
        plt.title("Magnetometer Magnitudes")
        plt.legend()
        plt.show()

    results = {}
    results["t"] = t
    results["mx"] = mx
    results["my"] = my
    results["mz"] = mz
    results_df = pd.DataFrame(results)
    results_df.to_csv("Output Data\Mag.csv", index=False)


def main():
    transform_mag()



    pass


if __name__ == "__main__":
    main()




















