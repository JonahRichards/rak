import datetime

import numpy as np
import pandas
import pandas as pd

import utilities
import tkinter as tk


root = tk.Tk()
root.withdraw()
root.wm_attributes('-topmost', 1)


def read(name: str) -> pandas.DataFrame:
    file_name = f"{root}\Processed Data\Raks_tm_readout_{name}.csv"
    data = pandas.read_csv(file_name)
    return data


def calc_ang_vel_1():
    SP = 64

    times = read("imu")["t_imu_gx"]
    mags = read("analogue")[["t", "mag"]]

    freqs_dicts = []

    for t in times:
        print(datetime.datetime.now().time())

        i = next(i for i, v in mags.iterrows() if v["t"] >= t)

        min_i = max(0, int(i-SP/2))
        max_i = min(len(mags), int(i+SP/2))
        points = list(mags["mag"][min_i:max_i])

        coeffs = np.fft.fft(points)
        freq_num = coeffs.argmax()

        spacing = mags["t"][max_i] - mags["t"][min_i]/SP
        freqs = np.fft.fftfreq(SP, spacing)
        freq = freqs[freq_num]*2*np.pi

        freqs_dicts.append({"t": t, "freq": freq})

    print()


def main():
    calc_ang_vel_1()




if __name__ == "__main__":
    main()