# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 09:22:05 2024

@author: Janek
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sfactor(start_date, end_date, amplitude=1, phase_shift=0, frequency=1):
    # Generate datetime range for a year
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')

    # Calculate cosine values
    values = amplitude * np.cos(2 * np.pi * frequency * (date_range - start_date).days / 365 + phase_shift)

    # Create a DataFrame with datetime index and cosine values
    cosine_wave = pd.DataFrame({'Datetime': date_range, 'Wert': values})
    cosine_wave.set_index('Datetime', inplace=True)

    return cosine_wave

#%% Preparing River stage data
slach = pd.read_csv('model_data/Steinlach.csv', encoding='latin-1', skiprows=8)
tssl     = slach[['Datum / Uhrzeit', 'Wert']]
tssl.loc[:, 'Datum / Uhrzeit'] = pd.to_datetime(tssl['Datum / Uhrzeit'], format='%Y-%m-%d %H:%M')
tssl.set_index('Datum / Uhrzeit', inplace=True)

# alter data so it meets the 2m amplitude
tssl.loc[:,'Wert'] = tssl['Wert'] / 100 *3.8  +12.1

# Resample and calculate the mean for every 6 hours
tssl6h = tssl.resample('6H').mean() - 1

print(np.max(tssl6h['Wert']), np.min(tssl6h['Wert']), np.mean(tssl6h['Wert']))

#%% Generating recharge scaling factor

start_date  = tssl.index[0]
end_date    = tssl.index[-1]


sfac     = sfactor(start_date, end_date, amplitude=1.25, phase_shift=-0.75, frequency=1) +1
sfac     = sfac.resample('6H').mean()
sfac[sfac < 0.1] = 0.1

#%% Plotting

# chaning timestamps
days_since_start = (tssl6h.index - tssl6h.index[0]).days
slc = np.array([1,101,165])/256
sfc = np.array([188,75,31])/256

fig2, ax3 = plt.subplots(figsize=(6,3))
ax3.plot(days_since_start,tssl6h['Wert'], label='Steinlach', color = slc, linewidth=0.9)
ax3.set_ylabel('River Stage (m)', color=slc)
ax3.tick_params('y', colors=slc)
sl_ticks = np.linspace(13, 15.5, 6)
ax3.set_yticks(sl_ticks)
ax3.set_ylim(12, 17)

# Create the second plot with the second y-axis
ax4 = ax3.twinx()
ax4.plot(days_since_start,sfac['Wert'], label='Steinlach', color=sfc, linewidth=0.9)
ax4.set_ylabel('Scaling Factor (-)', color=sfc)
ax4.tick_params('y', colors=sfc)
sf_ticks = np.linspace(0,2.5, 6)
ax4.set_yticks(sf_ticks)
ax4.set_ylim(0, 2.5)

plt.xlim(days_since_start[0], days_since_start[-1])
plt.xticks(np.linspace(50, 350, 7))
plt.xlabel('Time (days)')

plt.show()

#%% Store results
sfac.to_csv('model_data/sfac.csv')
tssl6h.to_csv('model_data/tssl.csv')

