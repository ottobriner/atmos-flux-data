import os
from datetime import datetime
import pandas as pd
import numpy as np
import pywt
from utils import mra8
# import wavefuncs as wave

from cycler import cycler
from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

def read_l3(filepath, datecol='DATE'):
    l3 = (pd.read_csv(filepath, header=0, low_memory=False)
      .replace(-9999, np.nan).replace(99999, np.nan)
      .assign(date = lambda x: pd.to_datetime(x[datecol]))
      .set_index('date', drop=False)
     )
    l3.index = l3.index.set_names(None)
    return l3

def plot_wavelet_raw(l3):
    fig, axs = plt.subplots(level+1, 1, 
        figsize = (10*1.5, 5.65*1.5), 
        # layout='constrained'
        )

    for i, ax in enumerate(axs):
        ax.plot(l3[cols[0]+'_f'] * 1000, **point_style)
        ax.margins(x=0)
        if i != len(axs)-1:
            ax.set_xticks([])
        rax = ax.twinx()
        rax.plot(l3[cols[0]+f'_w{i}'], **line_style)
        rax.margins(x=0)

    ax = fig.add_subplot(1, 1, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    [ax.spines[side].set_visible(False) for side in ('left', 'top', 'right', 'bottom')]
    ax.patch.set_visible(False)
    ax.set_xlabel('date', labelpad=30)
    ax.set_ylabel('FCH4_f (nmol m-2 s-1)', labelpad=35)

    ax = fig.add_subplot(1, 1, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    [ax.spines[side].set_visible(False) for side in ('left', 'top', 'right', 'bottom')]
    ax.patch.set_visible(False)
    ax.yaxis.set_label_position('right')
    ax.set_ylabel('Wavelet detail (nmol m-2 m-1)', labelpad=45, rotation=270)

    fig.tight_layout(pad=0)

    outpath = os.path.join(os.getcwd(), 'output', f'fig_wavelet_raw_{datetime.now().strftime("%Y-%m-%d")}.png')
    plt.savefig(outpath)
    return

def plot_wavelet_summed(l3):
    fig, axs = plt.subplots(len(timescales_to_sum), 1, 
        figsize = (10*1.5, 5.65*1.5), 
        # layout='constrained'
        )

    for i, ax in enumerate(axs):
        ax.plot(l3[cols[0]+'_f'] * 1000, **point_style)
        ax.margins(x=0)
        if i != len(axs)-1:
            ax.set_xticks([])
        rax = ax.twinx()
        rax.plot(l3[cols[0]+f'_W{i}'], **line_style)
        rax.margins(x=0)

    ax = fig.add_subplot(1, 1, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    [ax.spines[side].set_visible(False) for side in ('left', 'top', 'right', 'bottom')]
    ax.patch.set_visible(False)
    ax.set_xlabel('date', labelpad=5)
    ax.set_ylabel('FCH4_f (nmol m-2 s-1)', labelpad=35)

    ax = fig.add_subplot(1, 1, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    [ax.spines[side].set_visible(False) for side in ('left', 'top', 'right', 'bottom')]
    ax.patch.set_visible(False)
    ax.yaxis.set_label_position('right')
    ax.set_ylabel('Wavelet detail (nmol m-2 m-1)', labelpad=45, rotation=270)

    fig.tight_layout(pad=0)

    outpath = os.path.join(os.getcwd(), 'output', f'fig_wavelet_{datetime.now().strftime("%Y-%m-%d")}.png')
    plt.savefig(outpath)
    return

def plot_wavelet_variance(l3):
    fig, ax = plt.subplots()

    scales = [str(scale+1) for scale in range(level+1)]
    wavelet_variances = l3.loc[:, [f'FCH4_w{i}' for i in range(level+1)]].var(axis=0)
    total_var = wavelet_variances.sum()
    heights = wavelet_variances / total_var * 100

    ax.bar(x = scales, height = heights, color = 'k')
    ax.set(xlabel = 'Time scale', ylabel = 'FCH4_f wavelet variance (% of total)')

    textstr = '\n'.join((
        '1: 1h',
        '2: 2h',
        '3: 4h',
        '4: 8h',
        '5: 16h',
        '6: 1.33 days',
        '7: 2.67 days',
        '8: 5.33 days',
        '9: 10.67 days',
        '10: 21.33 days',
        '11: 42.67 days',
    ))
    
    ax.text(0.05, 0.98, textstr, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', 
    )
    
    fig.tight_layout()
    
    outpath = os.path.join(os.getcwd(), 'output', f'fig_wavelet_variances_{datetime.now().strftime("%Y-%m-%d")}.png')
    plt.savefig(outpath)
    return

if __name__ == '__main__':
    
    fp = r"C:\Users\ottob\data\atmos-flux-data\processed\ATMOS_L3_2024-04-09_fluxgapfill.csv"
    l3 = read_l3(fp)
    l3['FCH4_f_nmol'] = l3.loc[:, 'FCH4_f'] * 1000

    # max usable decomposition level is 12 for JP-BBY and LA8 wavelet
    level = pywt.dwt_max_level(len(l3), pywt.Wavelet('sym8'))
    wavelet = 'sym8'

    # wavelet decompose chosen columns over full time series
    # cols = ['FCH4', 'LE']
    cols = ['FCH4']

    for col in cols:
        c=mra8(l3[f'{col}_f_nmol'].to_numpy(), level=level, axis=0)
        l3 = l3.join(pd.DataFrame(np.array(c).T, index=l3.index, columns=[f'{col}_w{j}' for j in range(level+1)]))

    timescales_to_sum = {
        'hourly': [0,1],
        'diel': [2,3,4,5],
        'multiday': [6,7,8,9],
        'seasonal': [10],
    }

    for col in cols:
        for i, Wscale in enumerate(timescales_to_sum):
            l3[f'{col}_W{i}'] = l3.loc[:, [f'{col}_w{wscale}' for wscale in timescales_to_sum[Wscale]]].sum(axis=1)

    point_style = {
        'color': '#777777',
        'marker': '.',
        'linestyle': 'None',
        'markersize': 2
        }

    line_style = {
        'color' : 'r', 
        'marker': 'None',
        'linestyle': '-',
        'lw': 2
        }

    plot_wavelet_raw(l3)
    plot_wavelet_summed(l3)
    plot_wavelet_variance(l3)