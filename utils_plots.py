import os
from datetime import datetime
import folium
from folium import plugins
from folium.plugins import HeatMap
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely import Point, LineString
from geographiclib.geodesic import Geodesic
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def plot_sensor_grid(dataloggers, sensors, zones, savefig=False):
    fig, ax = plt.subplots()

    zones.plot(color = 'red', alpha = 0.6, ax=ax)
    snsrs = sensors.plot(column='zone', cmap='tab20', legend=True, ax=ax, label='sensors', 
                        legend_kwds={
                            # 'label': 'zone',
                            'location': 'right',
                        })
    ax.text(1.06, -0.07, 'Zone', transform=ax.transAxes)
    # cbar = plt.colorbar(ax=ax)
    
    dataloggers.plot(color='black', ax=ax, marker='x', label='dataloggers')
    ax.legend(facecolor='grey')
    ax.ticklabel_format(useOffset=False)
    ax.tick_params(axis='both', which='major', labelsize=8)
    plt.xticks(rotation=45)
    fig.tight_layout()
    fn = os.path.join(os.getcwd(), 'output', f'atmos_soilsensors_{datetime.now().strftime("%Y-%m-%d")}.png')
    if savefig: plt.savefig(fn)
    plt.show()
    return