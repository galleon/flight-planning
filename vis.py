import time
import warnings

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import openap.top as otop
import pandas as pd
from cartopy import crs as ccrs
from openap.top.vis import trajectory_on_map

# Interesting reads:
# Please see: https://matplotlib.org/cheatsheets/
# Get meteo data from https://cds.climate.copernicus.eu/cdsapp#!/yourrequests?tab=form
# https://medium.com/@elliottwobler/how-to-get-started-with-grib2-weather-data-and-python-757df9433d19

warnings.filterwarnings("ignore")


print("Loading wind data...")
# 14 July 2022
# fgrib = "data/adaptor.mars.internal-1659542120.0818555-7415-17-28021c92-3c3e-4776-aa1d-44b6e760d38a.grib"
# 4 October 2021
# fgrib = "data/adaptor.mars.internal-1659545781.3572798-29765-5-4e4de681-9034-4050-a1e4-d023eab9de77.grib"
# 21 February 2022 (Franklin Hurricane)
# fgrib = "data/adaptor.mars.internal-1659546292.644144-3871-12-dd46e7b3-d9f4-4b4b-9e70-9f06039b4fc0.grib"
# 1st May 2021 8am
fgrib = "data/adaptor.mars.internal-1659588957.360821-19344-9-b2135488-9725-486e-b3d2-adf40cb68242.grib"
fgrib = "data/adaptor.mars.internal-1660049600.592577-31348-4-6fc2ef7e-de0a-4dcb-9a4e-6d61345f126f.grib"
windfield = otop.wind.read_grib(fgrib)

print(windfield.columns)

print(f"Loading trajectory data...")
flight = pd.read_csv("data/trajectory_LFPG_WSSS.csv")

print(f"Flight time: {flight.iloc[-1]['ts']}")

print("Plotting trajectory...")

fig = plt.figure(figsize=(11, 5))  # tight_layout=True
gs = gridspec.GridSpec(3, 2, wspace=0.4, hspace=0.3)

ax4 = fig.add_subplot(
    gs[:, 1],
    projection=ccrs.TransverseMercator(
        central_longitude=flight.lon.mean(), central_latitude=flight.lat.mean()
    ),
)

wind_sample = int(flight.iloc[-1]["ts"] / 500)
trajectory_on_map(flight, windfield=windfield, ax=ax4, wind_sample=wind_sample)

ax1 = fig.add_subplot(gs[0, 0])
ax1.plot("ts", "alt", data=flight, marker="o", color="grey", alpha=0.3)
ax1.yaxis.tick_right()
ax1.xaxis.grid(True, which="major")
ax1.yaxis.grid(True, which="major")
ax1.set_ylabel("Altitude (feet)", fontsize=10)

ax2 = fig.add_subplot(gs[1, 0])
ax2.plot("ts", "tas", data=flight, marker="o", color="grey", alpha=0.3)
ax2.yaxis.tick_right()
ax2.xaxis.grid(True, which="major")
ax2.yaxis.grid(True, which="major")
ax2.set_ylabel("TAS (kts)", fontsize=10)

ax3 = fig.add_subplot(gs[2, 0])
ax3.plot("ts", "vs", data=flight, marker="o", color="grey", alpha=0.3)
ax3.yaxis.tick_right()
ax3.set_ylabel("Vs (feet/mn)", fontsize=10)
ax3.set_xlabel("t (s)", fontsize=10)

plt.savefig("openap-top.png")
plt.show()
