import os
import numpy as np
from datetime import datetime, timedelta
from functools import reduce
import urllib.request as request

# try:
#    from osgeo import gdal
#    from osgeo.gdalconst import GA_ReadOnly
# except:
#    pass

import getgfs

import collections
import calendar

from pybada.bada4 import AircraftPosition


class WeatherModel(object):
    def __init__(self, dof: datetime, forecast: str = "nowcast"):
        self.dof = dof
        #
        # Set forecast
        #
        self.f = getgfs.Forecast("0p25", "1hr")

    def get_wind_speed(self, aircraft_position: AircraftPosition):
        #
        # Get the wind speed from the GFS
        current_datetime = self.dof + timedelta(seconds=aircraft_position.t)

        date_str = current_datetime.strftime("%Y%m%d %H:%M:%S")

        print(
            f"date_str: {date_str} <lat, lon>: {aircraft_position.lat, aircraft_position.lon}"
        )

        res = self.f.get(
            ["ugrdtrop", "vgrdtrop"],
            date_str,
            aircraft_position.lat,
            aircraft_position.lon,
        )


if __name__ == "__main__":
    now = datetime.now()
    weather_model = WeatherModel(now)
