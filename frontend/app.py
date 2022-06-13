import streamlit as st

from streamlit_folium import st_folium
import folium
from folium.features import CustomIcon
from folium.plugins import MarkerCluster

import pandas as pd

from pygeodesy import geohash
from pygeodesy.ellipsoidalVincenty import LatLon

import random


def city_to_lat_lon(city):
    # https://www.latlong.net/
    if city == "New York":
        return LatLon(40.730610, -73.935242)
    elif city == "Tokyo":
        return LatLon(35.6895, 139.6917)
    elif city == "London":
        return LatLon(51.5074, 0.1278)
    elif city == "Paris":
        return LatLon(48.8566, 2.3522)
    elif city == "Toulouse":
        return LatLon(43.6044, 1.4436)


c1, c2 = st.sidebar.columns(2)
city_list1 = ["Paris", "New York", "Tokyo", "London", "Toulouse"]
city1 = c1.selectbox("City pair", city_list1)

city_list2 = [city for city in city_list1 if city != city1]
city2 = c2.selectbox("", city_list2)

date = st.sidebar.date_input("Date", value=pd.Timestamp("2020-01-01"))
aircraft = st.sidebar.selectbox("Aircraft", ("A320", "A380"))

p0 = city_to_lat_lon(city1)
p1 = city_to_lat_lon(city2)

dist_p0top1 = p0.distanceTo(p1)

np0 = int(dist_p0top1 / 100000)
if np0 % 2 == 0:
    np0 += 1

np0 = max(11, np0)

np_min = np0 - 8
np_max = np0 + 8

nc0 = 11

nc_min = nc0 - 8
mc_max = nc0 + 8

np: int = st.sidebar.slider("Number of trajectory points", np_min, np_max, value=np0)
nc: int = st.sidebar.slider("Number of side points", 3, 50, value=nc0)

distp = 10 * dist_p0top1 / np / nc  # meters

if city1 != city2 and np % 2 == 1 and nc % 2 == 1:
    np2 = np // 2
    nc2 = nc // 2

    pt = [[None for j in range(nc)] for i in range(np)]

    # set boundaries
    for j in range(nc):
        pt[0][j] = p0
        pt[np - 1][j] = p1

    # direct path between p0 and p1
    for i in range(1, np - 1):
        bearing = pt[i - 1][nc2].initialBearingTo(p1)
        total_distance = pt[i - 1][nc2].distanceTo(pt[np - 1][nc2])
        pt[i][nc2] = pt[i - 1][nc2].destination(total_distance / (np - i), bearing)

    bearing = pt[np2 - 1][nc2].initialBearingTo(pt[np2 + 1][nc2])
    pt[np2][nc - 1] = pt[np2][nc2].destination(distp * nc2, bearing + 90)
    pt[np2][0] = pt[np2][nc2].destination(distp * nc2, bearing - 90)

    for j in range(1, nc2 + 1):
        # +j (left)
        bearing = pt[np2][nc2 + j - 1].initialBearingTo(pt[np2][nc - 1])
        total_distance = pt[np2][nc2 + j - 1].distanceTo(pt[np2][nc - 1])
        pt[np2][nc2 + j] = pt[np2][nc2 + j - 1].destination(
            total_distance / (nc2 - j + 1), bearing
        )
        # -j (right)
        bearing = pt[np2][nc2 - j + 1].initialBearingTo(pt[np2][0])
        total_distance = pt[np2][nc2 - j + 1].distanceTo(pt[np2][0])
        pt[np2][nc2 - j] = pt[np2][nc2 - j + 1].destination(
            total_distance / (nc2 - j + 1), bearing
        )
        for i in range(1, np2):
            # first halp (p0 to np2)
            bearing = pt[i - 1][nc2 + j].initialBearingTo(pt[np2][nc2 + j])
            total_distance = pt[i - 1][nc2 + j].distanceTo(pt[np2][nc2 + j])
            pt[i][nc2 + j] = pt[i - 1][nc2 + j].destination(
                total_distance / (np2 - i + 1), bearing
            )
            bearing = pt[i - 1][nc2 - j].initialBearingTo(pt[np2][nc2 - j])
            total_distance = pt[i - 1][nc2 - j].distanceTo(pt[np2][nc2 - j])
            pt[i][nc2 - j] = pt[i - 1][nc2 - j].destination(
                total_distance / (np2 - i + 1), bearing
            )
            # second half (np2 to p1)
            bearing = pt[np2 + i - 1][nc2 + j].initialBearingTo(pt[np - 1][nc2 + j])
            total_distance = pt[np2 + i - 1][nc2 + j].distanceTo(pt[np - 1][nc2 + j])
            pt[np2 + i][nc2 + j] = pt[np2 + i - 1][nc2 + j].destination(
                total_distance / (np2 - i + 1), bearing
            )
            bearing = pt[np2 + i - 1][nc2 - j].initialBearingTo(pt[np - 1][nc2 - j])
            total_distance = pt[np2 + i - 1][nc2 - j].distanceTo(pt[np - 1][nc2 - j])
            pt[np2 + i][nc2 - j] = pt[np2 + i - 1][nc2 - j].destination(
                total_distance / (np2 - i + 1), bearing
            )

    sw = (
        min(pt[i][j].lat for i in range(np) for j in range(nc)),
        min(pt[i][j].lon for i in range(np) for j in range(nc)),
    )
    ne = (
        max(pt[i][j].lat for i in range(np) for j in range(nc)),
        max(pt[i][j].lon for i in range(np) for j in range(nc)),
    )

    m = folium.Map(
        location=[pt[np2][nc2].lat, pt[np2][nc2].lon]
    )  # , tiles="Stamen Terrain")

    m.fit_bounds([sw, ne])

    for j in range(nc):
        line = []
        for i in range(np):
            line.append([pt[i][j].lat, pt[i][j].lon])
        color = "darkred" if j == nc2 else "yellow"
        folium.PolyLine(locations=line, color=color, no_clip=True).add_to(m)

    for j in range(nc):
        for i in range(np):
            # print(f"({i}, {j})")
            folium.CircleMarker(
                location=(pt[i][j].lat, pt[i][j].lon),
                radius=2,
                color="#3186cc",
                fill=True,
                fill_color="#3186cc",
            ).add_to(m)

    position = [0, 0]
    aircraft_location = pt[position[0]][position[1]]
    aircraft_marker = folium.Marker(
        location=[aircraft_location.lat, aircraft_location.lon],
        icon=folium.Icon(color="red", icon="plane", prefix="fa", angle=0),
    )

    aircraft_marker.add_to(m)

    while position[0] < np - 1:
        # get the next posible state
        current_j = position[1]
        possible_j = [current_j]
        if current_j > 0:
            possible_j.append(current_j - 1)
        if current_j < nc - 2:
            possible_j.append(current_j + 1)

        # get the next position
        position = [position[0] + 1, random.choice(possible_j)]

        aircraft_marker.location = [
            pt[position[0]][position[1]].lat,
            pt[position[0]][position[1]].lon,
        ]

        st_data = st_folium(m, width=800, height=500)
