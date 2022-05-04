import math as m
import os
import sys
import xml.etree.ElementTree
from enum import Enum
import numpy as np

from os import environ as osenv

from pygeodesy import geohash
from pygeodesy.sphericalNvector import LatLon


class GreatCircleNetwork:
    def __init__(self, pt1, pt2, forward=100.0, lateral=50.0, n_circles=20) -> None:
        radius = 6371008.77141
        # distance between two points on great circle
        p = LatLon(*pt1)
        distance = p.distanceTo(
            LatLon(
                *pt2,
            )
        )
        # number of points on great circle
        nsteps = int(2 * m.ceil(distance / (2 * forward)))

        # Locate destination point
        flight_level = 330
        height = flight_level * 100
        # pd = p.destination(distance, radius, height)

        print(distance, nsteps)

    def get_next_points(self, p):
        return []


class Environment:
    """Atmospheric environment implementing International Standard Atmosphere"""

    a0: float = 340.29
    P_MSL: float = 101325.0
    T_MSL: float = 288.15
    rho_MSL: float = 1.225
    R: float = 287.05287
    g: float = 9.80665
    h_tropopause: float = 11000.0
    lapse_rate_tropopause: float = -0.0065

    def __init__(self, deltaT=0):
        # constants
        self.deltaT = deltaT

        z = np.arange(0, 20500, 500)
        T = np.maximum(
            Environment.T_MSL + Environment.lapse_rate_tropopause * z,
            216.65 * np.ones_like(z),
        )

        P_tropo = Environment.P_MSL * np.exp(
            -Environment.g / (Environment.R * T) * (z - Environment.h_tropopause)
        )

    def get_environment(self, Hp):
        """
        Returns the atmospheric parameters at a given altitude.
        """
        z = Hp + self.deltaT
        return (
            self.T_MSL + self.lapse_rate_tropopause * z,
            self.P_MSL
            * np.exp(-self.g / (self.R * self.T_MSL) * (z - self.h_tropopause)),
        )


class ConfigurationParameters:
    pass


class SimulationInput:
    pass


class TerminationCondition:
    pass


class ClimbAltitudeTerminationCondition(TerminationCondition):
    pass


class DescentAltitudeTerminationCondition(TerminationCondition):
    pass


class DistanceTerminationCondition(TerminationCondition):
    pass


class TimeTerminationCondition(TerminationCondition):
    pass


class Airspeed:
    def getCAS(self, altitude):
        pass

    def getTAS(self, altitude):
        pass

    def getMach(self, altitude):
        pass


class AircraftModel:
    type: str
    id: str


class FlightPhase(Enum):
    CRUISE = 0


class AMB4(AircraftModel):
    def __init__(self, label: str = "A320-231"):
        try:
            BADA4_PATH = osenv.get("BADA4_PATH", "data")
        except KeyError:
            raise KeyError(
                "Please set the environment variable BADA4_PATH to the path of the BADA4 installation."
            )

        full_path = os.path.join(BADA4_PATH, f"{label}.xml")

        xml_root = xml.etree.ElementTree.parse(full_path).getroot()

        self.label = xml_root.find("model").text
        print(self.label)

    def v_dot(self, v, t):
        pass

    def m_dot(self, m, t):
        pass

    def getCAS(self, altitude: int):
        pass

    def getTrust(self):
        thrust: float = 0.0
        # if phase == FlightPhase.CRUISE:
        #     thrust = drag + mass*acceleration_target
        #     maxThrust = compute_Th(airState, Rating.MCRZ)
        #     minThrust = compute_Th(airState, Rating.LIDL)
        #     if thrust > maxThrust:
        #         thrust = maxThrust
        #         acceleration_target = (thrust - drag)/mass
        #     if thrust < minThrust:
        #         thrust = minThrust
        #         acceleration_target = (thrust - drag)/mass
        # else:
        #     thrust = compute_Th(airState)

        return thrust


if __name__ == "__main__":
    LFBO = (43.6294375, 1.3654988)
    LFPO = (48.7262321, 2.3630535)

    print(
        geohash.encode(*LFBO), sys.getsizeof(LFBO), sys.getsizeof(geohash.encode(*LFBO))
    )

    p = LatLon(*LFBO)
    radius = 6371008.77141
    print(p.distanceTo(LatLon(*LFPO, radius)))

    gcn = GreatCircleNetwork(LFBO, LFPO)

    aircraft = AMB4()
