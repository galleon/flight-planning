import sys
from argparse import Action
from dataclasses import dataclass
from datetime import datetime, timedelta
import random as rnd
from typing import List, Tuple

import networkx as nx
from openlocationcode import openlocationcode as olc
from pygeodesy import geohash
from pygeodesy.ellipsoidalVincenty import LatLon

from pybada.bada4 import (
    ISA,
    AircraftPosition,
    AircraftState,
    BADA4_jet_CR,
    ControlLaws,
    EnvironmentState,
    g,
    ft2m,
)

from pygeodesy.ellipsoidalVincenty import LatLon


from weather.weather import WeatherModel


@dataclass(frozen=True)
class State:
    aircraft_position: AircraftPosition
    aircraft_state: AircraftState
    node_id: Tuple[int, int]


# @dataclass(frozen=True)
# class AircraftAction(object):
#    aircraft_position: AircraftPosition
#    aircraft_state: AircraftState


def get_data(p0: LatLon, p1: LatLon, np: int, nc: int):
    np2 = np // 2
    nc2 = nc // 2

    distp = 10 * p0.distanceTo(p1) / np / nc  # meters

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
    return pt


class FlightPlanningDomain(object):
    def __init__(
        self,
        initial_state: State,
        final_state: State,
        aircraft_performance_model: BADA4_jet_CR,
        weather_model: WeatherModel = None,
        cost_index: float = 1.0,
        pt: List[List[LatLon]] = None,
    ):
        """A simple class to compute a flight plan.

        Parameters
        ----------
        initiall_state : State
            The initial state of the planning problem.
        final_state : State
            The goal state of the planning problem.
        aircraft : BADA4_jet_CR
            The BADA aircraft model.
        """
        self.initial_state = initial_state
        self.final_state = final_state
        self.apm = aircraft_performance_model
        self.weather_model = weather_model
        self.cost_index = cost_index
        assert pt is not None
        self.pt = pt

    # CD: Drag coefficient
    # CL: Lift coefficient

    def get_next_state(self, current_state: State, current_action: Action) -> State:

        u_wind, v_wind = self.weather_env.get_wind_speed(
            current_state.aircraft_position
        )

        # Moving from point A (current_state) to point B (given by action)

        delta_t = 0.001

        # if bada_transition_method == BadaTransitionMethod.V1_MANON_AIRBUS:
        #    self.evaluate_function = badatransition_v1.evaluate_transition_bada
        # if bada_transition_method == BadaTransitionMethod.V2_OBJECT_FREE_TRANSITION:
        #    self.evaluate_function = badatransition_v2.evaluate_transition_badani
        # if bada_transition_method == BadaTransitionMethod.V3_POPO:
        #    self.evaluate_function = badatransition_v3.evaluate_transition_bada_objectless

        p0 = LatLon(
            current_state.aircraft_position.latitude,
            current_state.aircraft_position.longitude,
            current_state.aircraft_position.altitude,
        )
        p1 = LatLon(
            current_action.aircraft_position.latitude,
            current_action.aircraft_position.longitude,
            current_action.aircraft_position.altitude,
        )

        total_distance = p0.distanceTo(p1)
        course = p0.bearingTo(p1)

        if self.weather_model is not None:
            temperature = self.weather_model.get_temperature(
                current_state.aircraft_position.altitude
            )
        else:
            temperature = ISA().get_temperature(current_state.aircraft_position)

        dh = (
            current_action.aircraft_position.altitude
            - current_state.aircraft_position.altitude
        )
        controls = None
        dvdh = 0
        if dh > 0:
            # Climb
            controls = ControlLaws(self.apm).get_climb_controls(
                current_state.environment_state, current_state.aircraft_state, dvdh
            )
        elif dh < 0:
            # Descent
            controls = ControlLaws(self.apm).get_descent_controls(
                current_state.environment_state, current_state.aircraft_state, dvdh
            )
        else:
            # Cruise
            controls = ControlLaws(self.apm).get_cruise_controls(
                current_state.environment_state, current_state.aircraft_state
            )

        v_dot = self.apm.v_dot(
            current_state.environment_state, current_state.aircraft_state, controls
        )
        m_dot = self.apm.m_dot(
            current_state.environment_state, current_state.aircraft_state, controls
        )

    def is_terminal_state(self, state: State) -> bool:
        # Did we reach the end of the graph?
        return state.node_id[0] == self.final_state.node_id[0]

    def get_next_available_actions(self, current_state: State) -> List[State]:

        print(f"State: {current_state}")

        x0, y0 = initial_state.node_id
        x1, y1 = final_state.node_id

        x, y = current_state.node_id

        if self.is_terminal_state(current_state):
            print("We are done !!!")
            return []

        # find best local mach & altitude

        # get environment state from ISA model [TODO: Could also be done with a weather simulation]
        environment_state = ISA().get_environment_state(current_state.aircraft_position)
        # TODO: Could also be doe with a weather model
        # pressure = ISA().get_pressure(current_state.aircraft_position)

        # inject wind speed from weather model if any
        environment_state.wind = self.weather_model.get_wind_speed(
            current_state.aircraft_position
        )

        a = environment_state.get_sound_speed()
        print(f"speed of sound: {a}")
        print(f"LHV: {self.apm.LHV}")
        print(f"delta: {environment_state.delta()}")
        print(f"MTOW: {self.apm.MTOW}")

        CCI = (
            self.cost_index
            * self.apm.LHV
            / environment_state.delta()
            / (self.apm.MTOW * g)
            / a
        )
        # check coeff from bada tables
        CW = current_state.aircraft_state.m / (
            self.apm.MTOW * environment_state.delta()
        )

        print(f"CCI: {CCI} CW: {CW}")
        # CCI in [-0.000503, 2.161215]
        # CW  in [ 1.097560, 5.351188]
        try:
            best_mach = self.apm.best_mach(CCI, CW)
        except ValueError:
            # no mach available outside range
            raise ValueError("No mach available outside range")

        print(f"best mach: {best_mach}")

        mach = current_state.aircraft_state.M(environment_state)
        try:
            best_altitude = self.apm.best_alt(current_state.aircraft_state.m, mach)
        except ValueError:
            raise ValueError("No altitude available outside range")

        print(f"best alt: {best_altitude}")

        best_altitude_m = best_altitude * ft2m

        next_nodes = []
        if y + 1 < y1:
            next_nodes.append((x + 1, y + 1))
        next_nodes.append((x + 1, y))
        if y > y0:
            next_nodes.append((x + 1, y - 1))

        deltat_mach = 0.1
        machs = [best_mach - deltat_mach, best_mach, best_mach + deltat_mach]
        delta_altitude = 100
        altitudes = [
            best_altitude_m - delta_altitude,
            best_altitude_m,
            best_altitude_m + delta_altitude,
        ]

        print(next_nodes)

        # create new states
        next_actions = [
            State(
                AircraftPosition(
                    self.pt[node[0]][node[1]].lat, self.pt[node[0]][node[1]].lon, alt, 0
                ),
                AircraftState(true_air_speed, heading, flight_path_angle, mass),
                (node[0], node[1]),
            )
            for node in next_nodes
            for alt in altitudes
            for mach in machs
        ]

        return next_actions


if __name__ == "__main__":
    # Initialize the environment

    LFBO = (43.6294375, 1.3654988)
    LFPO = (48.7262321, 2.3630535)

    np: int = 11  # number of points along a trajectory
    nc: int = 11  # number of trajectories

    Hp = 10000  # m

    p0: LatLon = LatLon(*LFBO, Hp)
    p1: LatLon = LatLon(*LFPO, Hp)

    # Flying from LFBO to LFPO (see http://rfinder.asalink.net/free/)
    # ID      FREQ   TRK   DIST   Coords                       Name/Remarks
    # LFBO             0      0   N43°38'06.46" E001°22'04.28" TOULOUSE BLAGNAC
    # FISTO          353     50   N44°27'41.00" E001°13'37.99" FISTO
    # LMG     114.5  354     82   N45°48'56.99" E001°01'31.99" LIMOGES
    # BEVOL          356     72   N47°00'43.00" E000°55'50.99" BEVOL
    # AMB     106.3   11     26   N47°25'44.00" E001°03'52.00" AMBOISE
    # LFPO            33     94   N48°43'23.81" E002°22'46.48" PARIS ORLY

    # Initialize the planning domain

    aircraft = BADA4_jet_CR("A320-231")

    now = datetime.now() - timedelta(days=1)
    print(f"Current date and time {now}")

    # Initial conditions
    # K[T|I|C]AS = Knots [True | Indicated | Calibrated] Air Speed
    true_air_speed = 225  # ms^-1
    heading = 0
    flight_path_angle = 0
    mass = aircraft.MTOW * 0.8  # 80% of MTOW

    print(
        f"Initial conditions:{p0.lat} {p0.lon} {p0.height} {true_air_speed} {heading} {flight_path_angle} {mass}"
    )

    initial_state = State(
        AircraftPosition(p0.lat, p0.lon, p0.height, 0),
        AircraftState(true_air_speed, heading, flight_path_angle, mass),
        (0, 0),
    )
    final_state = State(
        AircraftPosition(p1.lat, p1.lon, p1.height, None),
        AircraftState(0, 0, 0, 0),
        (np, nc),
    )

    # A380-841
    # Departure: LFPG (N49° 00' 36", E2° 32' 55")
    # Arrival: KJFK (N40° 38' 26", W73° 46' 44")
    # 10 June 2017 Wind NOAA Nowcast - delta_time: 0.1 hour
    # Altitude: 35000 ft
    # Mach: 0.86
    # Time: 0
    # Bank: 0
    # Track: 0
    # Mass: mass_properties.OEW + 50000 + 76300 + (i-2)*1000 for i in range(5)
    # {}
    # fuel_mass = mass_properties.OEW
    # empty_mass = 50000
    # payload_mass = 76300 + (i-2)*1000 for i in range(5)

    # Heuristiques ? fuel = distance_restante_grand_cercle * mean_consumption_fuel

    weather_model = WeatherModel(now)

    domain = FlightPlanningDomain(
        initial_state, final_state, aircraft, weather_model, pt=get_data(p0, p1, np, nc)
    )

    current_state = initial_state
    while not domain.is_terminal_state(current_state):
        print(
            f"At time: {weather_model.dof + timedelta(seconds=current_state.aircraft_position.t)}"
        )
        print(
            f"Aircraft position: {current_state.aircraft_position.lat:2.2f} {current_state.aircraft_position.lon:2.2f} {current_state.aircraft_position.Hp}"
        )
        actions = domain.get_next_available_actions(current_state)

        actions = rnd.choices(actions, k=10)
