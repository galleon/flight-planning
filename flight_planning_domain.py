from argparse import Action
import sys
from dataclasses import dataclass
from openlocationcode import openlocationcode as olc
from pybada import AircraftPosition, AircraftState, ControlLaws, EnvironmentState, ISA, g
from pybada.bada4 import BADA4_jet_CR
from typing import List

from pygeodesy import geohash
from pygeodesy.sphericalNvector import LatLon

@dataclass(frozen=True)
class State:
    aircraft_position: AircraftPosition
    aircraft_state: AircraftState


@dataclass(frozen=True)
class AircraftAction(object):
    aircraft_position: AircraftPosition
    aircraft_state: AircraftState


class WeatherEnvironment(object):
    def __init__(self, file: str):
        self.file = file

    def get_wind_speed(self, altitude: int):
        return (0, 0)


class FlightPlanningDomain(object):
    def __init__(
        self,
        initial_state: State,
        final_state: State,
        aircraft_performance_model: BADA4_jet_CR,
        weather_env: WeatherEnvironment = None,
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
        self.weather_env = weather_env

    # CD: Drag coefficient
    # CL: Lift coefficient

    def get_next_state(self, current_state: State, current_action: Action) -> State:

        u_wind, v_wind = self.weather_env.get_wind_speed(current_state.aircraft_position.altitude)

        # Moving from point A (current_state) to point B (given by action)

        delta_t = 0.001

        #if bada_transition_method == BadaTransitionMethod.V1_MANON_AIRBUS:
        #    self.evaluate_function = badatransition_v1.evaluate_transition_bada
        #if bada_transition_method == BadaTransitionMethod.V2_OBJECT_FREE_TRANSITION:
        #    self.evaluate_function = badatransition_v2.evaluate_transition_badani
        #if bada_transition_method == BadaTransitionMethod.V3_POPO:
        #    self.evaluate_function = badatransition_v3.evaluate_transition_bada_objectless

        p0 = LatLon(current_state.aircraft_position.latitude, current_state.aircraft_position.longitude, current_state.aircraft_position.altitude)
        p1 = LatLon(current_action.aircraft_position.latitude, current_action.aircraft_position.longitude, current_action.aircraft_position.altitude)

        total_distance = p0.distanceTo(p1)
        course = p0.bearingTo(p1)


        if self.weather_env is not None:
            temperature = self.weather_env.get_temperature(current_state.aircraft_position.altitude)
        else:
            temperature = ISA().get_temperature(current_state.aircraft_position)

        dh = current_action.aircraft_position.altitude - current_state.aircraft_position.altitude
        controls = None
        dvdh = 0
        if dh > 0:
            # Climb
            controls = ControlLaws(self.apm).get_climb_controls(current_state.environment_state, current_state.aircraft_state, dvdh)

        elif dh < 0:
            # Descent
            controls = ControlLaws(self.apm).get_descent_controls(current_state.environment_state, current_state.aircraft_state, dvdh)
        else:
            # Cruise








    def is_terminal_state(self, state: State) -> bool:
        return state == self.final_state


    def get_next_available_actions(self, current_state: State) -> List[AircraftAction]:

        if is_terminal_state(current_state):
            return []

        # find best local mach & altitude

        # get environment state from ISA model [TODO: Could also be done with a weather simulation]
        environment_state = ISA().get_environment_state(current_state.aircraft_position)
        # TODO: Could also be doe with a weather model
        # pressure = ISA().get_pressure(current_state.aircraft_position)

        # inject wind speed from weather model if any
        environment_state.wind = self.weather_env.get_wind_speed(
            current_state.aircraft_position.altitude
        )

        a = environment_state.get_sound_speed()

        cost_index = 0
        CCI = (
            cost_index
            * self.apm.LHV
            / environment_state.delta
            / (self.apm.MTOW * g)
            / a
        )
        # check coeff from bada tables
        CW = current_state.aircraft_state.m / (
            self.apm.MTOW * environment_state.delta
        )

        try:
            best_mach = self.apm.get_best_mach(CCI, CW)
        except ValueError:
            # no mach available outside range
            raise ValueError("No mach available outside range")

        mach = current_state.aircraft_state.M(current_state.environment_state)
        try:
            best_altitude = self.apm.get_best_altitude(
                current_state.aircraft_state.m, mach
            )
        except ValueError:
            raise ValueError("No altitude available outside range")

        latitudes = []
        longitudes = []
        deltat_mach = 0.1
        machs = [best_mach - deltat_mach, best_mach, best_mach + deltat_mach]
        delta_altitude = 100
        altitudes = [best_altitude - delta_altitude, best_altitude, best_altitude + delta_altitude]

        next_actions = [
            AircraftAction(lat, lon, alt, mach)
            for lat in latitudes
            for lon in longitudes
            for alt in altitudes
            for mach in machs
        ]


if __name__ == "__main__":
    # Initialize the environment
    env = WeatherEnvironment()

    LFBO = (43.6294375, 1.3654988)
    LFPO = (48.7262321, 2.3630535)

    # Flying from LFBO to LFPO (see http://rfinder.asalink.net/free/)
    # ID      FREQ   TRK   DIST   Coords                       Name/Remarks
    # LFBO             0      0   N43°38'06.46" E001°22'04.28" TOULOUSE BLAGNAC
    # FISTO          353     50   N44°27'41.00" E001°13'37.99" FISTO
    # LMG     114.5  354     82   N45°48'56.99" E001°01'31.99" LIMOGES
    # BEVOL          356     72   N47°00'43.00" E000°55'50.99" BEVOL
    # AMB     106.3   11     26   N47°25'44.00" E001°03'52.00" AMBOISE
    # LFPO            33     94   N48°43'23.81" E002°22'46.48" PARIS ORLY

    # Initialize the planning domain
    Hp = 0.5
    departure_time = 0
    arrival_time = 2

    aircraft = BADA4_jet_CR("A320-231")
    weather = None

    initial_state = State(
        AircraftPosition(LFBO[0], LFBO[1], Hp, departure_time), AircraftState(0, 0, 0)
    )
    final_state = State(
        AircraftPosition(LFPO[0], LFPO[1], Hp, arrival_time), AircraftState(0, 0, 0)
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

    domain = FlightPlanningDomain(initial_state, final_state, aircraft, weather)

    # Print the domain
    print(domain, olc.encode(*LFBO), olc.encode(*LFPO))

    print(sys.getsizeof(LFBO), sys.getsizeof(olc.encode(*LFBO)))
