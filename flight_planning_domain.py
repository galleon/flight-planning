import sys
import warnings

from argparse import Action
from dataclasses import dataclass
from typing import List, Tuple, Union

import pandas as pd
from openap.top import Climb, Cruise, Descent, wind
from pygeodesy.ellipsoidalVincenty import LatLon

warnings.filterwarnings("ignore")

# @dataclass(frozen=True)
# class AircraftAction(object):
#    aircraft_position: AircraftPosition
#    aircraft_state: AircraftState


@dataclass
class State(object):
    trajectory: pd.DataFrame
    pos: Tuple[int, int]


def get_cost(state: State, action: State):
    return state.aircraft_position.distanceTo(action.aircraft_position)


class FlightPlanningDomain(object):
    def __init__(
        self,
        origin: Union[str, tuple],
        destination: Union[str, tuple],
        actype: str,
        m0: float = 0.8,
        winfield: pd.DataFrame = None,
        objective: Union[str, tuple] = "fuel",
    ):
        """A simple class to compute a flight plan.

        Parameters
        ----------
        origin: Union[str, tuple])
            ICAO or IATA code of airport, or tuple (lat, lon)
        destination: Union[str, tuple]
            ICAO or IATA code of airport, or tuple (lat, lon)
        aircraft : str
            The OpenAP aircraft type.
        m0: float
            Takeoff mass factor. Defaults to 0.8 (of MTOW)
        windfield: pd.DataFrame
            Wind field data. Defaults to None.
        objective: Union[str, tuple]
            The objective of the flight. Defaults to "fuel". Can be a tuple if different objective
            for climb, cruise and descent.
        """
        self.origin = origin
        self.destination = destination
        #
        if isinstance(objective, str):
            self.objective = (objective, objective, objective)
        else:
            self.objective = objective

        print(f"origin: {origin}, destination: {destination}, actype: {actype}")
        # Define optimizers
        self.cruise = Cruise(actype, origin, destination, m0)
        self.climb = Climb(actype, origin, destination, m0)
        self.descent = Descent(actype, origin, destination, m0)

        if windfield is not None:
            w = wind.PolyWind(
                windfield,
                self.cruise.proj,
                self.cruise.lat1,
                self.cruise.lon1,
                self.cruise.lat2,
                self.cruise.lon2,
            )
            self.cruise.wind = w
            self.climb.wind = w
            self.descent.wind = w

        # Find the approximate cruise altitude
        dfcr = self.cruise.trajectory(self.objective[1])

        # Find optimal climb trajectory
        self.dfcl = self.climb.trajectory(self.objective[0], dfcr)

        # Build network
        self.np: int = 41
        self.nc: int = 11
        self.network = self.get_network(
            LatLon(self.dfcl.lat.iloc[-1], self.dfcl.lon.iloc[-1]),
            LatLon(self.cruise.lat2, self.cruise.lon2),
            self.np,
            self.nc,
        )

    def get_initial_state(self):
        return State(self.dfcl, (0, 0))

    def get_network(self, p0: LatLon, p1: LatLon, np: int, nc: int):
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
                total_distance = pt[np2 + i - 1][nc2 + j].distanceTo(
                    pt[np - 1][nc2 + j]
                )
                pt[np2 + i][nc2 + j] = pt[np2 + i - 1][nc2 + j].destination(
                    total_distance / (np2 - i + 1), bearing
                )
                bearing = pt[np2 + i - 1][nc2 - j].initialBearingTo(pt[np - 1][nc2 - j])
                total_distance = pt[np2 + i - 1][nc2 - j].distanceTo(
                    pt[np - 1][nc2 - j]
                )
                pt[np2 + i][nc2 - j] = pt[np2 + i - 1][nc2 - j].destination(
                    total_distance / (np2 - i + 1), bearing
                )
        return pt

    def is_terminal_state(self, state: State) -> bool:
        # Did we reach the end of the graph?
        return state.pos[0] == self.np - 1

    def get_next_available_actions(self, current_state: State) -> List[State]:
        # x0, y0 = self.initial_state.node_id
        # x1, y1 = self.final_state.node_id

        x, y = current_state.pos

        if self.is_terminal_state(current_state):
            print("We are done !!!")
            return []

        next_nodes = []
        if x == 0:
            for j in range(self.nc):
                next_nodes.append((x + 1, j))
        else:
            if y + 1 < self.nc:
                next_nodes.append((x + 1, y + 1))
            next_nodes.append((x + 1, y))
            if y > 0:
                next_nodes.append((x + 1, y - 1))

        next_actions = []
        for node in next_nodes:
            self.cruise.initial_mass = current_state.trajectory.mass.iloc[-1]
            self.cruise.lat1 = current_state.trajectory.lat.iloc[-1]
            self.cruise.lon1 = current_state.trajectory.lon.iloc[-1]
            ts = current_state.trajectory.ts.iloc[-1]
            # Update wind information
            # self.cruise.wind =
            self.cruise.lat2 = self.network[node[0]][node[1]].lat
            self.cruise.lon2 = self.network[node[0]][node[1]].lon
            dfcr = self.cruise.trajectory(self.objective[1])
            next_actions.append(State(dfcr, node))

        return next_actions

    def get_next_state(self, current_state: State, current_action: Action) -> State:
        current_action.trajectory.ts = (
            current_state.trajectory.ts.iloc[-1] + current_action.trajectory.ts
        )
        state = State(
            pd.concat(
                [current_state.trajectory, current_action.trajectory], ignore_index=True
            ),
            current_action.pos,
        )
        return state

    def get_best_action(self, actions: List[Action]) -> Action:
        best_action = actions[0]
        best_objective = sys.float_info.max
        for action in actions:
            if action.trajectory[self.objective[1]].iloc[-1] <= best_objective:
                best_objective = action.trajectory[self.objective[1]].iloc[-1]
                best_action = action

        return best_action


if __name__ == "__main__":
    # Initialize the environment

    origin = "LFPG"
    destination = "WSSS"
    # 1st May 2021 8am
    fgrib = "data/adaptor.mars.internal-1659588957.360821-19344-9-b2135488-9725-486e-b3d2-adf40cb68242.grib"
    windfield = wind.read_grib(fgrib)

    # Flying from LFBO to LFPO (see http://rfinder.asalink.net/free/)
    # ID      FREQ   TRK   DIST   Coords                       Name/Remarks
    # LFBO             0      0   N43°38'06.46" E001°22'04.28" TOULOUSE BLAGNAC
    # FISTO          353     50   N44°27'41.00" E001°13'37.99" FISTO
    # LMG     114.5  354     82   N45°48'56.99" E001°01'31.99" LIMOGES
    # BEVOL          356     72   N47°00'43.00" E000°55'50.99" BEVOL
    # AMB     106.3   11     26   N47°25'44.00" E001°03'52.00" AMBOISE
    # LFPO            33     94   N48°43'23.81" E002°22'46.48" PARIS ORLY

    # Initialize the planning domain

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

    domain = FlightPlanningDomain("LFPG", "WSSS", "A388", winfield=windfield)

    current_state = domain.get_initial_state()
    while not domain.is_terminal_state(current_state):
        actions = domain.get_next_available_actions(current_state)

        action = domain.get_best_action(actions)

        current_state = domain.get_next_state(current_state, action)

    current_state.trajectory.to_csv(f"data/trajectory_{origin}_{destination}.csv")
