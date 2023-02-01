import warnings
from argparse import Action
from datetime import datetime, timedelta
from enum import Enum
from time import sleep
from typing import Any, NamedTuple, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from flightplanning_utils import (
    WeatherRetrieverFromEcmwf,
    WindInterpolator,
    flying,
    plot_trajectory,
)
from IPython.display import clear_output
from openap.extra.aero import distance
from openap.extra.nav import airport
from openap.fuel import FuelFlow
from openap.prop import aircraft
from pygeodesy.ellipsoidalVincenty import LatLon
from skdecide import DeterministicPlanningDomain, Domain, Space, Value
from skdecide.builders.domain import Renderable, UnrestrictedActions
from skdecide.hub.solver.astar import Astar
from skdecide.hub.space.gym import EnumSpace, ListSpace, MultiDiscreteSpace
from skdecide.utils import match_solvers


class State:
    trajectory: pd.DataFrame
    pos: Tuple[int, int]

    def __init__(self, trajectory, pos):
        self.trajectory = trajectory
        self.pos = pos

    def __hash__(self):
        return hash(self.pos)

    def __eq__(self, other):
        return self.pos == other.pos

    def __ne__(self, other):
        return self.pos != other.pos

    def __str__(self):
        return f"[{self.trajectory.iloc[-1]['ts']:.2f} \
            {self.pos} \
            {self.trajectory.iloc[-1]['alt']:.2f} \
            {self.trajectory.iloc[-1]['mass']:.2f}]"


class Action(Enum):
    up = -1
    straight = 0
    down = 1


class D(DeterministicPlanningDomain, UnrestrictedActions, Renderable):
    T_state = State  # Type of states
    T_observation = State  # Type of observations
    T_event = Action  # Type of events
    T_value = float  # Type of transition values (rewards or costs)
    T_predicate = bool  # Type of transition predicates (terminal states)
    T_info = None  # Type of additional information in environment outcome
    T_agent = Union  # Type of agent


class FlightPlanningDomain(
    DeterministicPlanningDomain, UnrestrictedActions, Renderable
):
    T_state = State  # Type of states
    T_observation = State  # Type of observations
    T_event = Action  # Type of events
    T_value = float  # Type of transition values (rewards or costs)
    T_predicate = bool  # Type of transition predicates (terminal states)
    T_info = None  # Type of additional information in environment outcome
    T_agent = Union  # Type of agent

    def __init__(
        self,
        origin: Union[str, tuple],
        destination: Union[str, tuple],
        actype: str,
        m0: float = 0.8,
        wind_interpolator: WindInterpolator = None,
        objective: Union[str, tuple] = "fuel",
        nb_points_forward: int = 41,
        nb_points_lateral: int = 11,
    ):
        """A simple class to compute a flight plan.

        Parameters
        ----------
        origin: Union[str, tuple])
            ICAO or IATA code of airport, or tuple (lat, lon)
        destination: Union[str, tuple]
            ICAO or IATA code of airport, or tuple (lat, lon)
        aircraft : Aircraft
            Describe the aircraft.
        windfield: pd.DataFrame
            Wind field data. Defaults to None.
        objective: str
            The objective of the flight. Defaults to "fuel".
            for climb, cruise and descent.
        """

        if isinstance(origin, str):
            ap1 = airport(origin)
            self.lat1, self.lon1 = ap1["lat"], ap1["lon"]
        else:
            self.lat1, self.lon1 = origin

        if isinstance(destination, str):
            ap2 = airport(destination)
            self.lat2, self.lon2 = ap2["lat"], ap2["lon"]
        else:
            self.lat2, self.lon2 = destination
        #
        self.objective = objective

        self.wind_ds = None
        if wind_interpolator:
            self.wind_ds = wind_interpolator.get_dataset()

        # Build network between top of climb and destination airport
        self.np: int = nb_points_forward
        self.nc: int = nb_points_lateral
        self.network = self.get_network(
            LatLon(self.lat1, self.lon1),
            LatLon(self.lat2, self.lon2),
            self.np,
            self.nc,
        )

        ac = aircraft(actype)
        self.start = State(
            pd.DataFrame(
                [
                    {
                        "ts": 0,
                        "lat": self.lat1,
                        "lon": self.lon1,
                        "mass": m0 * ac["limits"]["MTOW"],
                        "mach": ac["cruise"]["mach"],
                        "fuel": 0.0,
                        "alt": ac["cruise"]["height"],
                    }
                ]
            ),
            (0, self.nc // 2),
        )
        self.fuel_flow = FuelFlow(actype).enroute

    def _get_next_state(self, memory: D.T_state, action: D.T_event) -> D.T_state:
        """
        Compute the next state from:
          - memory: the current state
          - action: the action to take
        """

        trajectory = memory.trajectory.copy()

        # Set intermediate destination point
        next_x, next_y = memory.pos

        next_x += 1

        if action == Action.up:
            next_y += 1
        if action == Action.down:
            next_y -= 1

        # Aircraft stays on the network
        if next_x >= self.np or next_y < 0 or next_y >= self.nc:
            return memory

        # Concatenate the two trajectories

        to_lat = self.network[next_x][next_y].lat
        to_lon = self.network[next_x][next_y].lon
        trajectory = flying(
            trajectory.tail(1), (to_lat, to_lon), self.wind_ds, self.fuel_flow
        )

        state = State(
            pd.concat([memory.trajectory, trajectory], ignore_index=True),
            (next_x, next_y),
        )
        return state

    def _get_transition_value(
        self,
        memory: D.T_state,
        action: D.T_event,
        next_state: Optional[D.T_state] = None,
    ) -> Value[D.T_value]:
        """
        Get the value (reward or cost) of a transition.

        Set cost to distance travelled between points
        """

        assert memory != next_state, "Next state is the same as the current state"
        # Have to change -> openAP top ?
        if self.objective == "distance":
            cost = distance(
                memory.trajectory.iloc[-1]["lat"],
                memory.trajectory.iloc[-1]["lon"],
                next_state.trajectory.iloc[-1]["lat"],
                next_state.trajectory.iloc[-1]["lon"],
            )
        elif self.objective == "fuel":
            cost = (
                memory.trajectory.iloc[-1]["mass"]
                - next_state.trajectory.iloc[-1]["mass"]
            )
        elif self.objective == "time":
            cost = (
                next_state.trajectory.iloc[-1]["mass"]
                - memory.trajectory.iloc[-1]["ts"]
            )
        # return Value(cost=1)
        return Value(cost=cost)

    def _get_initial_state_(self) -> D.T_state:
        """
        Get the initial state.

        Set the start position as initial state.
        """
        return self.start

    def _get_goals_(self) -> Space[D.T_observation]:
        """
        Get the domain goals space (finite or infinite set).

        Set the end position as goal.
        """
        return ListSpace([State(None, (self.np - 1, j)) for j in range(self.nc)])

    def _is_terminal(self, state: State) -> D.T_predicate:
        """
        Indicate whether a state is terminal.

        Stop an episode only when goal reached.
        """
        return state.pos[0] == self.np - 1

    def _get_applicable_actions_from(self, memory: D.T_state) -> Space[D.T_event]:
        """
        Get the applicable actions from a state.
        """
        x, y = memory.pos

        space = []
        if x < self.np - 1:
            space.append(Action.straight)
            if y + 1 < self.nc:
                space.append(Action.up)
            if y > 0:
                space.append(Action.down)

        return ListSpace(space)

    def _get_action_space_(self) -> Space[D.T_event]:
        """
        Define action space.
        """
        return EnumSpace(Action)

    def _get_observation_space_(self) -> Space[D.T_observation]:
        """
        Define observation space.
        """
        return MultiDiscreteSpace([self.np, self.nc])

    def _render_from(self, memory: State, **kwargs: Any) -> Any:
        """
        Render visually the map.

        Returns:
            matplotlib figure
        """

        return plot_trajectory(
            self.lat1, self.lon1, self.lat2, self.lon2, memory.trajectory, self.wind_ds
        )

    def heuristic(self, s: D.T_state) -> Value[D.T_value]:
        """Heuristic to be used by search algorithms.

        Here fuel consumption to reach target.

        """
        lat = s.trajectory.iloc[-1]["lat"]
        lon = s.trajectory.iloc[-1]["lon"]
        # Compute distance in meters
        distance_to_goal = distance(lat, lon, self.lat2, self.lon2)
        cost = distance_to_goal

        return Value(cost=cost)

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
