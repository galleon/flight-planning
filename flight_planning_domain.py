import logging
import os
import sys
import warnings
from argparse import Action
from enum import Enum
from logging.handlers import TimedRotatingFileHandler
from time import time
from typing import Any, List, NamedTuple, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
from openap.top import Climb, Cruise, Descent, wind
from openap.top.full import MultiPhase
from openap.top.vis import trajectory_on_map
from pygeodesy.ellipsoidalVincenty import LatLon
from skdecide import DeterministicPlanningDomain, ImplicitSpace, Solver, Space, Value
from skdecide.builders.domain import Actions, Renderable
from skdecide.hub.solver.astar import Astar
from skdecide.hub.solver.stable_baselines import StableBaseline
from skdecide.hub.space.gym import EnumSpace, ListSpace, MultiDiscreteSpace
from skdecide.utils import match_solvers

warnings.filterwarnings("ignore")


class State(NamedTuple):
    trajectory: pd.DataFrame
    pos: Tuple[int, int]

    def __hash__(self):
        return hash(self.pos)

    def __eq__(self, other):
        return self.pos == other.pos


class Action(NamedTuple):
    pos: Tuple[int, int]


class D(DeterministicPlanningDomain, Actions, Renderable):
    T_state = State  # Type of states
    T_observation = State  # Type of observations
    T_event = Action  # Type of events
    T_value = float  # Type of transition values (rewards or costs)
    T_predicate = bool  # Type of transition predicates (terminal states)
    T_info = None  # Type of additional information in environment outcome
    T_agent = Union  # Type of agent


class FlightPlanningDomain(D):
    def __init__(
        self,
        origin: Union[str, tuple],
        destination: Union[str, tuple],
        actype: str,
        m0: float = 0.8,
        windfield: pd.DataFrame = None,
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

        logging.info(f"origin: {origin}, destination: {destination}, actype: {actype}")
        # Define optimizers
        self.cruise = Cruise(actype, origin, destination, m0)
        self.climb = Climb(actype, origin, destination, m0)
        self.descent = Descent(actype, origin, destination, m0)

        self.multiphase = MultiPhase(actype, origin, destination, m0)

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
            self.multiphase.wind = w

        # Find the approximate cruise altitude
        tick = time()
        dfcr = self.cruise.trajectory(self.objective[1])
        logging.info(f"Cruise trajectory computed in {time() - tick:.2f} seconds")

        # Run multiphase flight
        # tick = time()
        # dfmp = self.multiphase.trajectory(self.objective)
        # logging.info(f"Multiphase trajectory computed in {time() - tick:.2f} seconds")
        # dfmp.to_csv(f"multiphase_{origin}_{destination}.csv")

        # Find optimal climb trajectory
        # tick = time()
        self.dfcl = self.climb.trajectory(self.objective[0], dfcr)
        logging.info(f"Climb trajectory computed in {time() - tick:.2f} seconds")

        self.start = State(self.dfcl, (0, 0))

        # Build network
        self.np: int = 41
        self.nc: int = 11
        self.network = self.get_network(
            LatLon(self.dfcl.lat.iloc[-1], self.dfcl.lon.iloc[-1]),
            LatLon(self.cruise.lat2, self.cruise.lon2),
            self.np,
            self.nc,
        )
        logger.info(f"Constructor initialized in {time() - tick:.2f} seconds")

    def _get_next_state(self, memory: D.T_state, action: D.T_event) -> D.T_state:
        """
        Compute the next state from:
          - memory: the current state
          - action: the action to take
        """
        trajectory = memory.trajectory.copy()

        self.cruise.initial_mass = memory.trajectory.mass.iloc[-1]
        self.cruise.lat1 = memory.trajectory.lat.iloc[-1]
        self.cruise.lon1 = memory.trajectory.lon.iloc[-1]
        # Extract current time to set wind conditions
        ts = trajectory.ts.iloc[-1]
        # Update wind information
        # self.cruise.wind =

        # Set intermediate destination point
        node = action.pos
        self.cruise.lat2 = self.network[node[0]][node[1]].lat
        self.cruise.lon2 = self.network[node[0]][node[1]].lon
        # Set number of colocation points
        self.cruise.setup_dc(nodes=5)
        # Compute the optimal trajectory
        dfcr = self.cruise.trajectory(self.objective[1])
        # Concatenate the two trajectories

        dfcr.ts = memory.trajectory.ts.iloc[-1] + dfcr.ts

        state = State(
            pd.concat([memory.trajectory, dfcr], ignore_index=True),
            action.pos,
        )
        logging.info(f"Next state: {state}")
        return state

    def _get_transition_value(
        self,
        memory: D.T_state,
        action: D.T_event,
        next_state: Optional[D.T_state] = None,
    ) -> Value[D.T_value]:
        """
        Get the value (reward or cost) of a transition.

        Set cost to 1 when moving (energy cost)
        and to 2 when bumping into a wall (damage cost).
        """
        self.cruise.initial_mass = memory.trajectory.mass.iloc[-1]
        self.cruise.lat1 = memory.trajectory.lat.iloc[-1]
        self.cruise.lon1 = memory.trajectory.lon.iloc[-1]
        # Extract current time to set wind conditions
        ts = memory.trajectory.ts.iloc[-1]
        # Update wind information
        # self.cruise.wind = None
        node = action.pos
        self.cruise.lat2 = self.network[node[0]][node[1]].lat
        self.cruise.lon2 = self.network[node[0]][node[1]].lon
        # Set number of colocation points
        self.cruise.setup_dc(nodes=5)
        # Compute the optimal trajectory
        dfcr = self.cruise.trajectory(self.objective[1])
        # Extract last row as a dataframe
        logging.info(dfcr.iloc[-1:])

        return Value(cost=1)

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
        return ListSpace([(self.np - 1, j) for j in range(self.nc)])

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

        next_nodes = []
        if x == 0:
            for j in range(self.nc):
                next_nodes.append(Action((x + 1, j)))
        elif x < self.np - 1:
            if y + 1 < self.nc:
                next_nodes.append(Action((x + 1, y + 1)))
            next_nodes.append(Action((x + 1, y)))
            if y > 0:
                next_nodes.append(Action((x + 1, y - 1)))

        return ListSpace(next_nodes)

    def _get_action_space_(self) -> Space[D.T_event]:
        """
        Define action space.
        """
        return ImplicitSpace(lambda x: isinstance(x, State))

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
        # store used matplotlib subplot and image to only update them afterwards
        plt.ioff()
        fig, ax = plt.subplots(1)
        ax.set_aspect("equal")  # set the x and y axes to the same scale
        plt.xticks([])  # remove the tick marks by setting to an empty list
        plt.yticks([])  # remove the tick marks by setting to an empty list
        ax.invert_yaxis()  # invert the y-axis so the first row of data is at the top
        plt.ion()
        fig.canvas.header_visible = False
        fig.canvas.footer_visible = False
        fig.canvas.resizable = False
        fig.set_dpi(1)
        fig.set_figwidth(600)
        fig.set_figheight(600)
        # if image is None:
        #     image = ax.imshow(image_data)
        # else:
        #     image.set_data(image_data)
        #     image.figure.canvas.draw()
        return ax, None

    def heuristic(self, s: D.T_state) -> Value[D.T_value]:
        """Heuristic to be used by search algorithms.

        Here Euclidean distance to goal.

        """
        return Value(cost=0)

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


if __name__ == "__main__":
    # Initialize the environment

    logger = logging.getLogger()
    fhandler = TimedRotatingFileHandler(f"{__name__}.log", when="midnight")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)

    origin = "LFPG"
    destination = "WSSS"
    #
    tick = time()
    fgrib = "data/adaptor.mars.internal-1659588957.360821-19344-9-b2135488-9725-486e-b3d2-adf40cb68242.grib"
    windfield = wind.read_grib(fgrib)
    logging.info(f"Wind field read in {time() - tick:.2f} seconds")

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

    domain_factory = lambda: FlightPlanningDomain(
        "LFPG", "WSSS", "A388", windfield=windfield
    )

    logging.info("Creating domain")
    domain = domain_factory()

    logging.info("Looking for compatible solvers")
    candidate_solvers = match_solvers(domain=domain)
    logging.info("Found {} compatible solvers".format(len(candidate_solvers)))
    for solver in candidate_solvers:
        logging.info(solver)

    solver = Astar(heuristic=lambda d, s: d.heuristic(s))
    from skdecide.utils import rollout_episode

    a = rollout_episode(domain, from_memory=domain.get_initial_state())
    logging.info(a)

    # FlightPlanningDomain.solve_with(solver, domain_factory)
