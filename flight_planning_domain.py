import logging
import os
import sys
import warnings
from argparse import Action
from enum import Enum
from logging.handlers import TimedRotatingFileHandler
from math import pi
from time import sleep, time
from typing import Any, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import openap
import openap.casadi as oc
import pandas as pd
from cartopy import crs as ccrs
from cartopy.feature import BORDERS, LAND, OCEAN
from matplotlib.figure import Figure
from openap.top import Climb, Cruise, Descent, wind
from openap.top.full import MultiPhase
from openap import aero
from pygeodesy.ellipsoidalVincenty import LatLon
from skdecide import DeterministicPlanningDomain, ImplicitSpace, Solver, Space, Value
from skdecide.builders.domain import Actions, Renderable, UnrestrictedActions
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

    def __ne__(self, other):
        return self.pos != other.pos

    def __str__(self):
        return f"[{self.trajectory.iloc[-1]['ts']:.2f} \
            {self.pos} \
            {self.trajectory.iloc[-1]['alt']:.2f} \
            {self.trajectory.iloc[-1]['fuel']:.2f}]"


class Action(Enum):
    up = -1
    straight = 0
    down = 1


# class D(DeterministicPlanningDomain, Actions, Renderable):
# because of `Actions`need to implement _get_applicable_actions_from


class D(DeterministicPlanningDomain, UnrestrictedActions, Renderable):
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
        if isinstance(origin, str):
            ap1 = openap.nav.airport(origin)
            self.lat1, self.lon1 = ap1["lat"], ap1["lon"]
        else:
            self.lat1, self.lon1 = origin

        if isinstance(destination, str):
            ap2 = openap.nav.airport(destination)
            self.lat2, self.lon2 = ap2["lat"], ap2["lon"]
        else:
            self.lat2, self.lon2 = destination
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

        self.windfield = windfield

        if windfield is not None:
            w = wind.PolyWind(
                windfield,
                self.cruise.proj,
                self.lat1,
                self.lon1,
                self.lat2,
                self.lon2,
            )
            self.cruise.wind = w
            self.climb.wind = w
            self.descent.wind = w

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

        # Build network between top of climb and destination airport
        self.np: int = 41
        self.nc: int = 11
        self.network = self.get_network(
            LatLon(self.dfcl.lat.iloc[-1], self.dfcl.lon.iloc[-1]),
            LatLon(self.lat2, self.lon2),
            self.np,
            self.nc,
        )

        self.start = State(self.dfcl, (0, self.nc // 2))

        logging.info(f"Climb trajectory:\n{self.dfcl}")

        logging.info(f"Constructor initialized in {time() - tick:.2f} seconds")

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
        # self.cruise.
        # Extract current time to set wind conditions
        ts = trajectory.ts.iloc[-1]
        # Update wind information
        # self.cruise.wind =

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

        # node = action.pos
        self.cruise.lat2 = self.network[next_x][next_y].lat
        self.cruise.lon2 = self.network[next_x][next_y].lon
        # Set number of colocation points
        self.cruise.setup_dc(nodes=5)
        # Compute the optimal trajectory
        dfcr = self.cruise.trajectory(self.objective[1])
        # Concatenate the two trajectories

        dfcr.ts = memory.trajectory.ts.iloc[-1] + dfcr.ts

        state = State(
            pd.concat([memory.trajectory, dfcr], ignore_index=True),
            (next_x, next_y),
        )
        logging.debug(f"Next state: {state}")
        return state

    def _get_transition_value(
        self,
        memory: D.T_state,
        action: D.T_event,
        next_state: Optional[D.T_state] = None,
    ) -> Value[D.T_value]:
        """
        Get the value (reward or cost) of a transition.

        Set cost to 1 when moving forward
        and to 2 when staying at the same location
        """

        assert memory != next_state, "Next state is the same as the current state"

        cost = oc.aero.distance(
            memory.trajectory.iloc[-1]["lat"],
            memory.trajectory.iloc[-1]["lon"],
            next_state.trajectory.iloc[-1]["lat"],
            next_state.trajectory.iloc[-1]["lon"],
        )

        # return Value(cost=1)
        logging.debug(f"comparing {next_state} with {memory} following action {action}")
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
        # return ImplicitSpace(lambda x: isinstance(x, State))
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
        # store used matplotlib subplot and image to only update them afterwards
        fig = Figure(figsize=(600, 600))
        fig.canvas.header_visible = False
        fig.canvas.footer_visible = False
        fig.canvas.resizable = False
        fig.set_dpi(1)

        latmin, latmax = min(self.lat1, self.lat2), max(self.lat1, self.lat2)
        lonmin, lonmax = min(self.lon1, self.lon2), max(self.lon1, self.lon2)

        ax = plt.axes(
            projection=ccrs.TransverseMercator(
                central_longitude=(lonmax - lonmin) / 2,
                central_latitude=(latmax - latmin) / 2,
            )
        )

        wind_sample = 30

        ax.set_extent([lonmin - 4, lonmax + 4, latmin - 2, latmax + 2])
        ax.add_feature(OCEAN, facecolor="#d1e0e0", zorder=-1, lw=0)
        ax.add_feature(LAND, facecolor="#f5f5f5", lw=0)
        ax.add_feature(BORDERS, lw=0.5, color="gray")
        ax.gridlines(draw_labels=True, color="gray", alpha=0.5, ls="--")
        ax.coastlines(resolution="50m", lw=0.5, color="gray")

        if self.windfield is not None:
            # get the closed altitude
            h_max = memory.trajectory.alt.max() * aero.ft
            fl = int(round(h_max / aero.ft / 100, -1))
            idx = np.argmin(abs(windfield.h.unique() - h_max))
            df_wind = (
                self.windfield.query(f"h=={self.windfield.h.unique()[idx]}")
                .query(f"longitude <= {lonmax + 2}")
                .query(f"longitude >= {lonmin - 2}")
                .query(f"latitude <= {latmax + 2}")
                .query(f"latitude >= {latmin - 2}")
            )

            ax.barbs(
                df_wind.longitude.values[::wind_sample],
                df_wind.latitude.values[::wind_sample],
                df_wind.u.values[::wind_sample],
                df_wind.v.values[::wind_sample],
                transform=ccrs.PlateCarree(),
                color="k",
                length=5,
                lw=0.5,
                label=f"Wind FL{fl}",
            )

        # great circle
        ax.scatter(self.lon1, self.lat1, c="darkgreen", transform=ccrs.Geodetic())
        ax.scatter(self.lon2, self.lat2, c="tab:red", transform=ccrs.Geodetic())

        ax.plot(
            [self.lon1, self.lon2],
            [self.lat1, self.lat2],
            label="Great Circle",
            color="tab:red",
            ls="--",
            transform=ccrs.Geodetic(),
        )

        # trajectory
        ax.plot(
            memory.trajectory.lon,
            memory.trajectory.lat,
            color="tab:green",
            transform=ccrs.Geodetic(),
            linewidth=2,
            marker=".",
            label="Optimal",
        )

        ax.legend()

        # Save it to a temporary buffer.
        # buf = BytesIO()
        # fig.savefig(buf, format="png")
        # Embed the result in the html output.
        # data = base64.b64encode(buf.getbuffer()).decode("ascii")

        return fig

    def heuristic(self, s: D.T_state) -> Value[D.T_value]:
        """Heuristic to be used by search algorithms.

        Here fuel consumption to reach target.

        """
        lat = s.trajectory.iloc[-1]["lat"]
        lon = s.trajectory.iloc[-1]["lon"]
        # Compute distance in meters
        distance_to_goal = oc.aero.distance(lat, lon, self.lat2, self.lon2)
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

    pause_between_steps = None
    max_steps = 100

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

    tick = time()
    FlightPlanningDomain.solve_with(solver, domain_factory)
    logging.info(f"Solved in {time() - tick:.2f} seconds")

    solver.reset()
    observation = domain.reset()

    logging.info("Starting planning")
    logging.info(observation)

    # Initialize image
    figure = domain.render(observation)
    # display(figure)
    plt.draw()
    plt.pause(0.001)

    # loop until max_steps or goal is reached
    for i_step in range(1, max_steps + 1):
        if pause_between_steps is not None:
            sleep(pause_between_steps)

        # choose action according to solver
        action = solver.sample_action(observation)
        # get corresponding action
        outcome = domain.step(action)
        observation = outcome.observation

        # update image
        figure = domain.render(observation)
        # clear_output(wait=True)
        # display(figure)
        plt.draw()
        plt.pause(0.001)

        # final state reached?
        if domain.is_terminal(observation):
            break

    # goal reached?
    is_goal_reached = domain.is_goal(observation)
    if is_goal_reached:
        logging.info(f"Goal reached in {i_step} steps!")
    else:
        logging.info(f"Goal not reached after {i_step} steps!")
