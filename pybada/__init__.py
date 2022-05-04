from dataclasses import dataclass

import casadi as ca
import numpy as np
from scipy.optimize import brentq

from pybada.bada4 import BADA4_jet_CR

kappa = 1.4
Rg = 287.05287
ft2m = 0.3048
m2ft = ft2m ** -1
T0_MSL = 288.15
P0_MSL = 101325.0
rho0_MSL = 1.225
a0_MSL = 340.294
ms2kt = 1.943844
g = 9.81
lapse_rate_troposphere = -0.0065
R_mean = 6356.8e3
d2r = np.pi / 180
flattening = 1 / 298.257223563
first_eccentricity = (flattening * (2 - flattening)) ** 0.5
R_a = 6378.137e3
R_b = 6356.752


class AircraftPosition(object):
    def __init__(self, latitude, longitude, Hp, t):
        self.lat = latitude
        self.lon = longitude
        self.Hp = Hp  # MSL height also called geopotential altitude (FL)
        self.t = t

    @classmethod
    def from_FL(cls, latitude, longitude, FL, t):
        altitude_ft = 100 * FL
        return cls(latitude, longitude, altitude_ft * ft2m, t)


class EnvironmentState(object):
    def __init__(self, pressure, temperature, density="auto", wind_vector=(0.0, 0.0)):
        if density == "auto":
            density = pressure / (Rg * temperature)
        self.P = pressure
        self.rho = density
        self.T = temperature
        self.wind = wind_vector

    def get_sound_speed(self):
        return (kappa * Rg * self.T) ** 0.5

    def theta(self):
        return self.T / T0_MSL

    def delta(self):
        return self.P / P0_MSL

    def sigma(self):
        return self.rho / rho0_MSL

    def __repr__(self):
        return f"EnvironmentState(P={self.P}, T={self.T}, rho={self.rho}, wind={self.wind})"


class AircraftState(object):
    def __init__(self, tas, heading, flight_path_angle, mass, mu=0):
        self.TAS = tas
        self.heading = heading
        self.gamma = flight_path_angle
        self.m = mass
        self.bank = mu

    def __repr__(self):
        return f"AircraftState(tas={self.TAS}, heading={self.heading}, flight_path_angle={self.gamma}, mass={self.m}, mu={self.bank})"

    def M(self, environment_state):
        return self.TAS / environment_state.get_sound_speed()

    def q(self, environment_state):
        es = environment_state
        return 0.5 * es.rho * self.TAS ** 2

    def CAS(self, environment_state):
        mu_air = 1 / 3.5
        es = environment_state
        expr1 = 1 + mu_air / 2 * es.rho / es.P * self.TAS ** 2
        expr2 = expr1 ** (1 / mu_air) - 1
        expr3 = 1 + es.P / P0_MSL * expr2
        expr4 = expr3 ** mu_air - 1
        return (2 * P0_MSL / mu_air / rho0_MSL * expr4) ** 0.5

    def CAS_kt(self, environment_state):
        return self.CAS(environment_state) * ms2kt

    def TAS_kt(self):
        return self.TAS * ms2kt

    def copy(self):
        return type(self)(self.TAS, self.heading, self.gamma, self.m, self.bank)


class ISA(object):
    """
    Casadi implementation of the ISA atmosphere (troposphere and tropopause only)
    """

    def __init__(self, deltaT=0):
        self.deltaT = deltaT
        a0 = 340.29
        P_MSL = 1.013250e5
        T_MSL = 288.15
        rho_MSL = 1.2250
        R = self.R = 287.05287

        h_tropopause = 11e3
        g = 9.80665
        lapse_rate_troposphere = -6.5e-3
        z = np.arange(0, 20.5e3, 500)
        T = np.maximum(T_MSL + lapse_rate_troposphere * z, 216.65 * np.ones_like(z))

        P_tropo = P_MSL * (T / T_MSL) ** (-g / R / lapse_rate_troposphere)
        P_pause = P_tropo[22] * np.exp(-g / R / T * (z - h_tropopause))
        P = np.hstack([P_tropo[:23], P_pause[23:]])
        z_I = np.arange(0, 20.5e3, 341)
        T_I = np.maximum(
            T_MSL + lapse_rate_troposphere * z_I, 216.65 * np.ones_like(z_I)
        )

        self.IP = ca.interpolant("P", "bspline", [z], P, {})
        self.IT = ca.interpolant("T", "bspline", [z_I], T_I, {})

    def get_environment_state(
        self, aircraft_position: AircraftPosition
    ) -> EnvironmentState:
        z = aircraft_position.Hp
        P = self.IP(z)
        T = self.IT(z)
        return EnvironmentState(P, T)


@dataclass
class VerticalControls(object):
    CT: float
    CL: float
    gamma: float


class ControlLaws:
    def __init__(self, apm: BADA4_jet_CR):
        self.apm = apm

    def get_climb_controls(
        self,
        environment_state: EnvironmentState,
        aircraft_state: AircraftState,
        dvdh: float = None,
    ) -> VerticalControls:
        assert dvdh is not None, "dvdh must be provided"

        CT = 1 * self.apm.CT_max_MCRZ(environment_state, aircraft_state)
        CL_factor = (
            2
            * aircraft_state.m
            * g
            / (environment_state.rho * aircraft_state.TAS ** 2 * self.apm.S)
        )

        def slope(gamma: float) -> float:
            h_dot = aircraft_state.TAS * np.sin(gamma)
            CL = CL_factor * np.cos(gamma)
            D = self.apm.D(environment_state, aircraft_state, CL)
            T = self.apm.T(environment_state, CT)
            v_dot = (T - D) / aircraft_state.m - g * np.sin(gamma)
            return v_dot / h_dot

        slope_diff = lambda gamma: slope(gamma) - dvdh
        output = brentq(slope_diff, 1e-6, 0.5, full_output=True)
        assert output[1].converged

        gamma = output[0]
        CL = CL_factor * np.cos(gamma)
        return VerticalControls(CT, CL, gamma)

    def get_descent_controls(
        self,
        environment_state: EnvironmentState,
        aircraft_state: AircraftState,
        dvdh: float = None,
    ) -> VerticalControls:
        assert dvdh is not None, "dvdh must be provided"

        CT = 1.05 * self.apm.CT_min(environment_state, aircraft_state)
        CL_factor = (
            2
            * aircraft_state.m
            * g
            / (environment_state.rho * aircraft_state.TAS ** 2 * self.apm.S)
        )

        def slope(gamma: float) -> float:
            h_dot = aircraft_state.TAS * np.sin(gamma)
            CL = CL_factor * np.cos(gamma)
            D = self.apm.D(environment_state, aircraft_state, CL)
            T = self.apm.T(environment_state, CT)
            v_dot = (T - D) / aircraft_state.m - g * np.sin(gamma)
            return v_dot / h_dot

        slope_diff = lambda gamma: slope(gamma) - dvdh
        output = brentq(slope_diff, -0.5, -1e-6, full_output=True)
        assert output[1].converged

        gamma = output[0]
        CL = CL_factor * np.cos(gamma)
        return VerticalControls(CT, CL, gamma)

    def get_cruise_controls(
        self, environment_state: EnvironmentState, aircraft_state: AircraftState
    ) -> VerticalControls:
        CL = self.apm.CL(environment_state, aircraft_state)
        D = self.apm.D(environment_state, aircraft_state, CL)
        CT = self.apm.CT(environment_state, D)

        return VerticalControls(CT, CL, 0.0)


class AircraftPerformanceModel(object):
    def __init__(self):
        pass

    def v_dot(
        self, env: EnvironmentState, acs: AircraftState, vc: VerticalControls
    ) -> float:
        T = self.T(env, vc.CT)
        D = self.D(env, acs, vc.CL)
        return (T - D) / acs.m - g * np.sin(acs.gamma)

    def m_dot(self, env: EnvironmentState, acs: AircraftState, vc: VerticalControls):
        CF = self.CF(env, acs, vc.CT)
        return -self.fc(env, CF)
