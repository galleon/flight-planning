import csv
import math as m
import os
import sys
import xml.etree.ElementTree
from enum import Enum
from os import environ as osenv
from itertools import product

import casadi as ca
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp2d, LinearNDInterpolator


from pybada import AircraftPerformanceModel, AircraftState, EnvironmentState, a0_MSL, g


class Table(object):
    def __init__(self, filename):
        x = None
        y = np.array([])
        z = np.array([])
        with open(filename, "r") as txtfile:
            for index, line in enumerate(txtfile):
                if line.startswith("Table name:"):
                    _, name = line.split(":")
                    name = name.strip()
                elif line.startswith("Table variables:"):
                    _, vars = line.split(":")
                    vars = tuple(map(str.strip, vars.split(",")))
                elif line.startswith("Table dimension:"):
                    _, dims = line.split(":")
                    dims = tuple(map(lambda x: int(x.strip()), dims.split("x")))
                else:
                    if "|" in line:
                        line = line.replace("|", "")
                        if x is None:
                            x = np.array(line.split()[1:], dtype=float)
                            assert len(x) == dims[0]
                        else:
                            items = line.split()
                            if len(items) == dims[0] + 1:
                                y = np.append(y, float(items[0]))
                                z = np.append(z, np.array(items[1:], dtype=float))

            assert x.size == dims[0]
            assert y.size == dims[1]
            z = np.reshape(z, (dims[1], dims[0]))
            assert z.shape == (dims[1], dims[0])

            # print(list(zip(list(product(x, y)), z.T.flatten())))

            # self.f = interp2d(x, y, z.T, kind="cubic", bounds_error=True)
            self.f = LinearNDInterpolator(list(product(x, y)), z.T.flatten())

    def __call__(self, x, y):
        return self.f(x, y)


class BADA4_jet_CR(AircraftPerformanceModel):
    """
    Implementation of the BADA4 APM with Casadi functions.
    """

    def __init__(self, label, full_path=None):
        if full_path is None:
            try:
                BADA4_PATH = osenv.get("BADA4_PATH", None)
            except KeyError:
                raise ValueError("No path specified for the aircraft data file")

            full_path = os.path.join(BADA4_PATH, label, label + ".xml")

        xml_tree = xml.etree.ElementTree.parse(full_path)
        xml_root = xml_tree.getroot()
        self.ac_label = label
        self.model_label = xml_root.find("model").text
        self.ac_type = xml_root.find("type").text
        ###
        # Aerodynamic Forces and Configurations Model
        ###
        self.AFCM = AFCM = xml_root.find("AFCM")
        self.S = float(AFCM.find("S").text)
        self.configs = AFCM.findall("Configuration")
        ###
        # CR
        ###
        self.clean_cfg = config0 = [
            cfg for cfg in self.configs if cfg.attrib["HLid"] == "0"
        ][0]
        self.CD_scalar = float(
            config0.find("LGUP").find("DPM_clean").find("scalar").text
        )
        self.V_FE_0 = float(config0.find("vfe").text)
        self.CD_clean = CD_clean = (
            config0.find("LGUP").find("DPM_clean").find("CD_clean")
        )
        self.d = [float(d.text) for d in CD_clean.findall("d")]
        LGUP = config0.find("LGUP")
        BLM__ = LGUP.find("BLM_clean")
        self.M_min = float(BLM__.find("Mmin").text)
        self.M_max = float(config0.find("LGUP").find("BLM_clean").find("Mmax").text)
        self.CL_clean = CL_clean = (
            config0.find("LGUP").find("BLM_clean").find("CL_clean")
        )
        self.b_f = [float(bf.text) for bf in CL_clean.findall("bf")]
        ###
        # Propulsive Forces Model
        ###
        self.PFM = PFM = xml_root.find("PFM")
        self.m_ref = float(PFM.find("MREF").text)
        self.LHV = float(PFM.find("LHV").text)
        self.rho_fuel = float(PFM.find("rho").text)
        self.n_eng = float(PFM.find("n_eng").text)
        ##
        # CT
        self.CT_ = PFM.find("TFM").find("CT")
        self.a = [float(a.text) for a in self.CT_.findall("a")]

        ##
        # CF
        CF_ = PFM.find("TFM").find("CF")
        self.f = [float(f.text) for f in CF_.findall("f")]

        ##
        # MCRZ
        # Flat rating
        flat_rating = PFM.find("TFM").find("MCRZ").find("flat_rating")
        self.b_MCRZ = [float(b.text) for b in flat_rating.findall("b")]

        # Temperature rating
        temp_rating = PFM.find("TFM").find("MCRZ").find("temp_rating")
        self.c_MCRZ = [float(c.text) for c in temp_rating.findall("c")]

        # Kink point
        self.kink_MCRZ = float(PFM.find("TFM").find("MCRZ").find("kink").text)

        ##
        # MCMB
        # Flat rating
        flat_rating = PFM.find("TFM").find("MCMB").find("flat_rating")
        self.b_MCMB = [float(b.text) for b in flat_rating.findall("b")]

        # Temperature rating
        temp_rating = PFM.find("TFM").find("MCMB").find("temp_rating")
        self.c_MCMB = [float(c.text) for c in temp_rating.findall("c")]

        # Kink point
        self.kink_MCMB = float(PFM.find("TFM").find("MCMB").find("kink").text)

        ##
        # LIDL
        CT = PFM.find("TFM").find("LIDL").find("CT")
        self.ti = [float(ti.text) for ti in CT.findall("ti")]

        CF = PFM.find("TFM").find("LIDL").find("CF")
        self.fi = [float(fi.text) for fi in CF.findall("fi")]

        ###
        # Aircraft Limitations Model
        ###
        ALM = xml_root.find("ALM")

        ##
        # GLM
        self.h_MO_ft = float(ALM.find("GLM").find("hmo").text)
        self.mfa = float(ALM.find("GLM").find("mfa").text)
        ##
        # KLM
        self.M_MO = float(ALM.find("KLM").find("mmo").text)
        self.V_MO_kn = float(ALM.find("KLM").find("vmo").text)
        self.mle = float(ALM.find("KLM").find("mle").text)
        self.V_LE_kn = float(ALM.find("KLM").find("vle").text)
        # mlo = float(ALM.find('KLM').find('mlo').text)
        self.vloe = float(ALM.find("KLM").find("vloe").text)
        self.vlor = float(ALM.find("KLM").find("vlor").text)
        ##
        # DLM
        self.MTW = float(ALM.find("DLM").find("MTW").text)
        self.MTOW = float(ALM.find("DLM").find("MTOW").text)
        self.MLW = float(ALM.find("DLM").find("MLW").text)
        self.MZFW = float(ALM.find("DLM").find("MZFW").text)
        self.OEW = float(ALM.find("DLM").find("OEW").text)
        self.MPL = float(ALM.find("DLM").find("MPL").text)
        self.MFL = float(ALM.find("DLM").find("MFL").text)
        self.n1 = float(ALM.find("DLM").find("n1").text)
        self.n3 = float(ALM.find("DLM").find("n3").text)
        self.nf1 = float(ALM.find("DLM").find("nf1").text)
        self.nf3 = float(ALM.find("DLM").find("nf3").text)

        econ_file = os.path.join(BADA4_PATH, label, "ECON.OPT")
        self.best_mach = None
        if os.path.exists(econ_file):
            self.best_mach = Table(econ_file)

        optalt_file = os.path.join(BADA4_PATH, label, "OPTALT.OPT")
        self.best_alt = None
        if os.path.exists(optalt_file):
            self.best_alt = Table(optalt_file)

        assert (
            self.best_mach is not None or self.best_alt is not None
        ), "No ECON or OPTALT file found"

    def get_marginal_fuel_burn(
        self,
        environment_state: EnvironmentState,
        initial_aircraft_state: AircraftState,
        elapsed_time_seconds: float,
    ) -> float:
        mf = self.integrate_mass_at_constant_tas_environment(
            environment_state, initial_aircraft_state, elapsed_time_seconds
        )
        acs = initial_aircraft_state.copy()
        acs.mf = mf
        return self.get_fuel_burn_at_cruise_conditions(environment_state, acs)

    def integrate_mass_at_constant_tas_environment(
        self,
        environment_state: EnvironmentState,
        initial_aircraft_state: AircraftState,
        elapsed_time_seconds: float,
    ) -> float:
        acs = initial_aircraft_state.copy()

        def dmdt(t, x, sign=1):
            m = x[0]
            acs.m = m
            fb = self.get_fuel_burn_at_cruise_conditions(environment_state, acs)
            return -fb * sign

        y0 = np.array([initial_aircraft_state.m])
        if elapsed_time_seconds > 0:
            output = solve_ivp(dmdt, [0, elapsed_time_seconds], y0, method="LSODA")
        else:
            # Integrate backwards in time (for example, when finding initial mass)
            output = solve_ivp(
                lambda t, x: dmdt(t, x, -1),
                [0, -elapsed_time_seconds],
                y0,
                method="LSODA",
            )
        assert output.success
        return output.y[0, -1]

    def get_fuel_burn_at_cruise_conditions(
        self, environment_state: EnvironmentState, aircraft_state: AircraftState
    ) -> float:
        D = self.D(environment_state, aircraft_state)
        CT = self.CT(environment_state, D)
        CF = self.CF(environment_state, aircraft_state, CT)
        return self.fc(environment_state, CF)

    def get_cruise_CL(self, environment_state, aircraft_state):
        env = environment_state
        acs = aircraft_state
        q = acs.q(env)
        return acs.m * g / (self.S * q * ca.cos(acs.bank) * ca.cos(acs.gamma))

    def CL(self, environment_state, aircraft_state):
        return self.get_cruise_CL(environment_state, aircraft_state)

    def CL_from_m_v_rho_gamma_nobank(self, m, v, rho, gamma):
        return 2 * m * g / (self.S * rho * v ** 2 * ca.cos(gamma))

    def CL_max(self, environment_state, aircraft_state):
        M = aircraft_state.M(environment_state)
        return (
            self.b_f[0]
            + self.b_f[1] * M
            + self.b_f[2] * M ** 2
            + self.b_f[3] * M ** 3
            + self.b_f[4] * M ** 4
        )

    def CD_from_CL_M(self, CL, M):
        d = [self.CD_scalar] + self.d  # align 0-index with 1-index
        # C0 = d[1] + d[2]/(1-softcap(M)**2)**0.5 + d[3]/(1-M**2) + d[4]/(1-M**2)**(3/2) + d[5]/(1-M**2)**2
        C0 = (
            d[1]
            + d[2] / (1 - M ** 2) ** 0.5
            + d[3] / (1 - M ** 2)
            + d[4] / (1 - M ** 2) ** (3 / 2)
            + d[5] / (1 - M ** 2) ** 2
        )
        C2 = (
            d[6]
            + d[7] / (1 - M ** 2) ** (3 / 2)
            + d[8] / (1 - M ** 2) ** 3
            + d[9] / (1 - M ** 2) ** (9 / 2)
            + d[10] / (1 - M ** 2) ** 6
        )
        C6 = (
            d[11]
            + d[12] / (1 - M ** 2) ** (7)
            + d[13] / (1 - M ** 2) ** (15 / 2)
            + d[14] / (1 - M ** 2) ** (8)
            + d[15] / (1 - M ** 2) ** (17 / 2)
        )
        return d[0] * (C0 + C2 * CL ** 2 + C6 * CL ** 6)

    def CD(self, environment_state, aircraft_state, CL=None):
        if CL is None:
            CL = self.CL(environment_state, aircraft_state)
        M = aircraft_state.M(environment_state)
        d = [self.CD_scalar] + self.d  # align 0-index with 1-index
        # C0 = d[1] + d[2]/(1-softcap(M)**2)**0.5 + d[3]/(1-M**2) + d[4]/(1-M**2)**(3/2) + d[5]/(1-M**2)**2
        C0 = (
            d[1]
            + d[2] / (1 - M ** 2) ** 0.5
            + d[3] / (1 - M ** 2)
            + d[4] / (1 - M ** 2) ** (3 / 2)
            + d[5] / (1 - M ** 2) ** 2
        )
        C2 = (
            d[6]
            + d[7] / (1 - M ** 2) ** (3 / 2)
            + d[8] / (1 - M ** 2) ** 3
            + d[9] / (1 - M ** 2) ** (9 / 2)
            + d[10] / (1 - M ** 2) ** 6
        )
        C6 = (
            d[11]
            + d[12] / (1 - M ** 2) ** 7
            + d[13] / (1 - M ** 2) ** (15 / 2)
            + d[14] / (1 - M ** 2) ** 8
            + d[15] / (1 - M ** 2) ** (17 / 2)
        )
        return d[0] * (C0 + C2 * CL ** 2 + C6 * CL ** 6)

    def D(self, environment_state, aircraft_state, CL=None):
        # CD = self.CD(environment_state, aircraft_state, CL)
        if CL is None:
            CL = self.CL(environment_state, aircraft_state)
        CD = self.CD_from_CL_M(CL, aircraft_state.M(environment_state))
        return aircraft_state.q(environment_state) * self.S * CD

    def CT(self, environment_state, thrust):
        return thrust / environment_state.delta() / g / self.m_ref

    def T(self, environment_state, CT):
        return CT * environment_state.delta() * g * self.m_ref

    def CT_min(self, environment_state, aircraft_state):
        delta = environment_state.delta()
        M = aircraft_state.M(environment_state)
        ti = self.ti
        return (
            ti[0] / delta
            + ti[1]
            + ti[2] * delta
            + ti[3] * delta ** 2
            + M * (ti[4] / delta + ti[5] + ti[6] * delta + ti[7] * delta ** 2)
            + M ** 2 * (ti[8] / delta + ti[9] + ti[10] * delta + ti[11] * delta ** 2)
        )

    def CT_max_MCRZ(self, environment_state, aircraft_state):
        M = aircraft_state.M(environment_state)
        delta = environment_state.delta()
        delta_T_flat_MCRZ = 0
        for i in range(6):
            delta_T_flat_MCRZ += delta ** i * sum(
                self.b_MCRZ[i * 6 + j] * M ** j for j in range(6)
            )
        CT_max = 0
        for i in range(6):
            CT_max += delta_T_flat_MCRZ ** i * sum(
                self.a[i * 6 + j] * M ** j for j in range(6)
            )
        return CT_max

    def CF_from_CT_M(self, CT, M):
        CF = self._CF(M, CT)
        return CF

    def fc_from_thrust_P_M_T(self, thrust, P, M, T):
        env = EnvironmentState(P, T)
        CT = self.CT(env, thrust)
        CF = self._CF(M, CT)
        fc = self.fc(env, CF)
        return fc

    def _CF(self, M, CT):
        CF = 0
        for i in range(5):
            CF += M ** i * sum(self.f[i * 5 + j] * CT ** j for j in range(5))
        return CF

    def CF(self, environment_state, aircraft_state, CT):
        M = aircraft_state.M(environment_state)
        return self._CF(M, CT)

    def fc(self, environment_state, CF):
        delta = environment_state.delta()
        theta = environment_state.theta()
        return delta * theta ** 0.5 * self.m_ref * g * a0_MSL * CF / self.LHV

    def CF_idle(self, environment_state, aircraft_state):
        M = aircraft_state.M(environment_state)
        delta = environment_state.delta()
        theta = environment_state.theta()
        CFi = 0
        for i in range(3):
            CFi += M ** i * sum(self.fi[i * 3 + j] * delta ** (j - 1) for j in range(3))
        CFi *= theta ** -0.5
        return CFi

    def get_parameters(self):
        attribs = dir(self)
        ga = getattr

        def is_parameter(p):
            if isinstance(p, float):
                return True
            elif isinstance(p, list):
                if isinstance(p[0], float):
                    return True
            return False

        d = dict([(a, ga(self, a)) for a in attribs if is_parameter(ga(self, a))])
        d["d"] = [self.CD_scalar] + self.d
        return d


if __name__ == "__main__":
    aircraft = BADA4_jet_CR("A330-321")
