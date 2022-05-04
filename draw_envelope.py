import numpy as np
import scipy.optimize

from pybada import (
    AircraftState,
    EnvironmentState,
    T0_MSL,
    P0_MSL,
    lapse_rate_troposphere,
    ft2m,
)
from pybada.bada4 import BADA4_jet_CR


Htrop = 11000.0
T_ISA_trop = T0_MSL + Htrop * lapse_rate_troposphere
P_trop = P0_MSL * (1 + lapse_rate_troposphere * Htrop / T0_MSL) ** 5.2561

stall_margin = 0


def T(z):
    T_ts = T0_MSL + z * lapse_rate_troposphere
    return np.maximum(T_ts, T_ISA_trop * np.ones_like(T_ts))


def P(z):
    P_ts = P0_MSL * (1 + lapse_rate_troposphere * z / T0_MSL) ** 5.2561
    for i in range(z.shape[0]):
        if z[i] >= Htrop:
            R = 287.05287
            P_ts[i] = P_trop * np.exp(-9.8 / R / T_ISA_trop * (z[i] - Htrop))
    return P_ts


aircraft = BADA4_jet_CR("A350-941")  # "A320-231"

Z = np.linspace(0, aircraft.h_MO_ft * ft2m, 40)
Z_plot = Z / ft2m
Tz = T(Z)
Pz = P(Z)

for i, cm in enumerate([0.8, 0.9, 1.0]):
    m = cm * aircraft.MTOW

    tas_min = np.zeros_like(Z)
    tas_max = np.zeros_like(Z)
    M_min = np.zeros_like(Z)
    M_max = np.zeros_like(Z)
    ES = [EnvironmentState(p, t) for p, t in zip(Pz, Tz)]
    for j, z in enumerate(Z):

        def f(tas, j=j, z=z):
            ac_state = AircraftState(tas, 0, 0, m)
            return aircraft.CL_max(ES[j], ac_state) - aircraft.CL(ES[j], ac_state)

        try:
            x0 = scipy.optimize.brentq(f, 60, aircraft.M_MO * ES[j].get_sound_speed())
            tas_min[j] = x0
        except ValueError:
            # print(cm, z)
            # print("Error in computation of Vstall", f(60), f(self.M_MO*ES[j].get_sound_speed()))
            tas_min[j] = np.nan

        # assert r.converged

        def g(tas, j=j, z=z):
            ac_state = AircraftState(tas, 0, 0, m)
            CT_max = aircraft.CT_max_MCRZ(ES[j], ac_state)
            Tr = aircraft.T(ES[j], CT_max)
            D = aircraft.D(ES[j], ac_state)
            return Tr - D

        # print(g(tas_min[j]), g(340))
        Mmax_speed = aircraft.M_MO * ES[j].get_sound_speed()
        if g(Mmax_speed) >= 0.0:
            tas_max[j] = Mmax_speed
        else:
            v_max = tas_min[j]
            v = tas_min[j]
            delta_v = 1
            while g(v) <= 0.0 and v <= Mmax_speed:
                v += 1
            if g(v) >= 0.0:
                v_max = scipy.optimize.brentq(g, v, Mmax_speed)
            tas_max[j] = v_max
    M_max[j] = AircraftState(tas_max[j], 0, 0, m).M(ES[j])
    M_min[j] = AircraftState(tas_min[j], 0, 0, m).M(ES[j])
    import matplotlib.pyplot as plt

    plt.plot(tas_min, Z_plot, ".-", color="C0", alpha=cm)
    plt.plot(tas_max, Z_plot, ".-", color="C1", alpha=cm)
    plt.grid(True)
    if i == 2:
        if stall_margin:
            plt.plot(tas_min * (1 + stall_margin), Z_plot, "-.", color="C2", alpha=1.0)
a = EnvironmentState(P(Z), T(Z)).get_sound_speed()
plt.plot(a * 0.6, Z_plot, "k--", alpha=1.0, label="0.6")
plt.plot(a * 0.7, Z_plot, "k--", alpha=1.0)
plt.plot(a * 0.8, Z_plot, "k--", alpha=1.0, label="$M \\in $ [0.6, 0.7, 0.8]")
# plt.plot([100, 250], [P2Hp(20000)/ft2m, P2Hp(20000)/ft2m], 'k')

plt.ylabel("Altitude (ft)")
plt.xlabel("True Airspeed (m/s)")
plt.title(f"BADA4 flight envelope for {aircraft.model_label}")
plt.show()
