"""Couples the Kacker-Okapuu loss model (Losses/losses.py) to the meanline
solver (Meanline/solve_meanline.py).

The meanline takes kinetic-energy loss coefficients (Setpoints.loss_kinetic_*)
as inputs. This module closes the loop:

    xi guess (from setpoints) -> Solve() -> RowFlow/RowGeometry -> KO -> new xi
    ... repeat until xi stops changing.

The meanline code is not modified; new setpoints are made with
dataclasses.replace(). Run from src/:  python -m Losses.meanline_losses
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace, field

from scipy.optimize import brentq

from boundary_conditions import boundary_conditions as BCS
from Meanline.gas_props import GAS, GasProps
from Meanline.setpoints import SETPOINTS, Setpoints
from Meanline.solve_meanline import Solve, SolveResult
from Losses.losses import (LossOptions, RowFlow, RowGeometry, RowLoss, apply_tip_clearance,
                           print_breakdown, row_loss, zweifel_pitch_chord)

BAR = 1e5  # Pa


# ---------------------------------------------------------------------------
# Airfoil parameters: placeholders until the airfoil module exists
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AirfoilParams:
    aspect_ratio: float              # h / c
    axial_chord_ratio: float         # b_x / c
    zweifel: float = 0.8             # sets s/c through zweifel_pitch_chord()
    tmax_chord: float = 0.20         # t_max / c
    te_throat: float = 0.08          # t_TE / o
    metal_in: float | None = None    # deg, signed; None -> taken from setpoints


VANE_AIRFOIL = AirfoilParams(aspect_ratio=1.0, axial_chord_ratio=0.65)
ROTOR_AIRFOIL = AirfoilParams(aspect_ratio=1.2, axial_chord_ratio=0.85)


# ---------------------------------------------------------------------------
# Meanline solution -> loss-model inputs
# ---------------------------------------------------------------------------
def _v(sol: dict, key: str) -> float:
    """Value of a meanline state; entries are (value, unit) tuples or lists."""
    x = sol[key]
    return x[0] if isinstance(x, (tuple, list)) else x


def _inlet_mach(m_dot, area, T0, P0_pa, k, R):
    """Subsonic Mach number that passes m_dot through area at (T0, P0)."""
    def flux(M):
        return (P0_pa * area * math.sqrt(k / (R * T0)) * M
                * (1 + (k - 1) / 2 * M**2)**(-(k + 1) / (2 * (k - 1))))
    return brentq(lambda M: flux(M) - m_dot, 1e-6, 1.0)


def row_inputs(result: SolveResult, sp: Setpoints = SETPOINTS, gas: GasProps = GAS,
               vane_af: AirfoilParams = VANE_AIRFOIL, rotor_af: AirfoilParams = ROTOR_AIRFOIL):
    """Build (vane_geo, vane_flow, rotor_geo, rotor_flow) from a meanline solution.

    Sign convention (losses.py): + along rotation, rotor angles relative.
    The meanline reports Beta2 unsigned and Beta3 positive against rotation,
    so they are re-signed here.
    """
    sol = result.solution
    k, R = gas.gamma, gas.R
    P01, Ps2, Ps3 = sp.P_01, _v(sol, 'P_s2'), sp.P_01 / sp.pressure_ratio   # bar
    T2, T3 = _v(sol, 'T_2'), _v(sol, 'T_3')
    v2, w2, w3, u2 = _v(sol, 'v_2'), _v(sol, 'w_2'), _v(sol, 'w_3'), _v(sol, 'u_2')
    m_dot = _v(sol, 'm_dot_2')

    # ---- vane ----
    alpha2 = sp.alpha_2
    M1 = _inlet_mach(m_dot, _v(sol, 'stator_area'), sp.T_01, P01 * BAR, k, R)
    h_v = (sp.D_tip_stator - sp.D_hub_stator) / 2
    c_v = h_v / vane_af.aspect_ratio
    vane_geo = RowGeometry(
        pitch_chord=zweifel_pitch_chord(sp.alpha_1, alpha2, vane_af.axial_chord_ratio, vane_af.zweifel),
        chord=c_v, axial_chord=c_v * vane_af.axial_chord_ratio, height=h_v,
        tmax_chord=vane_af.tmax_chord, te_throat=vane_af.te_throat,
        metal_in=sp.alpha_1 if vane_af.metal_in is None else vane_af.metal_in,
        r_hub=sp.D_hub_stator / 2, r_tip=sp.D_tip_stator / 2)
    vane_flow = RowFlow(
        'vane', angle_in=sp.alpha_1, angle_out=alpha2,
        M_in=M1, M_out=_v(sol, 'M_2'),
        P_in=P01 / (1 + (k - 1) / 2 * M1**2)**(k / (k - 1)), P_out=Ps2,
        Re=_v(sol, 'density_2') * v2 * c_v / gas.mu, gamma=k)

    # ---- rotor ----
    sign_in = 1.0 if v2 * math.sin(math.radians(alpha2)) >= u2 else -1.0
    beta2 = sign_in * _v(sol, 'Beta2')
    beta3 = -sp.Beta3_metal
    h_r = (sp.D_tip_rotor - sp.D_hub_rotor) / 2
    c_r = h_r / rotor_af.aspect_ratio
    rotor_geo = RowGeometry(
        pitch_chord=zweifel_pitch_chord(beta2, beta3, rotor_af.axial_chord_ratio, rotor_af.zweifel),
        chord=c_r, axial_chord=c_r * rotor_af.axial_chord_ratio, height=h_r,
        tmax_chord=rotor_af.tmax_chord, te_throat=rotor_af.te_throat,
        metal_in=sign_in * sp.Beta2_metal if rotor_af.metal_in is None else rotor_af.metal_in,
        r_hub=sp.D_hub_rotor / 2, r_tip=sp.D_tip_rotor / 2,
        clearance=BCS.rotor_tip_clearance_ratio * h_r)
    rotor_flow = RowFlow(
        'rotor', angle_in=beta2, angle_out=beta3,
        M_in=_v(sol, 'M_2rel'), M_out=w3 / math.sqrt(k * R * T3),
        P_in=Ps2, P_out=Ps3,
        Re=_v(sol, 'density_3') * w3 * c_r / gas.mu, gamma=k)

    return vane_geo, vane_flow, rotor_geo, rotor_flow


# ---------------------------------------------------------------------------
# Efficiency of the (uncooled) meanline solution, consistent with its constant cp
# ---------------------------------------------------------------------------
def meanline_efficiency(result: SolveResult, sp: Setpoints = SETPOINTS, gas: GasProps = GAS):
    sol = result.solution
    k, cp = gas.gamma, gas.C_p
    e = (k - 1) / k
    T01, P01, Ps3 = sp.T_01, sp.P_01, sp.P_01 / sp.pressure_ratio
    T3, v3 = _v(sol, 'T_3'), _v(sol, 'v_3')
    T03 = T3 + v3**2 / (2 * cp)
    P03 = Ps3 * (T03 / T3)**(1 / e)
    dT = T01 - T03
    return dict(
        work_kJ_kg=cp * dT / 1e3,
        T03=T03, P03=P03,
        eta_tt=dT / (T01 * (1 - (P03 / P01)**e)),
        eta_ts=dT / (T01 * (1 - (Ps3 / P01)**e)),
    )


# ---------------------------------------------------------------------------
# Loss <-> meanline fixed-point loop
# ---------------------------------------------------------------------------
@dataclass
class LossCoupledResult:
    meanline: SolveResult
    setpoints: Setpoints             # with the converged loss coefficients
    vane: RowLoss
    rotor: RowLoss
    efficiency: dict                 # eta_tt, eta_ts (with tip-clearance debit) and _no_tip
    iterations: int
    history: list = field(default_factory=list)   # (xi_vane, xi_rotor) per iteration


def solve_with_losses(sp: Setpoints = SETPOINTS, gas: GasProps = GAS,
                      vane_af: AirfoilParams = VANE_AIRFOIL, rotor_af: AirfoilParams = ROTOR_AIRFOIL,
                      opts: LossOptions = LossOptions(), tol: float = 1e-6,
                      max_iter: int = 50) -> LossCoupledResult:
    """Iterate the meanline and KO until the kinetic loss coefficients converge."""
    history = []
    for it in range(1, max_iter + 1):
        result = Solve(gas, sp)
        vg, vf, rg, rf = row_inputs(result, sp, gas, vane_af, rotor_af)
        lv, lr = row_loss(vg, vf, opts), row_loss(rg, rf, opts)
        history.append((lv.xi, lr.xi))
        change = max(abs(lv.xi - sp.loss_kinetic_stator), abs(lr.xi - sp.loss_kinetic_rotor))
        if change < tol:
            break
        sp = replace(sp, loss_kinetic_stator=lv.xi, loss_kinetic_rotor=lr.xi)
    else:
        raise RuntimeError(f'loss loop did not converge in {max_iter} iterations (change {change:.2e})')

    eff = meanline_efficiency(result, sp, gas)
    eff['eta_tt_no_tip'], eff['eta_ts_no_tip'] = eff['eta_tt'], eff['eta_ts']
    eff['eta_tt'] = apply_tip_clearance(eff['eta_tt'], [lr])
    eff['eta_ts'] = apply_tip_clearance(eff['eta_ts'], [lr])
    return LossCoupledResult(result, sp, lv, lr, eff, it, history)


def main():
    res = solve_with_losses()
    sol = res.meanline.solution
    print(f'Converged in {res.iterations} loss iterations; '
          f'degree of reaction = {res.meanline.degree_of_reaction:.4f}, '
          f'm_dot = {_v(sol, "m_dot_2"):.2f} kg/s')
    print(f'xi_vane = {res.vane.xi:.4f}, xi_rotor = {res.rotor.xi:.4f} '
          f'(setpoint guesses were {SETPOINTS.loss_kinetic_stator}, {SETPOINTS.loss_kinetic_rotor})\n')
    print_breakdown([res.vane, res.rotor], ['vane', 'rotor'])
    e = res.efficiency
    print(f'\nWork = {e["work_kJ_kg"]:.1f} kJ/kg')
    print(f'eta_tt = {e["eta_tt"]:.4f}  (no tip clearance {e["eta_tt_no_tip"]:.4f})')
    print(f'eta_ts = {e["eta_ts"]:.4f}  (no tip clearance {e["eta_ts_no_tip"]:.4f})')


if __name__ == '__main__':
    main()
