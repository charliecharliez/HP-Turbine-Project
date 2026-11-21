"""Loss module for the ME 533 HP turbine design: Kacker-Okapuu row losses,
coolant mixing, and cooled-turbine efficiencies.

Gas properties come from Meanline/gas_props.py (GasProps: constant C_p, gamma,
R). The KO correlations themselves take gamma through RowFlow. The meanline and
airfoil modules fill the input records (RowGeometry, RowFlow); the coupling to
the meanline solver lives in Losses/meanline_losses.py.

Loss coefficients
-----------------
KO returns the total-pressure loss coefficient, in the row frame,
    Y = (P01 - P02) / (P02 - P2).
The meanline (Meanline/solve_meanline.py) uses the kinetic-energy loss
coefficient xi = 1 - V^2 / V_is^2. kinetic_loss_from_Y() converts between them.

Angle convention for inputs: angles from axial, in degrees, signed, positive
along the direction of blade rotation. Vane angles are absolute; rotor angles
are relative (so a rotor exit relative angle is negative). The module converts
to KO's convention internally.

Pressures may be in any consistent unit (only ratios enter the gas relations).

References
----------
Kacker & Okapuu, ASME J. Eng. Power 104 (1982) 111-119.
Ainley & Mathieson, ARC R&M 2974 (1951); Dunham & Came, ASME J. Eng. Power (1970).
Chart data (Losses/data/ko_fig*.csv): KO 1982 Figs. 1, 2 and 14 as digitized
in NASA turbo-design, references/Turbines/KackerOkapuu
(https://github.com/nasa/turbo-design), checked against the figure images.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

from Meanline.gas_props import GAS, GasProps

# ---------------------------------------------------------------------------
# Digitized KO 1982 charts
# ---------------------------------------------------------------------------
_DATA = Path(__file__).with_name('data')


def _load(name):
    return np.loadtxt(_DATA / name, delimiter=',', skiprows=1)


def _curves(table):
    """Split an (s/c, Yp, alpha2) table into {alpha2: (s/c sorted, Yp)}."""
    out = {}
    for a in np.unique(table[:, 2]):
        c = table[table[:, 2] == a]
        c = c[np.argsort(c[:, 0])]
        out[float(a)] = (c[:, 0], c[:, 1])
    return out


# Fig. 1: Yp vs s/c for beta1 = 0 (axial-entry nozzle), tmax/c = 0.2, alpha2 = 40..80 deg
FIG01_NOZZLE = _curves(_load('ko_fig01_yp_nozzle.csv'))
# Fig. 2: Yp vs s/c for beta1 = alpha2 (impulse), tmax/c = 0.2, alpha2 = 40..70 deg
FIG02_IMPULSE = _curves(_load('ko_fig02_yp_impulse.csv'))
# Fig. 14: trailing-edge energy coefficient Delta-phi^2 vs t_TE/o (0..0.4)
FIG14_NOZZLE = _load('ko_fig14_te_nozzle.csv')     # axial-entry nozzle (beta1 = 0)
FIG14_IMPULSE = _load('ko_fig14_te_impulse.csv')   # impulse blading (beta1 = alpha2)

# KO hub-to-mean inlet Mach ratio: M1h/M1 = 1 + K * |rh/rt - 1|^2.2.
# Only used when the spanwise module does not supply M_in_hub. Verify K.
HUB_MACH_K = {'vane': 1.8, 'rotor': 5.2}


# ---------------------------------------------------------------------------
# Input / output records
# ---------------------------------------------------------------------------
@dataclass
class RowGeometry:
    """From the airfoil module. Lengths in m, angles in deg."""
    pitch_chord: float      # s/c
    chord: float            # true chord c
    axial_chord: float      # b_x
    height: float           # blade height h (row exit)
    tmax_chord: float       # t_max / c
    te_throat: float        # t_TE / o
    metal_in: float         # inlet metal angle (signed)
    r_hub: float
    r_tip: float
    clearance: float = 0.0  # tip gap k [m]; 0 for vanes
    shrouded: bool = False
    n_seals: int = 1


@dataclass
class RowFlow:
    """From the meanline module, at midspan, in the row frame."""
    kind: str               # 'vane' or 'rotor'
    angle_in: float         # inlet gas angle [deg] (signed)
    angle_out: float        # exit gas angle [deg] (signed)
    M_in: float
    M_out: float
    P_in: float             # static pressure at row inlet
    P_out: float            # static pressure at row exit (same unit as P_in)
    Re: float               # chord-based, exit conditions
    gamma: float            # representative gamma for the row
    M_in_hub: float | None = None   # from the spanwise module, if available


@dataclass
class LossOptions:
    supersonic_model: str = 'ko'            # 'ko' (f_Ma on Yp), 'normal_shock' or 'none'
    supersonic_factor: float = 1.0          # 'normal_shock' only: fraction of normal-shock loss


@dataclass
class RowLoss:
    kind: str
    Y_profile: float        # 0.914 * 2/3 * Yp_AMDC * Kp * f(Re)
    Y_shock_hub: float      # 0.914 * Y_shock * f(Re)   (KO inlet/hub shock term)
    Y_secondary: float
    Y_trailing_edge: float
    Y_supersonic: float     # supersonic-exit loss, M_out > 1: (f_Ma - 1) * Yp for 'ko'
    Y_tip: float            # shrouded rotors only (Y form)
    Y_total: float          # total-pressure loss coefficient of the row
    xi: float               # equivalent kinetic-energy loss coefficient (meanline input)
    deta_tip: float         # unshrouded rotors: fractional efficiency debit
    Kp: float
    f_Re: float
    f_Ma: float
    notes: tuple = ()       # inputs outside the digitized chart ranges (values clamped)

    def as_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Helpers shared with the meanline coupling
# ---------------------------------------------------------------------------
def _ko_angles(a_in, a_out):
    """Signed angles -> KO convention: exit angle positive, inlet positive
    when on the opposite side of axial (flow turning through axial)."""
    s = 1.0 if a_out >= 0 else -1.0
    return -a_in * s, abs(a_out)


def _p0_over_p(M, g):
    return (1 + (g - 1) / 2 * M**2)**(g / (g - 1))


def kinetic_loss_from_Y(Y, M_out, gamma):
    """Convert Y = (P01-P02)/(P02-P2) to xi = 1 - V^2/V_is^2 at the same exit
    static pressure and total temperature (ideal gas, row frame)."""
    e = (gamma - 1) / gamma
    p02_p2 = _p0_over_p(M_out, gamma)
    p01_p2 = p02_p2 * (1 + Y) - Y
    return 1 - (1 - p02_p2**-e) / (1 - p01_p2**-e)


def zweifel_pitch_chord(angle_in, angle_out, axial_chord_ratio, zweifel=0.8):
    """Pitch-to-chord ratio s/c from the Zweifel criterion,
    Zw = 2 (s/bx) cos^2(a2) (tan a1 + tan a2), with flow angles (signed)."""
    a1, a2 = np.radians(_ko_angles(angle_in, angle_out))
    s_bx = zweifel / (2 * np.cos(a2)**2 * (np.tan(a1) + np.tan(a2)))
    return s_bx * axial_chord_ratio


# ---------------------------------------------------------------------------
# Kacker-Okapuu components
# ---------------------------------------------------------------------------
def _chart(curves, sc, a2, label, notes):
    """Yp from a family of digitized curves: linear in s/c along each alpha2
    curve, then linear in alpha2. Outside the digitized range values are
    clamped to the nearest curve end and a note is recorded."""
    angles = sorted(curves)
    if not angles[0] <= a2 <= angles[-1]:
        notes.append(f'{label}: alpha2 = {a2:.1f} deg outside {angles[0]:g}-{angles[-1]:g}')
    vals = []
    for a in angles:
        x, y = curves[a]
        vals.append(np.interp(sc, x, y))
    lo = max(curves[a][0][0] for a in angles)
    hi = min(curves[a][0][-1] for a in angles)
    if not lo <= sc <= hi:
        notes.append(f'{label}: s/c = {sc:.3f} outside {lo:.2f}-{hi:.2f}')
    return float(np.interp(a2, angles, vals))


def yp_nozzle(sc, a2, notes=None):
    """AM/KO profile loss for beta1 = 0, tmax/c = 0.2 (KO Fig. 1). a2 in deg."""
    return _chart(FIG01_NOZZLE, sc, a2, 'Fig. 1', [] if notes is None else notes)


def yp_impulse(sc, a2, notes=None):
    """AM/KO profile loss for beta1 = alpha2, tmax/c = 0.2 (KO Fig. 2). a2 in deg."""
    return _chart(FIG02_IMPULSE, sc, a2, 'Fig. 2', [] if notes is None else notes)


def _f_reynolds(Re):
    if Re <= 2e5:
        return (Re / 2e5)**-0.4
    if Re < 1e6:
        return 1.0
    return (Re / 1e6)**-0.2


def _kp(M1, M2):
    K1 = 1.0 if M2 <= 0.2 else max(0.0, 1 - 1.25 * (M2 - 0.2))   # K1 = 0 for M2 > 1
    K2 = (M1 / M2)**2
    return 1 - K2 * (1 - K1)


def _y_shock_hub(flow, geo, g):
    rh_rt = geo.r_hub / geo.r_tip
    M1h = flow.M_in_hub
    if M1h is None:
        M1h = flow.M_in * (1 + HUB_MACH_K[flow.kind] * abs(rh_rt - 1)**2.2)
    if M1h <= 0.4:
        return 0.0
    dY = 0.75 * (M1h - 0.4)**1.75
    ratio = (1 - _p0_over_p(flow.M_in, g)) / (1 - _p0_over_p(flow.M_out, g))
    return rh_rt * dY * (flow.P_in / flow.P_out) * ratio


def _y_trailing_edge(t_o, b1, a2, M2, g, notes):
    if not 0 <= t_o <= 0.4:
        notes.append(f'Fig. 14: t_TE/o = {t_o:.3f} outside 0-0.4')
    r = b1 / a2
    # max(0, .) removes small negative digitizing noise near t/o = 0
    d_n = max(0.0, np.interp(t_o, FIG14_NOZZLE[:, 0], FIG14_NOZZLE[:, 1]))
    d_i = max(0.0, np.interp(t_o, FIG14_IMPULSE[:, 0], FIG14_IMPULSE[:, 1]))
    dphi2 = d_n + abs(r) * r * (d_i - d_n)
    k = (g - 1) / 2 * M2**2
    num = (1 - k * (1 / (1 - dphi2) - 1))**(-g / (g - 1)) - 1
    den = 1 - (1 + k)**(-g / (g - 1))
    return num / den


def _f_mach(M2):
    """Supersonic-exit correction on the profile loss (KO as implemented in
    TurboFlow): f_Ma = 1 + 60 (M2 - 1)^2 for M2 > 1."""
    return 1.0 if M2 <= 1.0 else 1 + 60 * (M2 - 1)**2


def _y_normal_shock(M2, g, opts):
    """Alternative supersonic-exit loss: a fraction of the normal-shock
    total-pressure loss at M2 (factor = 1 is an upper bound, because the real
    trailing-edge shocks are oblique)."""
    if M2 <= 1.0:
        return 0.0
    M = M2
    p0r = (((g + 1) * M**2 / ((g - 1) * M**2 + 2))**(g / (g - 1))
           * ((g + 1) / (2 * g * M**2 - (g - 1)))**(1 / (g - 1)))
    p0p = _p0_over_p(M, g)
    return opts.supersonic_factor * (1 - p0r) * p0p / (p0p - 1)


def row_loss(geo: RowGeometry, flow: RowFlow, opts: LossOptions = LossOptions()) -> RowLoss:
    """Kacker-Okapuu loss of one blade row at midspan."""
    g = flow.gamma
    a1, a2 = _ko_angles(flow.angle_in, flow.angle_out)
    b1, _ = _ko_angles(geo.metal_in, flow.angle_out)
    ra1, ra2, rb1 = np.radians([a1, a2, b1])

    # --- profile ---
    ratio = b1 / a2
    tc = np.clip(geo.tmax_chord, 0.15, 0.25)          # KO validity range
    notes = []
    yn = yp_nozzle(geo.pitch_chord, a2, notes)
    yi = yp_impulse(geo.pitch_chord, a2, notes) if ratio != 0 else yn   # unused when beta1 = 0
    Yp_amdc = (yn + abs(ratio) * ratio * (yi - yn)) * (tc / 0.2)**ratio
    Kp = _kp(flow.M_in, flow.M_out)
    fRe = _f_reynolds(flow.Re)
    Y_prof = 0.914 * (2 / 3) * Yp_amdc * Kp * fRe
    Y_sh = 0.914 * _y_shock_hub(flow, geo, g) * fRe

    # --- secondary ---
    am = np.arctan((np.tan(ra2) - np.tan(ra1)) / 2)
    CL_sc = 2 * (np.tan(ra1) + np.tan(ra2)) * np.cos(am)
    hc = geo.height / geo.chord
    fAR = (1 - 0.25 * np.sqrt(2 - hc)) / hc if hc <= 2 else 1 / hc
    loading = CL_sc**2 * np.cos(ra2)**2 / np.cos(am)**3
    Ys_amdc = 0.0334 * fAR * (np.cos(ra2) / np.cos(rb1)) * loading
    Ks = 1 - (geo.axial_chord / geo.height)**2 * (1 - Kp)
    Y_sec = 1.2 * Ys_amdc * Ks

    # --- trailing edge and supersonic exit ---
    Y_te = _y_trailing_edge(geo.te_throat, b1, a2, flow.M_out, g, notes)
    fMa = 1.0
    if opts.supersonic_model == 'ko':
        fMa = _f_mach(flow.M_out)
        Y_ss = (fMa - 1) * (Y_prof + Y_sh)      # Y = f_Re * f_Ma * Yp, split out for reporting
    elif opts.supersonic_model == 'normal_shock':
        Y_ss = _y_normal_shock(flow.M_out, g, opts)
    else:
        Y_ss = 0.0

    # --- tip clearance ---
    Y_tip, deta = 0.0, 0.0
    if flow.kind == 'rotor' and geo.clearance > 0:
        if geo.shrouded:
            k_eff = geo.clearance / geo.n_seals**0.42
            Y_tip = 0.37 * (geo.chord / geo.height) * (k_eff / geo.chord)**0.78 * loading
        else:
            r_mean = 0.5 * (geo.r_hub + geo.r_tip)
            deta = 0.93 * geo.clearance / (geo.height * np.cos(ra2)) * geo.r_tip / r_mean

    Y_tot = Y_prof + Y_sh + Y_sec + Y_te + Y_ss + Y_tip
    xi = kinetic_loss_from_Y(Y_tot, flow.M_out, g)
    return RowLoss(flow.kind, Y_prof, Y_sh, Y_sec, Y_te, Y_ss, Y_tip, Y_tot, xi, deta, Kp, fRe, fMa,
                   tuple(notes))


def apply_tip_clearance(eta, row_losses):
    """Apply KO unshrouded tip-clearance debits: eta = eta0 * prod(1 - d_eta)."""
    for rl in row_losses:
        eta *= (1 - rl.deta_tip)
    return eta


def print_breakdown(row_losses, names=None):
    keys = ['Y_profile', 'Y_shock_hub', 'Y_secondary', 'Y_trailing_edge',
            'Y_supersonic', 'Y_tip', 'Y_total', 'xi', 'deta_tip', 'Kp', 'f_Re', 'f_Ma']
    names = names or [rl.kind for rl in row_losses]
    print(f'{"Loss component":<18}' + ''.join(f'{n:>12}' for n in names))
    print('-' * (18 + 12 * len(names)))
    for k in keys:
        print(f'{k:<18}' + ''.join(f'{getattr(rl, k):12.4f}' for rl in row_losses))
    for n, rl in zip(names, row_losses):
        for note in rl.notes:
            print(f'NOTE ({n}): {note}')


# ---------------------------------------------------------------------------
# Ideal-gas relations with the GasProps constants (C_p, gamma, R)
# ---------------------------------------------------------------------------
def _entropy(T, P, gas: GasProps):
    """Specific entropy relative to an arbitrary reference [J/kg-K]."""
    return gas.C_p * np.log(T) - gas.R * np.log(P)


def _dh_isentropic(T0, P0, P2, gas: GasProps):
    """Ideal enthalpy drop expanding from (T0, P0) to pressure P2 [J/kg]."""
    e = (gas.gamma - 1) / gas.gamma
    return gas.C_p * T0 * (1 - (P2 / P0)**e)


# ---------------------------------------------------------------------------
# Coolant mixing (constant-area control volume, row frame)
# ---------------------------------------------------------------------------
@dataclass
class MixResult:
    m: float
    T0: float
    P0: float
    P: float
    T: float
    Vx: float
    Vt: float
    M: float
    P0_main_before: float
    ds_gen: float           # entropy generated per kg of mixed flow [J/kg-K]

    def as_dict(self):
        return asdict(self)


def mix_coolant(m_g, T0_g, P_g, Vx_g, Vt_g, m_c, T0_c, P0_c, Vx_c, Vt_c, area,
                gas: GasProps = GAS):
    """Mix coolant into the main stream at constant area, conserving mass,
    energy and axial/tangential momentum. Pressures in Pa, area in m^2.

    Work in the row frame: absolute for vanes; relative for rotors (then T0 is
    the relative total temperature and the coolant T0_c, Vx_c, Vt_c are in the
    relative frame too). P_g is the main-stream static pressure before mixing.
    Vx_c, Vt_c describe the coolant's ejection velocity (e.g. from ejection
    angle and blowing ratio). Returns the mixed-out state (subsonic-axial root).
    """
    cp, R, k = gas.C_p, gas.R, gas.gamma
    m = m_g + m_c
    T0 = (m_g * T0_g + m_c * T0_c) / m                 # energy, constant C_p
    Vt = (m_g * Vt_g + m_c * Vt_c) / m                 # tangential momentum
    Fx = m_g * Vx_g + m_c * Vx_c + P_g * area          # axial momentum + pressure force

    # Continuity + energy + axial momentum -> quadratic in Vx:
    # with P = (Fx - m Vx)/A and T = P A Vx/(m R):  cp T + (Vx^2 + Vt^2)/2 = cp T0
    a, b, c = 0.5 - cp / R, cp * Fx / (m * R), Vt**2 / 2 - cp * T0
    if b**2 - 4 * a * c < 0:
        raise ValueError('mixing is choked: no solution at this area')
    Vx = min(r for r in np.roots([a, b, c]).real if r > 0)   # subsonic-axial branch
    P = (Fx - m * Vx) / area
    T = P * area / (m * R) * Vx

    M = np.hypot(Vx, Vt) / np.sqrt(k * R * T)
    P0 = P * (T0 / T)**(k / (k - 1))

    # main-stream P0 before mixing, entropy generation
    Tg = T0_g - (Vx_g**2 + Vt_g**2) / (2 * cp)
    P0_g = P_g * (T0_g / Tg)**(k / (k - 1))
    s_in = (m_g * _entropy(T0_g, P0_g, gas) + m_c * _entropy(T0_c, P0_c, gas)) / m
    ds = _entropy(T0, P0, gas) - s_in
    return MixResult(m, T0, P0, P, T, Vx, Vt, M, P0_g, ds)


# ---------------------------------------------------------------------------
# Efficiencies
# ---------------------------------------------------------------------------
def cooled_efficiency(W, main, coolants, P3, P03, gas: GasProps = GAS):
    """Total-to-total and total-to-static efficiency of a cooled turbine.

    W        : actual shaft power [W]
    main     : (m, T0, P0) of the main gas at turbine inlet [kg/s, K, Pa]
    coolants : list of (m, T0, P0) for each coolant stream at injection
    P3, P03  : static and (mixed-out) total pressure at turbine exit [Pa]

    Ideal power = sum over all streams of m * isentropic dh to the exit
    pressure (the 'fully expanded' definition; see Young & Horlock 2006).
    Also returns the main-stream-only ('uncooled basis') values for comparison.
    """
    streams = [main] + list(coolants)
    Wtt = sum(m * _dh_isentropic(T0, P0, P03, gas) for m, T0, P0 in streams)
    Wts = sum(m * _dh_isentropic(T0, P0, P3, gas) for m, T0, P0 in streams)
    m0, T00, P00 = main
    return dict(
        eta_tt=W / Wtt, eta_ts=W / Wts,
        eta_tt_main_only=W / (m0 * _dh_isentropic(T00, P00, P03, gas)),
        eta_ts_main_only=W / (m0 * _dh_isentropic(T00, P00, P3, gas)),
        coolant_fraction=sum(s[0] for s in coolants) / m0,
    )
