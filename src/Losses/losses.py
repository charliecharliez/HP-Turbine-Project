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
Aungier, Turbine Aerodynamics, ASME Press (2006), ch. 7 (curve fits of AM charts).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np

from Meanline.gas_props import GAS, GasProps

# ---------------------------------------------------------------------------
# Curve-fit data you MUST verify against the KO paper before trusting results
# ---------------------------------------------------------------------------
# KO Fig. (trailing-edge energy coefficient Delta-phi^2 vs t_TE/o).
# PLACEHOLDER values of the right shape and magnitude: digitize the paper's
# two curves and replace these arrays.
TE_T_O = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
TE_DPHI2_NOZZLE = np.array([0.0, 0.009, 0.025, 0.047, 0.075])   # beta1 = 0
TE_DPHI2_IMPULSE = np.array([0.0, 0.016, 0.044, 0.083, 0.130])  # beta1 = alpha2

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
    supersonic_model: str = 'normal_shock'  # 'normal_shock' or 'none'
    supersonic_factor: float = 1.0          # fraction of normal-shock loss (1 = upper bound)


@dataclass
class RowLoss:
    kind: str
    Y_profile: float        # 0.914 * 2/3 * Yp_AMDC * Kp * f(Re)
    Y_shock_hub: float      # 0.914 * Y_shock * f(Re)   (KO inlet/hub shock term)
    Y_secondary: float
    Y_trailing_edge: float
    Y_supersonic: float     # trailing-edge shock system for M_out > 1 (beyond KO)
    Y_tip: float            # shrouded rotors only (Y form)
    Y_total: float          # total-pressure loss coefficient of the row
    xi: float               # equivalent kinetic-energy loss coefficient (meanline input)
    deta_tip: float         # unshrouded rotors: fractional efficiency debit
    Kp: float
    f_Re: float

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
def yp_nozzle(sc, a2):
    """AM profile loss, beta1 = 0 (Aungier fit). a2 in deg (magnitude)."""
    sc_min = 0.46 + a2 / 77 if a2 <= 30 else 0.614 + a2 / 130
    X = sc - sc_min
    A = 0.025 + (27 - a2) / 530 if a2 <= 27 else 0.025 + (27 - a2) / 3085
    B = 0.1583 - a2 / 1640
    if a2 <= 30:
        C = 0.08 * ((a2 / 30)**2 - 1)
        return A + B * X**2 + C * X**3
    n = 1 + a2 / 30
    return A + B * abs(X)**n


def yp_impulse(sc, a2):
    """AM profile loss, beta1 = alpha2 (Aungier fit). a2 in deg (magnitude)."""
    sc_min = 0.224 + 1.575 * (a2 / 90) - (a2 / 90)**2
    X = sc - sc_min
    A = 0.242 - a2 / 151 + (a2 / 127)**2
    B = 0.3 + (30 - a2) / 50 if a2 <= 30 else 0.3 + (30 - a2) / 275
    C = 0.88 - a2 / 42.4 + (a2 / 72.8)**2
    return A + B * X**2 - C * X**3


def _f_reynolds(Re):
    if Re <= 2e5:
        return (Re / 2e5)**-0.4
    if Re < 1e6:
        return 1.0
    return (Re / 1e6)**-0.2


def _kp(M1, M2):
    K1 = 1.0 if M2 <= 0.2 else 1 - 1.25 * (M2 - 0.2)
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


def _y_trailing_edge(t_o, b1, a2, M2, g):
    r = b1 / a2
    d_n = np.interp(t_o, TE_T_O, TE_DPHI2_NOZZLE)
    d_i = np.interp(t_o, TE_T_O, TE_DPHI2_IMPULSE)
    dphi2 = d_n + abs(r) * r * (d_i - d_n)
    k = (g - 1) / 2 * M2**2
    num = (1 - k * (1 / (1 - dphi2) - 1))**(-g / (g - 1)) - 1
    den = 1 - (1 + k)**(-g / (g - 1))
    return num / den


def _y_supersonic(M2, g, opts):
    """Trailing-edge shock loss for supersonic exit. KO's Kp was built for
    subsonic exits, so this is added explicitly. Default: a fraction of the
    normal-shock total-pressure loss at M2 (factor = 1 is an upper bound,
    because the real TE shocks are oblique). Replace with your chosen model."""
    if opts.supersonic_model == 'none' or M2 <= 1.0:
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
    yn, yi = yp_nozzle(geo.pitch_chord, a2), yp_impulse(geo.pitch_chord, a2)
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
    Y_te = _y_trailing_edge(geo.te_throat, b1, a2, flow.M_out, g)
    Y_ss = _y_supersonic(flow.M_out, g, opts)

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
    return RowLoss(flow.kind, Y_prof, Y_sh, Y_sec, Y_te, Y_ss, Y_tip, Y_tot, xi, deta, Kp, fRe)


def apply_tip_clearance(eta, row_losses):
    """Apply KO unshrouded tip-clearance debits: eta = eta0 * prod(1 - d_eta)."""
    for rl in row_losses:
        eta *= (1 - rl.deta_tip)
    return eta


def print_breakdown(row_losses, names=None):
    keys = ['Y_profile', 'Y_shock_hub', 'Y_secondary', 'Y_trailing_edge',
            'Y_supersonic', 'Y_tip', 'Y_total', 'xi', 'deta_tip', 'Kp', 'f_Re']
    names = names or [rl.kind for rl in row_losses]
    print(f'{"Loss component":<18}' + ''.join(f'{n:>12}' for n in names))
    print('-' * (18 + 12 * len(names)))
    for k in keys:
        print(f'{k:<18}' + ''.join(f'{getattr(rl, k):12.4f}' for rl in row_losses))


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
