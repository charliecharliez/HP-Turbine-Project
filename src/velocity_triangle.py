"""Velocity triangles for turbomachinery, plotted with matplotlib.

A velocity triangle relates the absolute flow velocity C, the relative flow
velocity W seen by the blade, and the blade speed U through C = W + U.

Sign convention
---------------
* Flow angles alpha (of C) and beta (of W) are measured from the meridional
  direction, in degrees.
* Tangential components (C_u, W_u) and both angles are positive in the
  direction of blade rotation, i.e. the direction of U.
* Hence C_u = W_u + U, and C and W share the meridional component C_m.

In a plot, rotation points right and the flow leaves a common apex heading
down, so a positive angle leans right and a negative one leans left. This is
what lets the triangles of a stator and rotor share one diagram.

Example
-------
>>> rotor_in = VelocityTriangle.from_U_C_alpha(680.7, 931.9, 71.0, station=2)
>>> rotor_out = VelocityTriangle.from_U_W_beta(816.8, 826.0, -61.0, station=3)
>>> VelocityTriangle.plot_combined(rotor_in, rotor_out)
"""
from __future__ import annotations

import math
from dataclasses import KW_ONLY, dataclass

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, FancyArrowPatch

__all__ = ["VelocityTriangle"]

# First three slots of a colour-blind-validated categorical palette.
_C_COLOR = "#2a78d6"  # absolute velocity (blue)
_W_COLOR = "#eb6834"  # relative velocity (orange)
_U_COLOR = "#1baf7a"  # blade speed (aqua)
_MUTED = "#898781"  # meridional dimension
_HAIRLINE = "#c3c2b7"  # reference and extension lines
_LINESTYLES = ("-", (0, (5, 2.5)))  # first triangle solid, second dashed
_FONTSIZE = 9


@dataclass(frozen=True)
class VelocityTriangle:
    """A velocity triangle, stored by its components U, C_m and C_u.

    Calling the class directly takes those three components. The ``from_*``
    class methods build a triangle from other combinations of quantities.

    Parameters
    ----------
    U : blade speed, positive in the direction of rotation.
    C_m : meridional velocity, must be positive.
    C_u : tangential component of the absolute velocity, + along U.
    station : station number, shown as a subscript (C_m2, alpha_2, ...).
    unit : unit of speed used in labels.
    """

    U: float
    C_m: float
    C_u: float
    _: KW_ONLY
    station: int | None = None
    unit: str = "m/s"

    def __post_init__(self):
        for name in ("U", "C_m", "C_u"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")
            object.__setattr__(self, name, value)
        if self.C_m <= 0:
            raise ValueError(f"C_m must be positive (flow through the blade row), got {self.C_m}")

    # ----------------------------------------------------------- constructors
    @classmethod
    def from_U_Cm_beta(cls, U, C_m, beta, *, station=None, unit="m/s") -> VelocityTriangle:
        """From blade speed U, meridional velocity C_m and relative flow angle beta [deg]."""
        return cls(U, C_m, U + C_m * _tan(beta, "beta"), station=station, unit=unit)

    @classmethod
    def from_U_Cm_alpha(cls, U, C_m, alpha, *, station=None, unit="m/s") -> VelocityTriangle:
        """From blade speed U, meridional velocity C_m and absolute flow angle alpha [deg]."""
        return cls(U, C_m, C_m * _tan(alpha, "alpha"), station=station, unit=unit)

    @classmethod
    def from_U_C_alpha(cls, U, C, alpha, *, station=None, unit="m/s") -> VelocityTriangle:
        """From blade speed U, absolute velocity C and absolute flow angle alpha [deg].

        Typical at a stator exit, where the stator sets C and alpha.
        """
        a = math.radians(_check_angle(alpha, "alpha"))
        return cls(U, C * math.cos(a), C * math.sin(a), station=station, unit=unit)

    @classmethod
    def from_U_W_beta(cls, U, W, beta, *, station=None, unit="m/s") -> VelocityTriangle:
        """From blade speed U, relative velocity W and relative flow angle beta [deg].

        Typical at a rotor exit, where the rotor sets W and beta.
        """
        b = math.radians(_check_angle(beta, "beta"))
        return cls(U, W * math.cos(b), U + W * math.sin(b), station=station, unit=unit)

    @classmethod
    def from_U_C_W(cls, U, C, W, *, station=None, unit="m/s") -> VelocityTriangle:
        """From the three side lengths: blade speed U and the magnitudes of C and W.

        C_u follows from C^2 - W^2 = 2 U C_u - U^2, and C_m is taken positive,
        so the signs of the flow angles come out of the geometry.
        """
        if U == 0 or C <= 0 or W <= 0:
            raise ValueError(f"Need U != 0 and C, W > 0, got U={U}, C={C}, W={W}")
        C_u = (C**2 - W**2 + U**2) / (2 * U)
        C_m_squared = C**2 - C_u**2
        if C_m_squared <= 0:
            raise ValueError(f"U={U}, C={C}, W={W} do not close a triangle")
        return cls(U, math.sqrt(C_m_squared), C_u, station=station, unit=unit)

    @classmethod
    def from_U_alpha_beta(cls, U, alpha, beta, *, station=None, unit="m/s") -> VelocityTriangle:
        """From blade speed U and both flow angles [deg], using U = C_m (tan alpha - tan beta)."""
        tan_a = _tan(alpha, "alpha")
        diff = tan_a - _tan(beta, "beta")
        if diff == 0 or U / diff <= 0:
            raise ValueError(f"No triangle with positive C_m for U={U}, alpha={alpha}, beta={beta}: "
                             "for U > 0, alpha must be larger than beta")
        C_m = U / diff
        return cls(U, C_m, C_m * tan_a, station=station, unit=unit)

    # ------------------------------------------------------ derived quantities
    @property
    def W_u(self) -> float:
        """Tangential component of the relative velocity, C_u - U."""
        return self.C_u - self.U

    @property
    def C(self) -> float:
        """Absolute velocity magnitude."""
        return math.hypot(self.C_m, self.C_u)

    @property
    def W(self) -> float:
        """Relative velocity magnitude."""
        return math.hypot(self.C_m, self.W_u)

    @property
    def alpha(self) -> float:
        """Absolute flow angle [deg] from meridional, + in the direction of U."""
        return math.degrees(math.atan2(self.C_u, self.C_m))

    @property
    def beta(self) -> float:
        """Relative flow angle [deg] from meridional, + in the direction of U."""
        return math.degrees(math.atan2(self.W_u, self.C_m))

    # ---------------------------------------------------------------- plotting
    def plot(self, ax: Axes | None = None, *, title: str | None = None,
             precision: int = 1, legend: bool = True) -> Axes:
        """Draw this triangle with every quantity labelled; returns the Axes.

        A new figure is created when ``ax`` is None. ``precision`` is the number
        of decimals shown for speeds and angles. The legend sits outside the
        axes on the right.
        """
        if title is None:
            title = "Velocity triangle" + ("" if self.station is None else f", station {self.station}")
        return _draw([self], ax, title, precision, legend)

    @staticmethod
    def plot_combined(first: VelocityTriangle, second: VelocityTriangle,
                      ax: Axes | None = None, *, title: str | None = None,
                      precision: int = 1, legend: bool = True) -> Axes:
        """Draw two triangles from a common apex, e.g. a rotor inlet and exit.

        The first triangle is drawn solid and the second dashed; the signs of
        the flow angles decide which side of the meridional line each vector
        falls on. Both triangles must use the same unit. Returns the Axes.
        """
        if first.unit != second.unit:
            raise ValueError(f"Cannot share one diagram between units {first.unit!r} and {second.unit!r}")
        if title is None:
            if first.station is None or second.station is None:
                title = "Combined velocity triangles"
            else:
                title = f"Velocity triangles, stations {first.station} and {second.station}"
        return _draw([first, second], ax, title, precision, legend)

    def _sym(self, base: str, sub: str = "") -> str:
        """Mathtext symbol with the station in the subscript, e.g. $C_{m2}$."""
        sub = f"{sub}{'' if self.station is None else self.station}"
        return f"${base}_{{{sub}}}$" if sub else f"${base}$"


def _check_angle(angle, name):
    if not -90 < angle < 90:
        raise ValueError(f"{name} must lie strictly between -90 and 90 degrees, got {angle}")
    return angle


def _tan(angle, name):
    return math.tan(math.radians(_check_angle(angle, name)))


def _fmt(value, precision):
    """Fixed-point number with a typographic minus and no '-0.0'."""
    text = f"{value:.{precision}f}"
    if float(text) == 0:
        text = text.lstrip("-")
    return text.replace("-", "−")


# =================================================================== drawing
def _draw(triangles, ax, title, precision, legend):
    ink = mpl.rcParams["text.color"]
    unit = triangles[0].unit

    # Layout in data units: dimension lines sit outside the triangles, spaced by `step`.
    xs = [0.0] + [x for t in triangles for x in (t.C_u, t.W_u)]
    x_lo, x_hi = min(xs), max(xs)
    depth = max(t.C_m for t in triangles)
    step = 0.075 * max(x_hi - x_lo, depth)

    # C_m columns go beside each triangle, on the side it leans towards.
    sides = [1 if t.C_u + t.W_u >= 0 else -1 for t in triangles]
    col_x, count = [], {1: 0, -1: 0}
    for side in sides:
        count[side] += 1
        col_x.append((x_hi if side > 0 else x_lo) + side * count[side] * step)
    # C_u rows go below the lowest base, one per triangle.
    # The extra half step leaves room for the U label under the base.
    row_y = [-depth - (k + 1.5) * step for k in range(len(triangles))]

    x_min, x_max = min([x_lo, *col_x]) - step, max([x_hi, *col_x]) + step
    y_min, y_max = row_y[-1] - 0.8 * step, 0.6 * step

    if ax is None:
        # Size the axes to the diagram's aspect ratio, tall enough for the legend.
        width = 7.4 if len(triangles) == 1 else 8.6  # inches of axes; the legend goes beside them
        height = width * (y_max - y_min) / (x_max - x_min) + 0.8
        if legend:
            height = max(height, 0.21 * 7 * len(triangles) + 1.0)
        _, ax = plt.subplots(figsize=(width + 3.6, min(max(height, 3.5), 9.0)), layout="constrained")
    fig = ax.figure

    obstacles = []  # (polyline in data units, weight) that labels should not cross
    vector_labels, angle_labels, dim_labels = [], [], []
    handles = []

    def line(points, weight, **kw):
        points = np.asarray(points, float)
        ax.plot(points[:, 0], points[:, 1], solid_capstyle="butt", **kw)
        obstacles.append((points, weight))

    # Meridional reference through the apex; angles start here and rows measure from it.
    line([(0, 0), (0, row_y[-1] - 0.4 * step)], 0.5, color=_HAIRLINE, lw=0.8, zorder=1)

    for i, t in enumerate(triangles):
        ls = _LINESTYLES[i]
        apex, c_tip, w_tip = np.zeros(2), np.array([t.C_u, -t.C_m]), np.array([t.W_u, -t.C_m])

        # The triangle itself: C and W from the apex, U from the tip of W to the tip of C.
        for p, q, color in ((apex, c_tip, _C_COLOR), (apex, w_tip, _W_COLOR), (w_tip, c_tip, _U_COLOR)):
            _arrow(ax, p, q, color, ls, lw=2.0, head=14, zorder=4)
            obstacles.append((np.array([p, q]), 2.0))
        centroid = (apex + c_tip + w_tip) / 3
        vector_labels += [
            (f"{t._sym('C')} = {_fmt(t.C, precision)} {unit}", _along(apex, c_tip, centroid)),
            (f"{t._sym('W')} = {_fmt(t.W, precision)} {unit}", _along(apex, w_tip, centroid)),
            (f"{t._sym('U')} = {_fmt(t.U, precision)} {unit}", _along(w_tip, c_tip, centroid)),
        ]
        angle_labels += [(t, t.alpha, r"\alpha", _C_COLOR, ls), (t, t.beta, r"\beta", _W_COLOR, ls)]

        # Absolute tangential component: signed arrow from the meridional line.
        y = row_y[i]
        line([(t.C_u, -t.C_m), (t.C_u, y - 0.3 * step)], 0.5, color=_HAIRLINE, lw=0.8, zorder=1)
        if t.C_u != 0:
            _arrow(ax, (0, y), (t.C_u, y), _C_COLOR, ls, lw=1.0, head=8, zorder=3)
        obstacles.append((np.array([(0, y), (t.C_u, y)]), 1.5))
        sign = 1 if t.C_u >= 0 else -1
        dim_labels.append((f"{t._sym('C', 'u')} = {_fmt(t.C_u, precision)} {unit}",
                           [((t.C_u / 2, y), (0, 1)), ((t.C_u / 2, y), (0, -1)),
                            ((t.C_u, y), (sign, 0))], 0))

        # Meridional component: double-headed arrow beside the triangle.
        x_c, side = col_x[i], sides[i]
        x_edge = max(t.C_u, t.W_u, 0) if side > 0 else min(t.C_u, t.W_u, 0)
        x_ext = x_c + side * 0.3 * step
        line([(0, 0), (x_ext, 0)], 0.5, color=_HAIRLINE, lw=0.8, zorder=1)
        line([(x_edge, -t.C_m), (x_ext, -t.C_m)], 0.5, color=_HAIRLINE, lw=0.8, zorder=1)
        _arrow(ax, (x_c, 0), (x_c, -t.C_m), _MUTED, ls, lw=1.0, head=8, zorder=3, style="<|-|>")
        obstacles.append((np.array([(x_c, 0), (x_c, -t.C_m)]), 1.5))
        dim_labels.append((f"{t._sym('C', 'm')} = {_fmt(t.C_m, precision)} {unit}",
                           [((x_c, -t.C_m / 2), (side, 0)), ((x_c, -t.C_m / 2), (-side, 0))], 90))

        handles += _legend_entries(t, ls, precision)

    # Equal aspect keeps the angles true; the axes box shrinks to fit the limits.
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    ax.set_title(title, loc="left", fontsize=11, color=ink)
    if legend:
        ax.legend(handles=[h for h, _ in handles], labels=[s for _, s in handles],
                  loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False,
                  fontsize=8.5, handlelength=2.6, alignment="left",
                  title="Angles from meridional,\n+ in the direction of U", title_fontsize=8.5)

    # Labels are placed in screen space, so the layout must be final first.
    fig.draw_without_rendering()
    style = dict(fontsize=_FONTSIZE, color=ink, zorder=6, annotation_clip=False,
                 bbox=dict(boxstyle="round,pad=0.15", fc=fig.get_facecolor(), ec="none", alpha=0.85))
    placer = _Placer(ax, obstacles, style)

    # Most constrained first: angles must sit near the apex, vectors are long.
    _place_angles(ax, placer, angle_labels, triangles, precision)
    for text, candidates in vector_labels:
        placer.place(text, candidates)
    for text, candidates, rotation in dim_labels:
        placer.place(text, [(xy, d, None) for xy, d in candidates], rotation=rotation)
    return ax


def _arrow(ax, p, q, color, ls, *, lw, head, zorder, style="-|>"):
    ax.add_patch(FancyArrowPatch(tuple(p), tuple(q), arrowstyle=style, mutation_scale=head,
                                 color=color, lw=lw, linestyle=ls, shrinkA=0, shrinkB=0,
                                 joinstyle="miter", capstyle="butt", zorder=zorder))


def _along(p, q, centroid):
    """Label candidates beside segment p->q: outside the triangle first, mid-length first."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    d = q - p
    length = math.hypot(*d)
    n = np.array([-d[1], d[0]]) / length if length else np.array([0.0, -1.0])
    if np.dot(n, (p + q) / 2 - centroid) < 0:
        n = -n
    fractions = (0.5, 0.62, 0.38, 0.74, 0.26, 0.86, 0.14)
    beside = [(p + f * d, s * n, None) for s in (1, -1) for f in fractions]
    # Around the arrow tip, which often has free space when the sides do not.
    compass = [(-1, -1), (1, -1), (-1, 0), (1, 0), (0, -1), (-1, 1), (1, 1)]
    around_tip = [(q, np.array(c, float) / math.hypot(*c), None) for c in compass]
    farther = [(p + f * d, 3 * s * n, None) for s in (1, -1) for f in fractions]
    return beside + around_tip + farther


def _place_angles(ax, placer, angle_labels, triangles, precision):
    """Draw each angle's arc at the first radius where its label fits.

    Arcs on the same side of the meridional line overlap in angle, so each one
    there gets its own radius.
    """
    r_ref = min(t.C_m for t in triangles)
    radii = [f * r_ref for f in (0.3, 0.42, 0.54, 0.66, 0.78, 0.9, 1.02)]
    used = {1: set(), -1: set()}
    for t, angle, symbol, color, ls in angle_labels:
        side = 1 if angle >= 0 else -1
        free = [r for r in radii if r not in used[side]] or radii
        phi = math.radians(-90 + angle)  # direction of the vector, CCW from +x
        start = -math.pi / 2
        candidates, owner, last_resort = [], [], []
        for r in free:
            arc = r * np.column_stack(_unit(np.linspace(start, phi, 24)))
            for f in (0.5, 0.7, 0.3, 0.85, 0.15):
                th = start + f * (phi - start)
                candidates.append((r * np.array(_unit(th)), np.array(_unit(th)), arc))
                owner.append(r)
            # Just beyond the vector end of the arc, for small angles.
            candidates.append((r * np.array(_unit(phi)), side * np.array(_tangent(phi)), arc))
            owner.append(r)
            # Across the meridional line, where it could be mistaken for the other side.
            last_resort.append(((r * np.array(_unit(start)), -side * np.array(_tangent(start)), arc), r))
        candidates += [c for c, _ in last_resort]
        owner += [r for _, r in last_resort]

        text = f"{t._sym(symbol)} = {_fmt(angle, precision)}°"
        r = owner[placer.place(text, candidates)]
        used[side].add(r)
        if abs(angle) >= 0.5:
            lo, hi = sorted((-90.0, -90.0 + angle))
            ax.add_patch(Arc((0, 0), 2 * r, 2 * r, theta1=lo, theta2=hi, color=color,
                             lw=1.2, linestyle=ls, zorder=3))
            placer.add_line(r * np.column_stack(_unit(np.linspace(start, phi, 24))), 1.0)


def _unit(theta):
    return np.cos(theta), np.sin(theta)


def _tangent(theta):
    """Unit tangent of a circle, pointing towards increasing theta."""
    return -math.sin(theta), math.cos(theta)


def _legend_entries(t, ls, precision):
    v = lambda value: f"{_fmt(value, precision)} {t.unit}"
    deg = lambda value: f"{_fmt(value, precision)}°"
    angle_marker = dict(ls="none", marker=r"$\angle$", markersize=11)
    return [
        (Line2D([], [], color=_C_COLOR, lw=2, ls=ls), f"Absolute velocity  {t._sym('C')} = {v(t.C)}"),
        (Line2D([], [], color=_W_COLOR, lw=2, ls=ls), f"Relative velocity  {t._sym('W')} = {v(t.W)}"),
        (Line2D([], [], color=_U_COLOR, lw=2, ls=ls), f"Blade speed  {t._sym('U')} = {v(t.U)}"),
        (Line2D([], [], color=_C_COLOR, **angle_marker), f"Absolute flow angle  {t._sym(r'\alpha')} = {deg(t.alpha)}"),
        (Line2D([], [], color=_W_COLOR, **angle_marker), f"Relative flow angle  {t._sym(r'\beta')} = {deg(t.beta)}"),
        (Line2D([], [], color=_MUTED, lw=1, ls=ls), f"Meridional velocity  {t._sym('C', 'm')} = {v(t.C_m)}"),
        (Line2D([], [], color=_C_COLOR, lw=1, ls=ls), f"Abs. tangential velocity  {t._sym('C', 'u')} = {v(t.C_u)}"),    ]


class _Placer:
    """Greedy label placement in screen space.

    Each label takes its first candidate that overlaps no earlier label, crosses
    no drawn line and stays inside the axes; failing that, the least bad one.
    A candidate is (anchor in data units, offset direction in units of GAP,
    optional polyline in data units that must also avoid earlier labels).
    """

    GAP = 5  # points between the anchor and the text, per unit of offset

    def __init__(self, ax, obstacles, style):
        self.ax, self.style = ax, style
        self.lines = [(ax.transData.transform(p), w) for p, w in obstacles]
        self.boxes = []
        self.pad = 2 * ax.figure.dpi / 72
        legend = ax.get_legend()
        if legend is not None:
            self.boxes.append(legend.get_window_extent())

    def add_line(self, points, weight):
        self.lines.append((self.ax.transData.transform(points), weight))

    def place(self, text, candidates, rotation=0):
        ann = self.ax.annotate(text, xy=(0, 0), xytext=(0, 0), textcoords="offset points",
                               rotation=rotation, **self.style)
        ann.set_in_layout(False)
        best = None
        for i, (xy, direction, extra) in enumerate(candidates):
            self._move(ann, xy, direction)
            box = ann.get_window_extent()
            box = type(box).from_extents(box.x0 - self.pad, box.y0 - self.pad,
                                         box.x1 + self.pad, box.y1 + self.pad)
            score = self._score(box, None if extra is None else self.ax.transData.transform(extra))
            if best is None or score < best[0]:
                best = (score, i, box)
            if score == 0:
                break
        _, i, box = best
        self._move(ann, *candidates[i][:2])
        self.boxes.append(box)
        return i

    def _move(self, ann, xy, offset):
        dx, dy = offset
        ann.xy = tuple(xy)
        ann.set_position((self.GAP * dx, self.GAP * dy))
        # Align the text box so it grows away from the anchor.
        ux, uy = np.asarray(offset) / math.hypot(dx, dy)
        ann.set_horizontalalignment("left" if ux > 0.3 else "right" if ux < -0.3 else "center")
        ann.set_verticalalignment("bottom" if uy > 0.3 else "top" if uy < -0.3 else "center")

    def _score(self, box, extra):
        score = sum(4.0 for b in self.boxes if _overlap(box, b))
        score += sum(w for pts, w in self.lines if _polyline_hits(pts, box))
        if extra is not None:
            score += sum(1.0 for b in self.boxes if _polyline_hits(extra, b))
        frame = self.ax.bbox
        if box.x0 < frame.x0 or box.x1 > frame.x1 or box.y0 < frame.y0 or box.y1 > frame.y1:
            score += 3.0
        return score


def _overlap(a, b):
    return a.x0 < b.x1 and b.x0 < a.x1 and a.y0 < b.y1 and b.y0 < a.y1


def _polyline_hits(points, box):
    return any(_segment_hits(points[k], points[k + 1], box) for k in range(len(points) - 1))


def _segment_hits(p, q, box):
    """Liang-Barsky test: does segment p-q pass through the box?"""
    (x0, y0), (x1, y1) = p, q
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    for pk, qk in ((-dx, x0 - box.x0), (dx, box.x1 - x0), (-dy, y0 - box.y0), (dy, box.y1 - y0)):
        if pk == 0:
            if qk < 0:
                return False
        else:
            t = qk / pk
            if pk < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
            if t0 > t1:
                return False
    return True


if __name__ == "__main__":
    # Rotor inlet and exit of the meanline design (see Meanline/solve_meanline.py).
    rotor_in = VelocityTriangle.from_U_C_alpha(680.7, 931.9, 71.0, station=2)
    rotor_out = VelocityTriangle.from_U_W_beta(816.8, 826.0, -61.0, station=3)
    rotor_in.plot()
    VelocityTriangle.plot_combined(rotor_in, rotor_out)
    plt.show()
