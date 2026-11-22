"""THIS CODE IS NOT RELEVANT TO THE DESIGN METHODOLOGY FOR THIS TURBINE PROJECT
ITS SOLE PURPOSE IS PLOTTING
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
from matplotlib.text import Annotation
from matplotlib.transforms import Bbox
from numpy.typing import ArrayLike

__all__ = ["VelocityTriangle"]

# Style
_C_COLOR = "#2a78d6"  # absolute velocity
_W_COLOR = "#eb6834"  # relative velocity
_U_COLOR = "#1baf7a"  # blade speed
_MUTED = "#898781"  # meridional dimension
_HAIRLINE = "#c3c2b7"  # reference and extension lines
_LINESTYLES = ("-", (0, (5, 2.5)))  # first triangle solid, second dashed
_FONTSIZE = 9  # pt
_DIMENSION_SPACING = 0.075  # gap between dimension lines, fraction of the diagram size
_ARC_RADII = (0.3, 0.42, 0.54, 0.66, 0.78, 0.9, 1.02)  # fractions of the smallest C_m
_MIN_ARC_ANGLE = 0.5  # deg

# Label placement penalties
_HIT_LABEL = 4.0
_HIT_OUTSIDE_AXES = 3.0
_HIT_VECTOR = 2.0
_HIT_DIMENSION = 1.5
_HIT_ARC = 1.0
_HIT_HAIRLINE = 0.5

LineStyle = str | tuple[float, tuple[float, float]]
# (anchor in data units, text offset in units of _Placer.GAP, arc in data units or None)
Candidate = tuple[np.ndarray, np.ndarray, np.ndarray | None]
Obstacle = tuple[np.ndarray, float]  # (polyline, penalty)
# (triangle, angle in deg, symbol, color, line style)
AngleLabel = tuple["VelocityTriangle", float, str, str, LineStyle]


@dataclass(frozen=True)
class VelocityTriangle:
    U: float  # blade speed, + in the direction of rotation
    C_m: float  # meridional velocity, > 0
    C_u: float  # absolute tangential velocity, + in the direction of U
    _: KW_ONLY
    station: int | None = None  # label subscript, e.g. C_m2
    unit: str = "m/s"  # speed unit shown in labels

    def __post_init__(self) -> None:
        for name in ("U", "C_m", "C_u"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")
            object.__setattr__(self, name, value)
        if self.C_m <= 0:
            raise ValueError(f"C_m must be positive, got {self.C_m}")

    # ------------------------------------------------------------- constructors
    # alpha, beta: deg, from meridional, + in the direction of U
    @classmethod
    def from_U_Cm_beta(cls, U: float, C_m: float, beta: float, *,
                       station: int | None = None, unit: str = "m/s") -> VelocityTriangle:
        C_u = U + C_m * _tan(beta, "beta")
        return cls(U, C_m, C_u, station=station, unit=unit)

    @classmethod
    def from_U_Cm_alpha(cls, U: float, C_m: float, alpha: float, *,
                        station: int | None = None, unit: str = "m/s") -> VelocityTriangle:
        C_u = C_m * _tan(alpha, "alpha")
        return cls(U, C_m, C_u, station=station, unit=unit)

    @classmethod
    def from_U_C_alpha(cls, U: float, C: float, alpha: float, *,
                       station: int | None = None, unit: str = "m/s") -> VelocityTriangle:
        a = math.radians(_check_angle(alpha, "alpha"))
        return cls(U, C * math.cos(a), C * math.sin(a), station=station, unit=unit)

    @classmethod
    def from_U_W_beta(cls, U: float, W: float, beta: float, *,
                      station: int | None = None, unit: str = "m/s") -> VelocityTriangle:
        b = math.radians(_check_angle(beta, "beta"))
        return cls(U, W * math.cos(b), U + W * math.sin(b), station=station, unit=unit)

    @classmethod
    def from_U_C_W(cls, U: float, C: float, W: float, *,
                   station: int | None = None, unit: str = "m/s") -> VelocityTriangle:
        if U == 0 or C <= 0 or W <= 0:
            raise ValueError(f"Need U != 0 and C, W > 0, got U={U}, C={C}, W={W}")
        C_u = (C**2 - W**2 + U**2) / (2 * U)
        C_m_squared = C**2 - C_u**2
        if C_m_squared <= 0:
            raise ValueError(f"U={U}, C={C}, W={W} do not close a triangle")
        return cls(U, math.sqrt(C_m_squared), C_u, station=station, unit=unit)

    @classmethod
    def from_U_alpha_beta(cls, U: float, alpha: float, beta: float, *,
                          station: int | None = None, unit: str = "m/s") -> VelocityTriangle:
        tan_a = _tan(alpha, "alpha")
        diff = tan_a - _tan(beta, "beta")
        if diff == 0 or U / diff <= 0:
            raise ValueError(f"No triangle with positive C_m for U={U}, alpha={alpha}, "
                             f"beta={beta}: for U > 0, alpha must be larger than beta")
        C_m = U / diff
        return cls(U, C_m, C_m * tan_a, station=station, unit=unit)

    # ------------------------------------------------------- derived quantities
    @property
    def W_u(self) -> float:
        return self.C_u - self.U

    @property
    def C(self) -> float:
        return math.hypot(self.C_m, self.C_u)

    @property
    def W(self) -> float:
        return math.hypot(self.C_m, self.W_u)

    @property
    def alpha(self) -> float:  # deg
        return math.degrees(math.atan2(self.C_u, self.C_m))

    @property
    def beta(self) -> float:  # deg
        return math.degrees(math.atan2(self.W_u, self.C_m))

    # ----------------------------------------------------------------- plotting
    def plot(self, ax: Axes | None = None, *, title: str | None = None,
             precision: int = 1, legend: bool = True) -> Axes:
        if title is None:
            suffix = "" if self.station is None else f", station {self.station}"
            title = "Velocity triangle" + suffix
        return _draw([self], ax, title, precision, legend)

    @staticmethod
    def plot_combined(first: VelocityTriangle, second: VelocityTriangle,
                      ax: Axes | None = None, *, title: str | None = None,
                      precision: int = 1, legend: bool = True) -> Axes:
        if first.unit != second.unit:
            raise ValueError(f"Cannot share one diagram between units "
                             f"{first.unit!r} and {second.unit!r}")
        if title is None:
            if first.station is None or second.station is None:
                title = "Combined velocity triangles"
            else:
                title = f"Velocity triangles, stations {first.station} and {second.station}"
        return _draw([first, second], ax, title, precision, legend)

    def _sym(self, base: str, sub: str = "") -> str:
        sub = f"{sub}{'' if self.station is None else self.station}"
        return f"${base}_{{{sub}}}$" if sub else f"${base}$"


def _check_angle(angle: float, name: str) -> float:
    if not -90 < angle < 90:
        raise ValueError(f"{name} must lie strictly between -90 and 90 degrees, got {angle}")
    return angle


def _tan(angle: float, name: str) -> float:
    return math.tan(math.radians(_check_angle(angle, name)))


def _format_value(value: float, precision: int) -> str:
    text = f"{value:.{precision}f}"
    if float(text) == 0:
        text = text.lstrip("-")
    return text.replace("-", "−")


# ==================================================================== drawing
def _draw(triangles: list[VelocityTriangle], ax: Axes | None, title: str,
          precision: int, legend: bool) -> Axes:
    ink = mpl.rcParams["text.color"]
    unit = triangles[0].unit

    # Layout (data units)
    xs = [0.0] + [x for t in triangles for x in (t.C_u, t.W_u)]
    x_lo, x_hi = min(xs), max(xs)
    depth = max(t.C_m for t in triangles)
    step = _DIMENSION_SPACING * max(x_hi - x_lo, depth)
    first_row = 1.5  # steps between the lowest base and the first C_u dimension line

    sides = [1 if t.C_u + t.W_u >= 0 else -1 for t in triangles]  # side each triangle leans to
    col_x: list[float] = []  # x of each C_m dimension line
    count = {1: 0, -1: 0}
    for side in sides:
        count[side] += 1
        edge = x_hi if side > 0 else x_lo
        col_x.append(edge + side * count[side] * step)
    row_y = [-depth - (first_row + k) * step for k in range(len(triangles))]  # y of each C_u line

    x_min = min([x_lo, *col_x]) - step
    x_max = max([x_hi, *col_x]) + step
    y_min = row_y[-1] - 0.8 * step
    y_max = 0.6 * step

    if ax is None:
        ax = _new_axes(len(triangles), legend, x_max - x_min, y_max - y_min)
    fig = ax.figure

    obstacles: list[Obstacle] = []
    vector_labels: list[tuple[str, list[Candidate]]] = []
    angle_labels: list[AngleLabel] = []
    dim_labels: list[tuple[str, list[Candidate], float]] = []  # (text, candidates, rotation)
    legend_entries: list[tuple[Line2D, str]] = []

    def hairline(points: ArrayLike) -> None:
        points = np.asarray(points, float)
        ax.plot(points[:, 0], points[:, 1], color=_HAIRLINE, lw=0.8, zorder=1,
                solid_capstyle="butt")
        obstacles.append((points, _HIT_HAIRLINE))

    hairline([(0, 0), (0, row_y[-1] - 0.4 * step)])

    for i, t in enumerate(triangles):
        ls = _LINESTYLES[i]
        apex = np.zeros(2)
        c_tip = np.array([t.C_u, -t.C_m])
        w_tip = np.array([t.W_u, -t.C_m])
        centroid = (apex + c_tip + w_tip) / 3

        # Triangle
        vectors = ((apex, c_tip, _C_COLOR, "C", t.C),
                   (apex, w_tip, _W_COLOR, "W", t.W),
                   (w_tip, c_tip, _U_COLOR, "U", t.U))
        for start, end, color, symbol, magnitude in vectors:
            _arrow(ax, start, end, color, ls, lw=2.0, head=14, zorder=4)
            obstacles.append((np.array([start, end]), _HIT_VECTOR))
            text = f"{t._sym(symbol)} = {_format_value(magnitude, precision)} {unit}"
            vector_labels.append((text, _along(start, end, centroid)))
        angle_labels.append((t, t.alpha, r"\alpha", _C_COLOR, ls))
        angle_labels.append((t, t.beta, r"\beta", _W_COLOR, ls))

        # C_u dimension
        y = row_y[i]
        hairline([(t.C_u, -t.C_m), (t.C_u, y - 0.3 * step)])
        if t.C_u != 0:
            _arrow(ax, (0, y), (t.C_u, y), _C_COLOR, ls, lw=1.0, head=8, zorder=3)
        obstacles.append((np.array([(0, y), (t.C_u, y)]), _HIT_DIMENSION))
        sign = 1.0 if t.C_u >= 0 else -1.0
        middle = np.array([t.C_u / 2, y])
        end = np.array([t.C_u, y])
        candidates = [(middle, np.array([0.0, 1.0]), None),
                      (middle, np.array([0.0, -1.0]), None),
                      (end, np.array([sign, 0.0]), None)]
        text = f"{t._sym('C', 'u')} = {_format_value(t.C_u, precision)} {unit}"
        dim_labels.append((text, candidates, 0))

        # C_m dimension
        x_c, side = col_x[i], sides[i]
        x_edge = max(t.C_u, t.W_u, 0) if side > 0 else min(t.C_u, t.W_u, 0)
        x_ext = x_c + side * 0.3 * step
        hairline([(0, 0), (x_ext, 0)])
        hairline([(x_edge, -t.C_m), (x_ext, -t.C_m)])
        _arrow(ax, (x_c, 0), (x_c, -t.C_m), _MUTED, ls, lw=1.0, head=8, zorder=3, style="<|-|>")
        obstacles.append((np.array([(x_c, 0), (x_c, -t.C_m)]), _HIT_DIMENSION))
        middle = np.array([x_c, -t.C_m / 2])
        candidates = [(middle, np.array([side, 0.0]), None),
                      (middle, np.array([-side, 0.0]), None)]
        text = f"{t._sym('C', 'm')} = {_format_value(t.C_m, precision)} {unit}"
        dim_labels.append((text, candidates, 90))

        legend_entries += _legend_entries(t, ls, precision)

    # Axes
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    ax.set_title(title, loc="left", fontsize=11, color=ink)
    if legend:
        _add_legend(ax, legend_entries)

    # Labels
    fig.draw_without_rendering()
    label_box = dict(boxstyle="round,pad=0.15", fc=fig.get_facecolor(), ec="none", alpha=0.85)
    style = dict(fontsize=_FONTSIZE, color=ink, zorder=6, annotation_clip=False, bbox=label_box)
    placer = _Placer(ax, obstacles, style)

    _place_angles(ax, placer, angle_labels, triangles, precision)
    for text, candidates in vector_labels:
        placer.place(text, candidates)
    for text, candidates, rotation in dim_labels:
        placer.place(text, candidates, rotation=rotation)
    return ax


def _new_axes(n_triangles: int, legend: bool, data_width: float, data_height: float) -> Axes:
    axes_width = 7.4 if n_triangles == 1 else 8.6  # in
    legend_width = 3.6  # in
    height = axes_width * data_height / data_width + 0.8  # in
    if legend:
        legend_rows = 7 * n_triangles
        height = max(height, 0.21 * legend_rows + 1.0)
    height = min(max(height, 3.5), 9.0)
    _, ax = plt.subplots(figsize=(axes_width + legend_width, height), layout="constrained")
    return ax


def _arrow(ax: Axes, start: ArrayLike, end: ArrayLike, color: str, ls: LineStyle, *,
           lw: float, head: float, zorder: float, style: str = "-|>") -> None:
    arrow = FancyArrowPatch(tuple(start), tuple(end), arrowstyle=style, mutation_scale=head,
                            color=color, lw=lw, linestyle=ls, shrinkA=0, shrinkB=0,
                            joinstyle="miter", capstyle="butt", zorder=zorder)
    ax.add_patch(arrow)


def _along(start: np.ndarray, end: np.ndarray, centroid: np.ndarray) -> list[Candidate]:
    d = end - start
    length = math.hypot(*d)
    outward = np.array([-d[1], d[0]]) / length if length else np.array([0.0, -1.0])
    if np.dot(outward, (start + end) / 2 - centroid) < 0:
        outward = -outward

    fractions = (0.5, 0.62, 0.38, 0.74, 0.26, 0.86, 0.14)
    compass = [(-1, -1), (1, -1), (-1, 0), (1, 0), (0, -1), (-1, 1), (1, 1)]
    beside = [(start + f * d, s * outward, None) for s in (1, -1) for f in fractions]
    around_tip = [(end, np.array(c, float) / math.hypot(*c), None) for c in compass]
    farther = [(start + f * d, 3 * s * outward, None) for s in (1, -1) for f in fractions]
    return beside + around_tip + farther


def _place_angles(ax: Axes, placer: _Placer, angle_labels: list[AngleLabel],
                  triangles: list[VelocityTriangle], precision: int) -> None:
    r_ref = min(t.C_m for t in triangles)
    radii = [f * r_ref for f in _ARC_RADII]
    used: dict[int, set[float]] = {1: set(), -1: set()}  # arc radii taken on each side
    meridional = -math.pi / 2  # rad

    for t, angle, symbol, color, ls in angle_labels:
        side = 1 if angle >= 0 else -1
        free = [r for r in radii if r not in used[side]] or radii
        phi = meridional + math.radians(angle)  # rad, direction of the vector

        on_arc: list[tuple[Candidate, float]] = []  # (candidate, radius)
        across_meridional: list[tuple[Candidate, float]] = []
        for r in free:
            arc = _arc_points(r, meridional, phi)
            for f in (0.5, 0.7, 0.3, 0.85, 0.15):
                direction = _unit_vector(meridional + f * (phi - meridional))
                on_arc.append(((r * direction, direction, arc), r))
            beyond_vector = side * _ccw_tangent(phi)
            on_arc.append(((r * _unit_vector(phi), beyond_vector, arc), r))
            beyond_meridional = -side * _ccw_tangent(meridional)
            across_meridional.append(((r * _unit_vector(meridional), beyond_meridional, arc), r))
        options = on_arc + across_meridional

        text = f"{t._sym(symbol)} = {_format_value(angle, precision)}°"
        chosen = placer.place(text, [candidate for candidate, _ in options])
        r = options[chosen][1]
        used[side].add(r)

        if abs(angle) >= _MIN_ARC_ANGLE:
            theta1, theta2 = sorted((-90.0, -90.0 + angle))
            arc_patch = Arc((0, 0), 2 * r, 2 * r, theta1=theta1, theta2=theta2,
                            color=color, lw=1.2, linestyle=ls, zorder=3)
            ax.add_patch(arc_patch)
            placer.add_line(_arc_points(r, meridional, phi), _HIT_ARC)


def _unit_vector(theta: float) -> np.ndarray:
    return np.array([math.cos(theta), math.sin(theta)])


def _ccw_tangent(theta: float) -> np.ndarray:
    return np.array([-math.sin(theta), math.cos(theta)])


def _arc_points(radius: float, theta_start: float, theta_end: float) -> np.ndarray:
    theta = np.linspace(theta_start, theta_end, 24)
    return radius * np.column_stack((np.cos(theta), np.sin(theta)))


# ===================================================================== legend
def _legend_entries(t: VelocityTriangle, ls: LineStyle,
                    precision: int) -> list[tuple[Line2D, str]]:
    def speed(value: float) -> str:
        return f"{_format_value(value, precision)} {t.unit}"

    def degrees(value: float) -> str:
        return f"{_format_value(value, precision)}°"

    vector = dict(lw=2, ls=ls)
    component = dict(lw=1, ls=ls)
    angle_marker = dict(ls="none", marker=r"$\angle$", markersize=11)
    alpha, beta = t._sym(r"\alpha"), t._sym(r"\beta")
    rows = [
        (_C_COLOR, vector, f"Absolute velocity  {t._sym('C')} = {speed(t.C)}"),
        (_W_COLOR, vector, f"Relative velocity  {t._sym('W')} = {speed(t.W)}"),
        (_U_COLOR, vector, f"Blade speed  {t._sym('U')} = {speed(t.U)}"),
        (_C_COLOR, angle_marker, f"Absolute flow angle  {alpha} = {degrees(t.alpha)}"),
        (_W_COLOR, angle_marker, f"Relative flow angle  {beta} = {degrees(t.beta)}"),
        (_MUTED, component, f"Meridional velocity  {t._sym('C', 'm')} = {speed(t.C_m)}"),
        (_C_COLOR, component, f"Abs. tangential velocity  {t._sym('C', 'u')} = {speed(t.C_u)}"),
    ]
    return [(Line2D([], [], color=color, **kw), label) for color, kw, label in rows]


def _add_legend(ax: Axes, entries: list[tuple[Line2D, str]]) -> None:
    handles = [handle for handle, _ in entries]
    labels = [label for _, label in entries]
    title = "Angles from meridional,\n+ in the direction of U"
    ax.legend(handles=handles, labels=labels, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              frameon=False, fontsize=8.5, handlelength=2.6, alignment="left",
              title=title, title_fontsize=8.5)


# ============================================================ label placement
class _Placer:
    GAP = 5  # pt between anchor and text, per unit of offset

    def __init__(self, ax: Axes, obstacles: list[Obstacle], style: dict) -> None:
        self.ax = ax
        self.style = style
        self.lines = [(ax.transData.transform(p), w) for p, w in obstacles]  # display units
        self.boxes: list[Bbox] = []  # placed labels and legend, display units
        self.pad = 2 * ax.figure.dpi / 72  # px
        legend = ax.get_legend()
        if legend is not None:
            self.boxes.append(legend.get_window_extent())

    def add_line(self, points: np.ndarray, penalty: float) -> None:
        self.lines.append((self.ax.transData.transform(points), penalty))

    def place(self, text: str, candidates: list[Candidate], rotation: float = 0) -> int:
        ann = self.ax.annotate(text, xy=(0, 0), xytext=(0, 0), textcoords="offset points",
                               rotation=rotation, **self.style)
        ann.set_in_layout(False)
        best_score, best_index = math.inf, 0
        for i, (anchor, offset, arc) in enumerate(candidates):
            self._move(ann, anchor, offset)
            arc_display = None if arc is None else self.ax.transData.transform(arc)
            score = self._score(self._padded_extent(ann), arc_display)
            if score < best_score:
                best_score, best_index = score, i
            if score == 0:
                break
        anchor, offset, _ = candidates[best_index]
        self._move(ann, anchor, offset)
        self.boxes.append(self._padded_extent(ann))
        return best_index

    def _padded_extent(self, ann: Annotation) -> Bbox:
        box = ann.get_window_extent()
        p = self.pad
        return Bbox.from_extents(box.x0 - p, box.y0 - p, box.x1 + p, box.y1 + p)

    def _move(self, ann: Annotation, anchor: np.ndarray, offset: np.ndarray) -> None:
        dx, dy = offset
        ux, uy = np.asarray(offset) / math.hypot(dx, dy)
        ann.xy = tuple(anchor)
        ann.set_position((self.GAP * dx, self.GAP * dy))
        ann.set_horizontalalignment("left" if ux > 0.3 else "right" if ux < -0.3 else "center")
        ann.set_verticalalignment("bottom" if uy > 0.3 else "top" if uy < -0.3 else "center")

    def _score(self, box: Bbox, arc: np.ndarray | None) -> float:
        score = sum(_HIT_LABEL for b in self.boxes if _overlap(box, b))
        score += sum(w for pts, w in self.lines if _polyline_hits(pts, box))
        if arc is not None:
            score += sum(_HIT_ARC for b in self.boxes if _polyline_hits(arc, b))
        frame = self.ax.bbox
        inside = (frame.x0 <= box.x0 and box.x1 <= frame.x1
                  and frame.y0 <= box.y0 and box.y1 <= frame.y1)
        if not inside:
            score += _HIT_OUTSIDE_AXES
        return score


def _overlap(a: Bbox, b: Bbox) -> bool:
    return a.x0 < b.x1 and b.x0 < a.x1 and a.y0 < b.y1 and b.y0 < a.y1


def _polyline_hits(points: np.ndarray, box: Bbox) -> bool:
    return any(_segment_hits(points[k], points[k + 1], box) for k in range(len(points) - 1))


def _segment_hits(p: np.ndarray, q: np.ndarray, box: Bbox) -> bool:
    """Liang-Barsky segment clipping against an axis-aligned box."""
    (x0, y0), (x1, y1) = p, q
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    edges = ((-dx, x0 - box.x0), (dx, box.x1 - x0), (-dy, y0 - box.y0), (dy, box.y1 - y0))
    for pk, qk in edges:
        if pk == 0:
            if qk < 0:
                return False
            continue
        t = qk / pk
        if pk < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return False
    return True
