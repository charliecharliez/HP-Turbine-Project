
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, callback, dcc, html, no_update

from Pipeline.Airfoil.airfoil import Airfoil, AirfoilSide, OffsetPoint
from Pipeline.Airfoil.bezier import bezier_points
from Pipeline.Airfoil.camberline import Camberline

NORMALISED_AXIAL_CHORD = 1.0
SUCTION = "suction"
PRESSURE = "pressure"


@dataclass(frozen=True)
class SliderSpec:
    slider_id: str  # Camberline keyword, or "<side>-<AirfoilSide field>"
    label: str
    symbol: str
    subscript: str
    value: float
    minimum: float
    maximum: float
    step: float
    marks: tuple[float, ...]
    unit: str = ""


_METAL_ANGLE_MARKS = (-80, -40, 0, 40, 80)
_SIDE_ANGLE_MARKS = (-60, -30, 0, 30, 60)
_CHORD_RATIO_MARKS = (0.2, 0.5, 0.8)


def _side_sliders(side: str, subscript: str, defaults: AirfoilSide) -> tuple[SliderSpec, ...]:
    interior_point = defaults.interior_points[0]
    return (
        SliderSpec(f"{side}-alpha", "LE angle", "α", subscript, defaults.alpha,
                   -60.0, 60.0, 0.5, _SIDE_ANGLE_MARKS, "°"),
        SliderSpec(f"{side}-alpha_chord_ratio", "α point", "x/c", "", defaults.alpha_chord_ratio,
                   0.01, 0.99, 0.01, _CHORD_RATIO_MARKS),
        SliderSpec(f"{side}-interior_chord_ratio", "Interior point", "x/c", "",
                   interior_point.chord_ratio, 0.01, 0.99, 0.01, _CHORD_RATIO_MARKS),
        SliderSpec(f"{side}-interior_offset", "Interior point", "d", "", interior_point.offset,
                   -0.2, 0.5, 0.0025, (-0.2, 0.0, 0.2, 0.4)),
        SliderSpec(f"{side}-epsilon_chord_ratio", "ε point", "x/c", "",
                   defaults.epsilon_chord_ratio, 0.01, 0.99, 0.01, _CHORD_RATIO_MARKS),
        SliderSpec(f"{side}-epsilon", "TE angle", "ε", subscript, defaults.epsilon,
                   -60.0, 60.0, 0.5, _SIDE_ANGLE_MARKS, "°"),
    )


DEFAULT_SUCTION_SIDE = AirfoilSide(alpha=25.0, epsilon=8.0, alpha_chord_ratio=0.10,
                                   epsilon_chord_ratio=0.80,
                                   interior_points=(OffsetPoint(0.35, 0.35),))
DEFAULT_PRESSURE_SIDE = AirfoilSide(alpha=5.0, epsilon=2.0, alpha_chord_ratio=0.10,
                                    epsilon_chord_ratio=0.80,
                                    interior_points=(OffsetPoint(0.50, -0.08),))

EDGE_SLIDERS = (
    SliderSpec("le_radius", "LE points", "d", "LE", 0.05, 0.0, 0.15, 0.0025,
               (0.0, 0.05, 0.1, 0.15)),
    SliderSpec("te_diam", "TE circle diameter", "D", "TE", 0.03, 0.0, 0.1, 0.001,
               (0.0, 0.05, 0.1)),
)

CAMBERLINE_SLIDERS = (
    SliderSpec("inlet_metal_angle", "Inlet metal angle", "β", "1", 0.0, -85.0, 85.0, 0.5,
               _METAL_ANGLE_MARKS, "°"),
    SliderSpec("exit_metal_angle", "Exit metal angle", "β", "2", 72.0, -85.0, 85.0, 0.5,
               _METAL_ANGLE_MARKS, "°"),
    SliderSpec("stagger", "Stagger angle", "γ", "", 50.0, -85.0, 85.0, 0.5,
               _METAL_ANGLE_MARKS, "°"),
)
SUCTION_SIDE_SLIDERS = _side_sliders(SUCTION, "ss", DEFAULT_SUCTION_SIDE)
PRESSURE_SIDE_SLIDERS = _side_sliders(PRESSURE, "ps", DEFAULT_PRESSURE_SIDE)
_ALL_SLIDERS = CAMBERLINE_SLIDERS + EDGE_SLIDERS + SUCTION_SIDE_SLIDERS + PRESSURE_SIDE_SLIDERS

# Style
_SUCTION_COLOR = "#2a78d6"
_PRESSURE_COLOR = "#eb6834"
_INK = "#2b2a27"
_MUTED_INK = "#6b6a65"
_HAIRLINE = "#c3c2b7"
_GRID = "#ecebe6"
_SURFACE = "#ffffff"
_BODY_FILL = "rgba(43, 42, 39, 0.06)"
_FONT_FAMILY = "Segoe UI, Inter, Helvetica, Arial, sans-serif"
_REFERENCE_LENGTH = 0.45  # axial references and exit tangent extension, in axial chords
_METAL_ARC_RADIUS = 0.18  # in axial chords
_STAGGER_ARC_RADIUS = 0.32  # in axial chords
_ARC_LABEL_GAP = 0.07  # between an arc and its symbol, in axial chords

_PAGE_STYLE = {"display": "flex", "flexWrap": "wrap", "gap": "24px", "padding": "16px",
               "fontFamily": _FONT_FAMILY, "color": _INK, "backgroundColor": _SURFACE}
_CONTROLS_STYLE = {"flex": "0 1 720px", "minWidth": "280px", "maxHeight": "88vh",
                   "overflowY": "auto", "paddingRight": "8px"}
_SIDE_COLUMNS_STYLE = {"display": "grid", "gap": "0 24px",
                       "gridTemplateColumns": "repeat(auto-fit, minmax(320px, 1fr))"}
_GRAPH_STYLE = {"flex": "1 1 600px", "minWidth": "320px"}
_HEADING_STYLE = {"fontSize": "20px", "fontWeight": 600, "margin": "0 0 4px"}
_SECTION_STYLE = {"fontSize": "13px", "fontWeight": 600, "color": _MUTED_INK,
                  "textTransform": "uppercase", "letterSpacing": "0.04em", "margin": "8px 0 12px"}
_NOTE_STYLE = {"fontSize": "13px", "color": _MUTED_INK, "margin": "0 0 16px", "lineHeight": 1.45}
_LABEL_STYLE = {"fontSize": "14px", "fontWeight": 600, "display": "block", "marginBottom": "6px"}
_READOUT_ROW_STYLE = {"display": "flex", "justifyContent": "space-between", "gap": "12px",
                      "fontSize": "14px", "padding": "6px 0", "borderTop": f"1px solid {_GRID}"}
_READOUT_VALUE_STYLE = {"fontVariantNumeric": "tabular-nums", "fontWeight": 600}
_ERROR_STYLE = {"fontSize": "13px", "color": _INK, "lineHeight": 1.45, "margin": "0 0 8px"}


# ===================================================================== layout
def build_app() -> Dash:
    app = Dash(__name__, title="Airfoil construction")
    description = ("Each side is a Bezier curve from the camber LE to a TE apex shared by both "
                   "sides, beyond the camber TE along β₂. Its other control points sit on camber "
                   "normals at a chord ratio x/c, offset by d. α and ε set the first and last legs "
                   "against the camber tangents. A circle of diameter D_TE, tangent to both sides "
                   "and through the camber TE, rounds off the apex. Angles in degrees; lengths in "
                   "axial chords.")
    side_columns = html.Div([
        _slider_section("Suction side", SUCTION_SIDE_SLIDERS),
        _slider_section("Pressure side", PRESSURE_SIDE_SLIDERS),
    ], style=_SIDE_COLUMNS_STYLE)
    controls = html.Div([
        html.H2("Airfoil", style=_HEADING_STYLE),
        html.P(description, style=_NOTE_STYLE),
        _slider_section("Camberline", CAMBERLINE_SLIDERS),
        _slider_section("Leading and trailing edges, shared by both sides", EDGE_SLIDERS),
        side_columns,
        html.Div(id="airfoil-readout"),
    ], style=_CONTROLS_STYLE)
    graph = dcc.Graph(id="airfoil-figure", style={"height": "88vh"},
                      config={"displaylogo": False})
    app.layout = html.Div([controls, html.Div(graph, style=_GRAPH_STYLE)], style=_PAGE_STYLE)
    return app


def _slider_section(title: str, specs: tuple[SliderSpec, ...]) -> html.Div:
    return html.Div([html.H3(title, style=_SECTION_STYLE), *[_slider(spec) for spec in specs]])


def _slider(spec: SliderSpec) -> html.Div:
    marks = {mark: f"{mark:g}{spec.unit}" for mark in spec.marks}
    slider = dcc.Slider(id=spec.slider_id, min=spec.minimum, max=spec.maximum, step=spec.step,
                        value=spec.value, marks=marks, updatemode="drag", allow_direct_input=True)
    label = html.Label([f"{spec.label}  ", spec.symbol, html.Sub(spec.subscript)],
                       style=_LABEL_STYLE)
    return html.Div([label, slider], style={"marginBottom": "20px"})


# ================================================================== callbacks
@callback(
    Output("airfoil-figure", "figure"),
    Output("airfoil-readout", "children"),
    *[Input(spec.slider_id, "value") for spec in _ALL_SLIDERS],
)
def update_airfoil(*slider_values: float | None) -> tuple[go.Figure, list[html.Div | html.P]]:
    if None in slider_values:
        return no_update, no_update

    values = dict(zip((spec.slider_id for spec in _ALL_SLIDERS), slider_values))
    camberline_parameters = {spec.slider_id: values[spec.slider_id] for spec in CAMBERLINE_SLIDERS}
    stagger_range = _stagger_range_row(camberline_parameters["inlet_metal_angle"],
                                       camberline_parameters["exit_metal_angle"])
    try:
        camberline = Camberline(NORMALISED_AXIAL_CHORD, **camberline_parameters)
        airfoil = Airfoil(camberline, values["le_radius"], values["te_diam"],
                          _airfoil_side(values, SUCTION), _airfoil_side(values, PRESSURE))
    except ValueError as error:
        message = "No airfoil for these parameters; see the note beside the sliders."
        return _message_figure(message), [html.P(str(error), style=_ERROR_STYLE), stagger_range]
    return _airfoil_figure(airfoil), _readout(airfoil.camberline) + [stagger_range]


def _airfoil_side(values: dict[str, float], side: str) -> AirfoilSide:
    def value(name: str) -> float:
        return values[f"{side}-{name}"]

    interior_point = OffsetPoint(value("interior_chord_ratio"), value("interior_offset"))
    return AirfoilSide(alpha=value("alpha"), epsilon=value("epsilon"),
                       alpha_chord_ratio=value("alpha_chord_ratio"),
                       epsilon_chord_ratio=value("epsilon_chord_ratio"),
                       interior_points=(interior_point,))


# ==================================================================== readout
def _readout(camberline: Camberline) -> list[html.Div]:
    intersection = camberline.tangent_intersection
    trailing_edge = camberline.trailing_edge
    return [
        _readout_row("Chord", f"{camberline.chord:.3f}"),
        _readout_row("Camber angle", f"{camberline.camber_angle:.1f}°"),
        _readout_row("Camber P₁", f"({intersection[0]:.3f}, {intersection[1]:.3f})"),
        _readout_row("Camber P₂ (TE)", f"({trailing_edge[0]:.3f}, {trailing_edge[1]:.3f})"),
    ]


def _stagger_range_row(inlet_metal_angle: float, exit_metal_angle: float) -> html.Div:
    lower_bound = min(inlet_metal_angle, exit_metal_angle)
    upper_bound = max(inlet_metal_angle, exit_metal_angle)
    return _readout_row("Feasible stagger", f"{lower_bound:.1f}° < γ < {upper_bound:.1f}°")


def _readout_row(label: str, value: str) -> html.Div:
    return html.Div([html.Span(label), html.Span(value, style=_READOUT_VALUE_STYLE)],
                    style=_READOUT_ROW_STYLE)


# ===================================================================== figure
def _airfoil_figure(airfoil: Airfoil) -> go.Figure:
    camberline = airfoil.camberline
    figure = go.Figure()
    _add_airfoil_body(figure, airfoil)
    _add_reference_lines(figure, camberline)
    _add_angle_arcs(figure, camberline)
    _add_chord(figure, camberline)
    _add_camber_control_polygon(figure, camberline)
    _add_camberline(figure, camberline)
    _add_side_control_polygons(figure, airfoil)
    _add_sharp_trailing_edge(figure, airfoil)
    _add_sides(figure, airfoil)
    _apply_layout(figure)
    return figure


def _add_airfoil_body(figure: go.Figure, airfoil: Airfoil) -> None:
    outline = airfoil.coordinates()
    figure.add_trace(go.Scatter(x=outline[:, 0], y=outline[:, 1], mode="lines", fill="toself",
                                fillcolor=_BODY_FILL, line={"width": 0},
                                hoverinfo="skip", showlegend=False))


def _add_reference_lines(figure: go.Figure, camberline: Camberline) -> None:
    length = _REFERENCE_LENGTH * camberline.axial_chord
    axial = _unit_vector(0.0)
    exit_direction = _unit_vector(camberline.exit_metal_angle)
    leading_edge, trailing_edge = camberline.leading_edge, camberline.trailing_edge

    axial_references = _polyline_with_gaps([
        (leading_edge, leading_edge + length * axial),
        (trailing_edge, trailing_edge + length * axial),
    ])
    figure.add_trace(go.Scatter(x=axial_references[0], y=axial_references[1], mode="lines",
                                line={"color": _HAIRLINE, "width": 1, "dash": "dot"},
                                hoverinfo="skip", showlegend=False))

    exit_extension = np.vstack((trailing_edge, trailing_edge + length * exit_direction))
    figure.add_trace(go.Scatter(x=exit_extension[:, 0], y=exit_extension[:, 1], mode="lines",
                                line={"color": _MUTED_INK, "width": 1, "dash": "dash"},
                                hoverinfo="skip", showlegend=False))


def _add_angle_arcs(figure: go.Figure, camberline: Camberline) -> None:
    arcs = (
        (camberline.leading_edge, camberline.inlet_metal_angle, _METAL_ARC_RADIUS, "β₁"),
        (camberline.leading_edge, camberline.stagger, _STAGGER_ARC_RADIUS, "γ"),
        (camberline.trailing_edge, camberline.exit_metal_angle, _METAL_ARC_RADIUS, "β₂"),
    )
    for center, angle, radius, symbol in arcs:
        arc_radius = radius * camberline.axial_chord
        sweep = np.radians(np.linspace(0.0, angle, 48))
        figure.add_trace(go.Scatter(x=center[0] + arc_radius * np.cos(sweep),
                                    y=center[1] + arc_radius * np.sin(sweep), mode="lines",
                                    line={"color": _MUTED_INK, "width": 1.2},
                                    hoverinfo="skip", showlegend=False))

        label_radius = arc_radius + _ARC_LABEL_GAP * camberline.axial_chord
        label_position = center + label_radius * _unit_vector(angle / 2)
        figure.add_annotation(x=label_position[0], y=label_position[1], text=symbol,
                              showarrow=False, font={"color": _INK, "size": 15})


def _add_chord(figure: go.Figure, camberline: Camberline) -> None:
    chord = np.vstack((camberline.leading_edge, camberline.trailing_edge))
    figure.add_trace(go.Scatter(x=chord[:, 0], y=chord[:, 1], mode="lines", name="Chord",
                                line={"color": _HAIRLINE, "width": 1.5},
                                hovertemplate=f"Chord = {camberline.chord:.3f}<extra></extra>"))


def _add_camber_control_polygon(figure: go.Figure, camberline: Camberline) -> None:
    points = camberline.control_points
    label_positions = ["middle left", _intersection_label_side(camberline),
                       "bottom right" if camberline.exit_metal_angle >= 0 else "top right"]
    figure.add_trace(go.Scatter(
        x=points[:, 0], y=points[:, 1], mode="lines+markers+text",
        name="Camber control polygon",
        line={"color": _MUTED_INK, "width": 1, "dash": "dot"},
        marker={"size": 9, "color": _MUTED_INK, "line": {"color": _SURFACE, "width": 2}},
        text=["P₀", "P₁", "P₂"], textposition=label_positions,
        textfont={"color": _INK, "size": 14},
        hovertemplate="%{text} = (%{x:.3f}, %{y:.3f})<extra></extra>",
    ))


def _intersection_label_side(camberline: Camberline) -> str:
    intersection = camberline.tangent_intersection
    chord_height_at_intersection = intersection[0] * math.tan(math.radians(camberline.stagger))
    return "bottom center" if intersection[1] < chord_height_at_intersection else "top center"


def _add_camberline(figure: go.Figure, camberline: Camberline) -> None:
    coordinates = camberline.coordinates()
    figure.add_trace(go.Scatter(
        x=coordinates[:, 0], y=coordinates[:, 1], mode="lines", name="Camberline",
        line={"color": _INK, "width": 1.5, "dash": "dash"},
        hovertemplate="(%{x:.3f}, %{y:.3f})<extra>Camberline</extra>",
    ))


def _add_side_control_polygons(figure: go.Figure, airfoil: Airfoil) -> None:
    polygons = (
        ("SS control polygon", airfoil.suction_control_points, _SUCTION_COLOR, "circle"),
        ("PS control polygon", airfoil.pressure_control_points, _PRESSURE_COLOR, "square"),
    )
    for name, points, color, symbol in polygons:
        interior_count = len(points) - 5
        point_names = ["camber LE", "LE point", "α point",
                       *[f"interior point {index + 1}" for index in range(interior_count)],
                       "ε point", "TE apex"]
        marker_sizes = [0] + [9] * (len(points) - 1)
        figure.add_trace(go.Scatter(
            x=points[:, 0], y=points[:, 1], mode="lines+markers", name=name,
            line={"color": color, "width": 1, "dash": "dash"},
            marker={"size": marker_sizes, "symbol": symbol, "color": color,
                    "line": {"color": _SURFACE, "width": 1.5}},
            customdata=point_names,
            hovertemplate=f"%{{customdata}} = (%{{x:.3f}}, %{{y:.3f}})<extra>{name}</extra>",
        ))


def _add_sharp_trailing_edge(figure: go.Figure, airfoil: Airfoil) -> None:
    circle = airfoil.te_circle
    trimmed_ends = ((airfoil.suction_control_points, circle.suction_parameter),
                    (airfoil.pressure_control_points, circle.pressure_parameter))
    x_values: list[float | None] = []
    y_values: list[float | None] = []
    for control_points, tangent_parameter in trimmed_ends:
        segment = bezier_points(control_points, np.linspace(tangent_parameter, 1.0, 24))
        x_values += [*segment[:, 0], None]
        y_values += [*segment[:, 1], None]
    figure.add_trace(go.Scatter(x=x_values, y=y_values, mode="lines",
                                name="Sharp TE before rounding",
                                line={"color": _MUTED_INK, "width": 1, "dash": "dot"},
                                hoverinfo="skip"))


def _add_sides(figure: go.Figure, airfoil: Airfoil) -> None:
    sides = (
        ("Suction side", airfoil.suction_side(), _SUCTION_COLOR),
        ("Pressure side", airfoil.pressure_side(), _PRESSURE_COLOR),
        ("Trailing-edge arc", airfoil.trailing_edge_arc(), _INK),
    )
    for name, coordinates, color in sides:
        figure.add_trace(go.Scatter(
            x=coordinates[:, 0], y=coordinates[:, 1], mode="lines", name=name,
            line={"color": color, "width": 2.5},
            hovertemplate=f"(%{{x:.3f}}, %{{y:.3f}})<extra>{name}</extra>",
        ))


def _apply_layout(figure: go.Figure) -> None:
    axis_style = {"gridcolor": _GRID, "zeroline": False, "showline": True, "linecolor": _HAIRLINE,
                  "ticks": "outside", "tickcolor": _HAIRLINE, "tickfont": {"color": _MUTED_INK}}
    figure.update_xaxes(title="x / C<sub>ax</sub>  (axial)", **axis_style)
    figure.update_yaxes(title="y / C<sub>ax</sub>  (tangential)", scaleanchor="x", scaleratio=1,
                        **axis_style)
    figure.update_layout(
        plot_bgcolor=_SURFACE, paper_bgcolor=_SURFACE, hovermode="closest",
        font={"family": _FONT_FAMILY, "color": _INK, "size": 13},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        margin={"l": 64, "r": 16, "t": 64, "b": 56},
    )


def _message_figure(message: str) -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper",
                          showarrow=False, font={"color": _MUTED_INK, "size": 15})
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    figure.update_layout(plot_bgcolor=_SURFACE, paper_bgcolor=_SURFACE,
                         font={"family": _FONT_FAMILY})
    return figure


def _polyline_with_gaps(segments: list[tuple[np.ndarray, np.ndarray]]) -> tuple[list, list]:
    """Join separate segments into one x, y pair, with None breaking the line between them."""
    x_values: list[float | None] = []
    y_values: list[float | None] = []
    for start, end in segments:
        x_values += [start[0], end[0], None]
        y_values += [start[1], end[1], None]
    return x_values, y_values


def _unit_vector(angle: float) -> np.ndarray:  # angle in deg from axial
    radians = math.radians(angle)
    return np.array([math.cos(radians), math.sin(radians)])


if __name__ == "__main__":
    build_app().run(debug=True)
