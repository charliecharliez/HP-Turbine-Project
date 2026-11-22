
from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, callback, dcc, html, no_update

from Airfoil.bezier_construction import Camberline

NORMALISED_AXIAL_CHORD = 1.0
ANGLE_LIMIT = 85.0  # deg, sliders span ±ANGLE_LIMIT
DEFAULT_INLET_METAL_ANGLE = 0.0  # deg
DEFAULT_EXIT_METAL_ANGLE = 72.0  # deg
DEFAULT_STAGGER = 50.0  # deg

# Style
_CAMBER_COLOR = "#2a78d6"
_CONTROL_COLOR = "#eb6834"
_INK = "#2b2a27"
_MUTED_INK = "#6b6a65"
_HAIRLINE = "#c3c2b7"
_GRID = "#ecebe6"
_SURFACE = "#ffffff"
_FONT_FAMILY = "Segoe UI, Inter, Helvetica, Arial, sans-serif"
_REFERENCE_LENGTH = 0.45  # axial references and exit tangent extension, in axial chords
_METAL_ARC_RADIUS = 0.18  # in axial chords
_STAGGER_ARC_RADIUS = 0.32  # in axial chords
_ARC_LABEL_GAP = 0.07  # between an arc and its symbol, in axial chords

_PAGE_STYLE = {"display": "flex", "flexWrap": "wrap", "gap": "24px", "padding": "16px",
               "fontFamily": _FONT_FAMILY, "color": _INK, "backgroundColor": _SURFACE}
_CONTROLS_STYLE = {"flex": "0 1 340px", "minWidth": "280px"}
_GRAPH_STYLE = {"flex": "1 1 600px", "minWidth": "320px"}
_HEADING_STYLE = {"fontSize": "20px", "fontWeight": 600, "margin": "0 0 4px"}
_NOTE_STYLE = {"fontSize": "13px", "color": _MUTED_INK, "margin": "0 0 24px", "lineHeight": 1.45}
_LABEL_STYLE = {"fontSize": "14px", "fontWeight": 600, "display": "block", "marginBottom": "6px"}
_READOUT_ROW_STYLE = {"display": "flex", "justifyContent": "space-between", "gap": "12px",
                      "fontSize": "14px", "padding": "6px 0", "borderTop": f"1px solid {_GRID}"}
_READOUT_VALUE_STYLE = {"fontVariantNumeric": "tabular-nums", "fontWeight": 600}
_ERROR_STYLE = {"fontSize": "13px", "color": _INK, "lineHeight": 1.45, "margin": "0 0 8px"}


# ===================================================================== layout
def build_app() -> Dash:
    app = Dash(__name__, title="Camberline construction")
    description = ("Quadratic Bezier from leading edge P₀ to trailing edge P₂. The middle control "
                   "point P₁ is where the inlet and exit metal-angle tangents meet. "
                   "Angles in degrees from axial; lengths in axial chords.")
    controls = html.Div([
        html.H2("Camberline", style=_HEADING_STYLE),
        html.P(description, style=_NOTE_STYLE),
        _angle_slider("inlet-metal-angle", "Inlet metal angle β₁", DEFAULT_INLET_METAL_ANGLE),
        _angle_slider("exit-metal-angle", "Exit metal angle β₂", DEFAULT_EXIT_METAL_ANGLE),
        _angle_slider("stagger", "Stagger angle γ", DEFAULT_STAGGER),
        html.Div(id="camberline-readout"),
    ], style=_CONTROLS_STYLE)
    graph = dcc.Graph(id="camberline-figure", style={"height": "88vh"},
                      config={"displaylogo": False})
    app.layout = html.Div([controls, html.Div(graph, style=_GRAPH_STYLE)], style=_PAGE_STYLE)
    return app


def _angle_slider(slider_id: str, label: str, value: float) -> html.Div:
    marks = {angle: f"{angle}°" for angle in range(-80, 81, 40)}
    slider = dcc.Slider(id=slider_id, min=-ANGLE_LIMIT, max=ANGLE_LIMIT, step=0.5, value=value,
                        marks=marks, updatemode="drag", allow_direct_input=True)
    return html.Div([html.Label(label, style=_LABEL_STYLE), slider],
                    style={"marginBottom": "28px"})


# ================================================================== callbacks
@callback(
    Output("camberline-figure", "figure"),
    Output("camberline-readout", "children"),
    Input("inlet-metal-angle", "value"),
    Input("exit-metal-angle", "value"),
    Input("stagger", "value"),
)
def update_camberline(inlet_metal_angle: float | None, exit_metal_angle: float | None,
                      stagger: float | None) -> tuple[go.Figure, list[html.Div | html.P]]:
    if None in (inlet_metal_angle, exit_metal_angle, stagger):
        return no_update, no_update

    stagger_range = _stagger_range_row(inlet_metal_angle, exit_metal_angle)
    try:
        camberline = Camberline(NORMALISED_AXIAL_CHORD, stagger, inlet_metal_angle,
                                exit_metal_angle)
    except ValueError as error:
        message = "No camberline for these angles; see the note beside the sliders."
        return _message_figure(message), [html.P(str(error), style=_ERROR_STYLE), stagger_range]
    return _camberline_figure(camberline), _readout(camberline) + [stagger_range]


# ==================================================================== readout
def _readout(camberline: Camberline) -> list[html.Div]:
    intersection = camberline.tangent_intersection
    trailing_edge = camberline.trailing_edge
    return [
        _readout_row("Chord", f"{camberline.chord:.3f}"),
        _readout_row("Camber angle", f"{camberline.camber_angle:.1f}°"),
        _readout_row("P₁", f"({intersection[0]:.3f}, {intersection[1]:.3f})"),
        _readout_row("P₂ (trailing edge)", f"({trailing_edge[0]:.3f}, {trailing_edge[1]:.3f})"),
    ]


def _stagger_range_row(inlet_metal_angle: float, exit_metal_angle: float) -> html.Div:
    lower_bound = min(inlet_metal_angle, exit_metal_angle)
    upper_bound = max(inlet_metal_angle, exit_metal_angle)
    return _readout_row("Feasible stagger", f"{lower_bound:.1f}° < γ < {upper_bound:.1f}°")


def _readout_row(label: str, value: str) -> html.Div:
    return html.Div([html.Span(label), html.Span(value, style=_READOUT_VALUE_STYLE)],
                    style=_READOUT_ROW_STYLE)


# ===================================================================== figure
def _camberline_figure(camberline: Camberline) -> go.Figure:
    figure = go.Figure()
    _add_reference_lines(figure, camberline)
    _add_angle_arcs(figure, camberline)
    _add_chord(figure, camberline)
    _add_control_polygon(figure, camberline)
    _add_camberline(figure, camberline)
    _apply_layout(figure)
    return figure


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
                                line={"color": _CONTROL_COLOR, "width": 1.5, "dash": "dash"},
                                hoverinfo="skip", showlegend=False))


def _add_angle_arcs(figure: go.Figure, camberline: Camberline) -> None:
    arcs = (
        (camberline.leading_edge, camberline.inlet_metal_angle, _METAL_ARC_RADIUS, "β₁",
         _CONTROL_COLOR),
        (camberline.leading_edge, camberline.stagger, _STAGGER_ARC_RADIUS, "γ", _INK),
        (camberline.trailing_edge, camberline.exit_metal_angle, _METAL_ARC_RADIUS, "β₂",
         _CONTROL_COLOR),
    )
    for center, angle, radius, symbol, color in arcs:
        arc_radius = radius * camberline.axial_chord
        sweep = np.radians(np.linspace(0.0, angle, 48))
        figure.add_trace(go.Scatter(x=center[0] + arc_radius * np.cos(sweep),
                                    y=center[1] + arc_radius * np.sin(sweep), mode="lines",
                                    line={"color": color, "width": 1.2},
                                    hoverinfo="skip", showlegend=False))

        label_radius = arc_radius + _ARC_LABEL_GAP * camberline.axial_chord
        label_position = center + label_radius * _unit_vector(angle / 2)
        figure.add_annotation(x=label_position[0], y=label_position[1], text=symbol,
                              showarrow=False, font={"color": _INK, "size": 15})


def _add_chord(figure: go.Figure, camberline: Camberline) -> None:
    chord = np.vstack((camberline.leading_edge, camberline.trailing_edge))
    figure.add_trace(go.Scatter(x=chord[:, 0], y=chord[:, 1], mode="lines", name="Chord",
                                line={"color": _INK, "width": 1.5},
                                hovertemplate=f"Chord = {camberline.chord:.3f}<extra></extra>"))


def _add_control_polygon(figure: go.Figure, camberline: Camberline) -> None:
    points = camberline.control_points
    label_positions = ["middle left", _intersection_label_side(camberline),
                       "bottom right" if camberline.exit_metal_angle >= 0 else "top right"]
    figure.add_trace(go.Scatter(
        x=points[:, 0], y=points[:, 1], mode="lines+markers+text", name="Control polygon",
        line={"color": _CONTROL_COLOR, "width": 1.5, "dash": "dash"},
        marker={"size": 10, "color": _CONTROL_COLOR, "line": {"color": _SURFACE, "width": 2}},
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
        line={"color": _CAMBER_COLOR, "width": 2.5},
        hovertemplate="(%{x:.3f}, %{y:.3f})<extra>Camberline</extra>",
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
        margin={"l": 64, "r": 16, "t": 48, "b": 56},
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
