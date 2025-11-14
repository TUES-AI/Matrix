#!/usr/bin/env python3
"""
Interactive 2D linear transformation visualizer with four sliders that
control how the x and y axes are transformed. The first two sliders set
the image of the x-axis basis vector (1, 0) and the second two sliders
set the image of the y-axis basis vector (0, 1).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from matplotlib.widgets import Slider


AXIS_SPAN = 5
SLIDER_MIN = -3.0
SLIDER_MAX = 3.0
SLIDER_STEP = 0.05
CLICK_RADIUS = 0.3
GRID_VALUES = np.arange(-AXIS_SPAN, AXIS_SPAN + 1)


def build_grid_lines() -> list[tuple[np.ndarray, np.ndarray]]:
    """Construct the untransformed grid (horizontal + vertical lines)."""
    coords: list[tuple[np.ndarray, np.ndarray]] = []
    line_points = np.linspace(-AXIS_SPAN, AXIS_SPAN, 64)
    for value in GRID_VALUES:
        coords.append((np.full_like(line_points, value), line_points))  # vertical line
        coords.append((line_points, np.full_like(line_points, value)))  # horizontal line
    return coords


def apply_transform(x: np.ndarray, y: np.ndarray, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply a linear transform to a vectorized set of points."""
    transformed = matrix @ np.vstack((x, y))
    return transformed[0], transformed[1]


def main() -> None:
    grid_coords = build_grid_lines()

    fig, ax = plt.subplots(figsize=(7, 8))
    plt.subplots_adjust(bottom=0.2)
    ax.set_xlim(-AXIS_SPAN, AXIS_SPAN)
    ax.set_ylim(-AXIS_SPAN, AXIS_SPAN)
    ax.set_aspect("equal")
    ax.axhline(0, color="black", linewidth=1, alpha=0.4)
    ax.axvline(0, color="black", linewidth=1, alpha=0.4)
    ax.grid(False)

    # Low-alpha background grid for reference.
    for x, y in grid_coords:
        ax.plot(x, y, color="lightgray", linewidth=0.5, zorder=0)

    # Transformed grid lines that will be updated by the sliders.
    transformed_lines = []
    for _ in grid_coords:
        line, = ax.plot([], [], color="#377eb8", linewidth=0.8, alpha=0.9, zorder=1)
        transformed_lines.append(line)

    arrow_style = dict(arrowstyle="-|>", mutation_scale=12, linewidth=2)
    basis_x_arrow = FancyArrowPatch((0, 0), (0, 0), color="#e41a1c", **arrow_style)
    basis_y_arrow = FancyArrowPatch((0, 0), (0, 0), color="#4daf4a", **arrow_style)
    ax.add_patch(basis_x_arrow)
    ax.add_patch(basis_y_arrow)
    sum_point = ax.scatter([], [], color="black", s=35, zorder=3)

    slider_red = "#f4c7c3"
    slider_green = "#c6e6c3"
    ax_xx = plt.axes([0.15, 0.18, 0.7, 0.03], facecolor=slider_red)
    ax_xy = plt.axes([0.15, 0.13, 0.7, 0.03], facecolor=slider_red)
    ax_yx = plt.axes([0.15, 0.08, 0.7, 0.03], facecolor=slider_green)
    ax_yy = plt.axes([0.15, 0.03, 0.7, 0.03], facecolor=slider_green)

    slider_kwargs = dict(valmin=SLIDER_MIN, valmax=SLIDER_MAX, valstep=SLIDER_STEP)
    slider_xx = Slider(ax_xx, "x axis (x)", valinit=1.0, **slider_kwargs)
    slider_xy = Slider(ax_xy, "x axis (y)", valinit=0.0, **slider_kwargs)
    slider_yx = Slider(ax_yx, "y axis (x)", valinit=0.0, **slider_kwargs)
    slider_yy = Slider(ax_yy, "y axis (y)", valinit=1.0, **slider_kwargs)

    current_vectors = {
        "x": np.array([1.0, 0.0]),
        "y": np.array([0.0, 1.0]),
        "sum": np.array([1.0, 1.0]),
    }

    def update(_value: float | None = None) -> None:
        matrix = np.array(
            [
                [slider_xx.val, slider_yx.val],
                [slider_xy.val, slider_yy.val],
            ]
        )

        for (orig_x, orig_y), line in zip(grid_coords, transformed_lines):
            tx, ty = apply_transform(orig_x, orig_y, matrix)
            line.set_data(tx, ty)

        basis_x = matrix @ np.array([1, 0])
        basis_y = matrix @ np.array([0, 1])
        basis_x_arrow.set_positions((0, 0), (basis_x[0], basis_x[1]))
        basis_y_arrow.set_positions((0, 0), (basis_y[0], basis_y[1]))
        sum_vec = basis_x + basis_y
        basis_x_arrow.set_zorder(3)
        basis_y_arrow.set_zorder(3)
        sum_point.set_offsets([sum_vec])
        current_vectors["x"] = basis_x
        current_vectors["y"] = basis_y
        current_vectors["sum"] = sum_vec

        fig.canvas.draw_idle()

    for slider in (slider_xx, slider_xy, slider_yx, slider_yy):
        slider.on_changed(update)

    drag_state = {"active": None}

    def handle_press(event) -> None:
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        click = np.array([event.xdata, event.ydata])
        for key in ("x", "y", "sum"):
            vec = current_vectors[key]
            if np.linalg.norm(click - vec) <= CLICK_RADIUS:
                drag_state["active"] = key
                break

    def handle_release(_event) -> None:
        drag_state["active"] = None

    def handle_motion(event) -> None:
        if drag_state["active"] is None or event.inaxes != ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        new_x = float(np.clip(event.xdata, SLIDER_MIN, SLIDER_MAX))
        new_y = float(np.clip(event.ydata, SLIDER_MIN, SLIDER_MAX))
        if drag_state["active"] == "x":
            slider_xx.set_val(new_x)
            slider_xy.set_val(new_y)
        elif drag_state["active"] == "y":
            slider_yx.set_val(new_x)
            slider_yy.set_val(new_y)
        elif drag_state["active"] == "sum":
            target = np.array(
                [
                    float(np.clip(event.xdata, -AXIS_SPAN, AXIS_SPAN)),
                    float(np.clip(event.ydata, -AXIS_SPAN, AXIS_SPAN)),
                ]
            )
            current_sum = current_vectors["sum"]
            delta_each = (target - current_sum) / 2.0

            def shift_pair(slider_x: Slider, slider_y: Slider) -> None:
                slider_x.set_val(float(np.clip(slider_x.val + delta_each[0], SLIDER_MIN, SLIDER_MAX)))
                slider_y.set_val(float(np.clip(slider_y.val + delta_each[1], SLIDER_MIN, SLIDER_MAX)))

            shift_pair(slider_xx, slider_xy)
            shift_pair(slider_yx, slider_yy)

    fig.canvas.mpl_connect("button_press_event", handle_press)
    fig.canvas.mpl_connect("button_release_event", handle_release)
    fig.canvas.mpl_connect("motion_notify_event", handle_motion)

    update()
    plt.show()


if __name__ == "__main__":
    main()
