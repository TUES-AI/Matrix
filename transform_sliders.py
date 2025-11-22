from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from matplotlib.widgets import Slider, Button, TextBox


AXIS_SPAN = 6
SLIDER_MIN = -3.0
SLIDER_MAX = 3.0
SLIDER_STEP = 0.05
CLICK_RADIUS = 0.3
GRID_VALUES = np.arange(-AXIS_SPAN, AXIS_SPAN + 1)


def build_grid_lines() -> list[tuple[np.ndarray, np.ndarray]]:
    coords: list[tuple[np.ndarray, np.ndarray]] = []
    line_points = np.linspace(-AXIS_SPAN, AXIS_SPAN, 64)
    for value in GRID_VALUES:
        coords.append((np.full_like(line_points, value), line_points))  # vertical line
        coords.append((line_points, np.full_like(line_points, value)))  # horizontal line
    return coords


def apply_transform(x: np.ndarray, y: np.ndarray, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    transformed = matrix @ np.vstack((x, y))
    return transformed[0], transformed[1]


def rotation_matrix(angle_deg: float) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def main() -> None:
    grid_coords = build_grid_lines()

    fig, ax = plt.subplots(figsize=(10, 12))
    plt.subplots_adjust(left=0.07, right=0.96, top=0.98, bottom=0.6)
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
    custom_arrow = FancyArrowPatch((0, 0), (0, 0), color="#ff7f00", linewidth=2, arrowstyle="-|>", mutation_scale=12)
    custom_arrow.set_visible(False)
    custom_arrow.set_zorder(4)
    ax.add_patch(basis_x_arrow)
    ax.add_patch(basis_y_arrow)
    ax.add_patch(custom_arrow)
    sum_point = ax.scatter([], [], color="black", s=35, zorder=3)

    slider_red = "#f4c7c3"
    slider_green = "#c6e6c3"
    slider_blue = "#d4e2ff"
    slider_height = 0.035
    slider_gap = 0.002
    slider_x = 0.15
    slider_width = 0.7
    top_y = 0.52
    ax_xx = plt.axes([slider_x, top_y, slider_width, slider_height], facecolor=slider_red)
    ax_xy = plt.axes([slider_x, top_y - (slider_height + slider_gap), slider_width, slider_height], facecolor=slider_red)
    ax_yx = plt.axes([slider_x, top_y - 2 * (slider_height + slider_gap), slider_width, slider_height], facecolor=slider_green)
    ax_yy = plt.axes([slider_x, top_y - 3 * (slider_height + slider_gap), slider_width, slider_height], facecolor=slider_green)
    ax_rot = plt.axes([slider_x, top_y - 4 * (slider_height + slider_gap), slider_width, slider_height], facecolor=slider_blue)
    controls_y = top_y - 5 * (slider_height + slider_gap) - 0.01
    ax_vec_x = plt.axes([0.15, controls_y, 0.18, 0.045])
    ax_vec_y = plt.axes([0.36, controls_y, 0.18, 0.045])
    ax_add_vec = plt.axes([0.58, controls_y, 0.17, 0.045])
    ax_reset = plt.axes([0.39, controls_y - 0.1, 0.22, 0.05])

    slider_kwargs = dict(valmin=SLIDER_MIN, valmax=SLIDER_MAX, valstep=SLIDER_STEP)
    slider_xx = Slider(ax_xx, "x axis (x)", valinit=1.0, **slider_kwargs)
    slider_xy = Slider(ax_xy, "x axis (y)", valinit=0.0, **slider_kwargs)
    slider_yx = Slider(ax_yx, "y axis (x)", valinit=0.0, **slider_kwargs)
    slider_yy = Slider(ax_yy, "y axis (y)", valinit=1.0, **slider_kwargs)
    slider_rotation = Slider(ax_rot, "rotation (°)", valmin=-180.0, valmax=180.0, valinit=0.0, valstep=1.0)
    vector_x_box = TextBox(ax_vec_x, "vector x", initial="2")
    vector_y_box = TextBox(ax_vec_y, "vector y", initial="3")
    add_vector_button = Button(ax_add_vec, "Add vector", color="#f8f8f8", hovercolor="#e8e8e8")
    reset_button = Button(ax_reset, "Reset", color="#f0f0f0", hovercolor="#e0e0e0")

    current_vectors = {
        "x": np.array([1.0, 0.0]),
        "y": np.array([0.0, 1.0]),
        "sum": np.array([1.0, 1.0]),
    }
    rotation_tracker = {"prev": 0.0}
    custom_vector = {"base": None, "display": None}
    current_matrix = {"matrix": np.eye(2)}

    def refresh_custom_arrow() -> None:
        if custom_vector["base"] is None:
            custom_vector["display"] = None
            custom_arrow.set_visible(False)
            return
        vec = current_matrix["matrix"] @ custom_vector["base"]
        custom_vector["display"] = vec
        custom_arrow.set_positions((0, 0), (vec[0], vec[1]))
        custom_arrow.set_visible(True)

    def set_custom_vector(base_vec: np.ndarray | None, sync_inputs: bool = False) -> None:
        if base_vec is None:
            custom_vector["base"] = None
            custom_vector["display"] = None
            custom_arrow.set_visible(False)
        else:
            base_vec = np.array(base_vec, dtype=float)
            custom_vector["base"] = base_vec
            if sync_inputs:
                vector_x_box.set_val(f"{base_vec[0]:.2f}")
                vector_y_box.set_val(f"{base_vec[1]:.2f}")
        refresh_custom_arrow()
        fig.canvas.draw_idle()

    def update(_value: float | None = None) -> None:
        base_x = np.array([slider_xx.val, slider_xy.val])
        base_y = np.array([slider_yx.val, slider_yy.val])
        matrix = np.column_stack((base_x, base_y))
        current_matrix["matrix"] = matrix
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

        refresh_custom_arrow()
        fig.canvas.draw_idle()

    for slider in (slider_xx, slider_xy, slider_yx, slider_yy):
        slider.on_changed(update)

    def handle_rotation(angle: float) -> None:
        delta = angle - rotation_tracker["prev"]
        if delta == 0.0:
            return
        rot_delta = rotation_matrix(delta)
        x_vec = np.array([slider_xx.val, slider_xy.val])
        y_vec = np.array([slider_yx.val, slider_yy.val])
        new_x = rot_delta @ x_vec
        new_y = rot_delta @ y_vec

        def set_axis(slider_x: Slider, slider_y: Slider, vec: np.ndarray) -> None:
            slider_x.set_val(float(np.clip(vec[0], SLIDER_MIN, SLIDER_MAX)))
            slider_y.set_val(float(np.clip(vec[1], SLIDER_MIN, SLIDER_MAX)))

        set_axis(slider_xx, slider_xy, new_x)
        set_axis(slider_yx, slider_yy, new_y)
        rotation_tracker["prev"] = angle

    slider_rotation.on_changed(handle_rotation)

    def handle_add_vector(_event) -> None:
        try:
            x_val = float(vector_x_box.text)
            y_val = float(vector_y_box.text)
        except ValueError:
            return
        set_custom_vector(np.array([x_val, y_val]), sync_inputs=True)

    add_vector_button.on_clicked(handle_add_vector)

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
        else:
            display_vec = custom_vector["display"]
            if display_vec is not None and np.linalg.norm(click - display_vec) <= CLICK_RADIUS:
                drag_state["active"] = "custom"

    def handle_release(_event) -> None:
        drag_state["active"] = None

    def handle_motion(event) -> None:
        if drag_state["active"] is None or event.inaxes != ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        target = np.array([event.xdata, event.ydata])
        if drag_state["active"] == "x":
            slider_xx.set_val(float(np.clip(target[0], SLIDER_MIN, SLIDER_MAX)))
            slider_xy.set_val(float(np.clip(target[1], SLIDER_MIN, SLIDER_MAX)))
        elif drag_state["active"] == "y":
            slider_yx.set_val(float(np.clip(target[0], SLIDER_MIN, SLIDER_MAX)))
            slider_yy.set_val(float(np.clip(target[1], SLIDER_MIN, SLIDER_MAX)))
        elif drag_state["active"] == "sum":
            clamped = np.array(
                [
                    float(np.clip(target[0], -AXIS_SPAN, AXIS_SPAN)),
                    float(np.clip(target[1], -AXIS_SPAN, AXIS_SPAN)),
                ]
            )
            current_sum = current_vectors["sum"]
            delta_each = (clamped - current_sum) / 2.0

            def shift_pair(slider_x: Slider, slider_y: Slider, new_vec: np.ndarray) -> None:
                slider_x.set_val(float(np.clip(new_vec[0], SLIDER_MIN, SLIDER_MAX)))
                slider_y.set_val(float(np.clip(new_vec[1], SLIDER_MIN, SLIDER_MAX)))

            base_x = np.array([slider_xx.val, slider_xy.val])
            base_y = np.array([slider_yx.val, slider_yy.val])
            shift_pair(slider_xx, slider_xy, base_x + delta_each)
            shift_pair(slider_yx, slider_yy, base_y + delta_each)
        elif drag_state["active"] == "custom":
            clamped = np.array(
                [
                    float(np.clip(target[0], -AXIS_SPAN, AXIS_SPAN)),
                    float(np.clip(target[1], -AXIS_SPAN, AXIS_SPAN)),
                ]
            )
            matrix = current_matrix["matrix"]
            try:
                base_vec = np.linalg.solve(matrix, clamped)
            except np.linalg.LinAlgError:
                base_vec, *_ = np.linalg.lstsq(matrix, clamped, rcond=None)
            set_custom_vector(base_vec, sync_inputs=True)

    def handle_reset(_event) -> None:
        rotation_tracker["prev"] = 0.0
        slider_rotation.set_val(0.0)
        slider_xx.set_val(1.0)
        slider_xy.set_val(0.0)
        slider_yx.set_val(0.0)
        slider_yy.set_val(1.0)
        set_custom_vector(None)
        vector_x_box.set_val("2")
        vector_y_box.set_val("3")

    reset_button.on_clicked(handle_reset)
    fig.canvas.mpl_connect("button_press_event", handle_press)
    fig.canvas.mpl_connect("button_release_event", handle_release)
    fig.canvas.mpl_connect("motion_notify_event", handle_motion)

    update()
    plt.show()


if __name__ == "__main__":
    main()
