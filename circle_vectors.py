from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.widgets import Button


RADIUS = 1.0
AXIS_LIMIT = 1.25
CLICK_DISTANCE = 0.12
VECTOR_COLORS = ["green", "blue", "red"]


def _format_1dp(value: float) -> str:
    if abs(value) < 0.05:
        value = 0.0
    return f"{value:.1f}"


def _distance_point_to_segment(point: np.ndarray, segment_end: np.ndarray) -> float:
    # Segment from origin -> segment_end.
    denom = float(np.dot(segment_end, segment_end))
    if denom == 0.0:
        return float(np.linalg.norm(point))
    t = float(np.dot(point, segment_end) / denom)
    t = min(1.0, max(0.0, t))
    closest = t * segment_end
    return float(np.linalg.norm(point - closest))


@dataclass
class UnitVectorArtist:
    angle: float
    color: str
    arrow: FancyArrowPatch
    proj_x: plt.Line2D
    proj_y: plt.Line2D
    label: plt.Text

    def set_selected(self, selected: bool) -> None:
        self.arrow.set_linewidth(3.5 if selected else 2.5)
        self.arrow.set_alpha(1.0 if selected else 0.9)

    def remove(self) -> None:
        self.arrow.remove()
        self.proj_x.remove()
        self.proj_y.remove()
        self.label.remove()

    def update_angle(self, angle: float) -> None:
        self.angle = float(angle)
        x, y = float(np.cos(self.angle)), float(np.sin(self.angle))

        self.arrow.set_positions((0, 0), (x, y))
        self.proj_x.set_data([x, x], [0, y])
        self.proj_y.set_data([0, x], [y, y])

        self.label.set_text(f"({_format_1dp(x)}, {_format_1dp(y)})")
        self.label.set_position((1.08 * x, 1.08 * y))
        self.label.set_horizontalalignment("left" if x >= 0 else "right")
        self.label.set_verticalalignment("bottom" if y >= 0 else "top")


def main() -> None:
    rng = np.random.default_rng()

    fig, ax = plt.subplots(figsize=(7, 7))
    plt.subplots_adjust(bottom=0.16)

    ax.set_xlim(-AXIS_LIMIT, AXIS_LIMIT)
    ax.set_ylim(-AXIS_LIMIT, AXIS_LIMIT)
    ax.set_aspect("equal")
    ax.axhline(0, color="black", linewidth=1, alpha=0.35)
    ax.axvline(0, color="black", linewidth=1, alpha=0.35)
    ax.set_title("Unit circle vectors (drag to rotate)")

    ax.add_patch(Circle((0, 0), RADIUS, fill=False, color="black", linewidth=1.5, alpha=0.85))

    vectors: list[UnitVectorArtist] = []
    drag_state = {"selected": None, "dragging": False}

    def set_selected(index: int | None) -> None:
        prev = drag_state["selected"]
        if prev is not None and 0 <= prev < len(vectors):
            vectors[prev].set_selected(False)
        drag_state["selected"] = index
        if index is not None and 0 <= index < len(vectors):
            vectors[index].set_selected(True)

    def next_available_color() -> str | None:
        used = {vec.color for vec in vectors}
        for color in VECTOR_COLORS:
            if color not in used:
                return color
        return None

    def add_vector() -> None:
        color = next_available_color()
        if color is None:
            return
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        x, y = float(np.cos(angle)), float(np.sin(angle))

        arrow = FancyArrowPatch(
            (0, 0),
            (x, y),
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=2.5,
            color=color,
            alpha=0.9,
            zorder=3,
        )
        ax.add_patch(arrow)

        dash = (0, (4, 4))
        (proj_x,) = ax.plot([x, x], [0, y], linestyle=dash, linewidth=0.8, color=color, alpha=0.55, zorder=2)
        (proj_y,) = ax.plot([0, x], [y, y], linestyle=dash, linewidth=0.8, color=color, alpha=0.55, zorder=2)

        label = ax.text(
            1.08 * x,
            1.08 * y,
            f"({_format_1dp(x)}, {_format_1dp(y)})",
            color=color,
            fontsize=10,
            ha="left" if x >= 0 else "right",
            va="bottom" if y >= 0 else "top",
            zorder=4,
        )

        vec = UnitVectorArtist(angle=angle, color=color, arrow=arrow, proj_x=proj_x, proj_y=proj_y, label=label)
        vectors.append(vec)
        set_selected(len(vectors) - 1)
        fig.canvas.draw_idle()

    def remove_vector() -> None:
        if not vectors:
            return
        idx = drag_state["selected"]
        if idx is None or not (0 <= idx < len(vectors)):
            idx = len(vectors) - 1
        removed = vectors.pop(idx)
        removed.remove()

        if vectors:
            set_selected(min(idx, len(vectors) - 1))
        else:
            drag_state["selected"] = None
        fig.canvas.draw_idle()

    ax_add = plt.axes([0.25, 0.04, 0.22, 0.08])
    ax_remove = plt.axes([0.53, 0.04, 0.22, 0.08])
    add_button = Button(ax_add, "Add vector", color="#f6f6f6", hovercolor="#e8e8e8")
    remove_button = Button(ax_remove, "Remove vector", color="#f6f6f6", hovercolor="#e8e8e8")

    def on_add(_event) -> None:
        add_vector()

    def on_remove(_event) -> None:
        remove_vector()

    add_button.on_clicked(on_add)
    remove_button.on_clicked(on_remove)

    def pick_vector(event) -> int | None:
        if event.xdata is None or event.ydata is None:
            return None
        click = np.array([event.xdata, event.ydata], dtype=float)

        best_idx = None
        best_dist = float("inf")
        for idx, vec in enumerate(vectors):
            end = np.array([np.cos(vec.angle), np.sin(vec.angle)], dtype=float)
            dist = min(_distance_point_to_segment(click, end), float(np.linalg.norm(click - end)))
            if dist < best_dist:
                best_dist = dist
                best_idx = idx

        if best_idx is not None and best_dist <= CLICK_DISTANCE:
            return best_idx
        return None

    def handle_press(event) -> None:
        if event.inaxes != ax or event.button != 1:
            return
        idx = pick_vector(event)
        if idx is None:
            set_selected(None)
            drag_state["dragging"] = False
            fig.canvas.draw_idle()
            return
        set_selected(idx)
        drag_state["dragging"] = True
        fig.canvas.draw_idle()

    def handle_release(_event) -> None:
        drag_state["dragging"] = False

    def handle_motion(event) -> None:
        if not drag_state["dragging"]:
            return
        idx = drag_state["selected"]
        if idx is None or not (0 <= idx < len(vectors)):
            return
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        x, y = float(event.xdata), float(event.ydata)
        if x == 0.0 and y == 0.0:
            return
        angle = float(np.arctan2(y, x))
        vectors[idx].update_angle(angle)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", handle_press)
    fig.canvas.mpl_connect("button_release_event", handle_release)
    fig.canvas.mpl_connect("motion_notify_event", handle_motion)

    plt.show()


if __name__ == "__main__":
    main()
