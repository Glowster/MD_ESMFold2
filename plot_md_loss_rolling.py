#!/usr/bin/env python3
"""Plot rolling averages from an MD ESMFold2 training run loss.csv.

Usage:

    python3 plot_md_loss_rolling.py md_esmfold2_20260629_162510

The argument can be either a run folder name under ./runs or an explicit path.
The output SVG is written into the same run folder as loss.csv.
"""

from __future__ import annotations

import argparse
import csv
import html
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run",
        help="Run folder name under ./runs, or an explicit path to a run folder.",
    )
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--stat", choices=["mean", "median"], default="mean")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def resolve_run_dir(run: str) -> Path:
    path = Path(run).expanduser()
    if path.exists():
        return path.resolve()

    path = ROOT / "runs" / run
    if path.exists():
        return path.resolve()

    raise FileNotFoundError(
        f"could not find run folder {run!r} or {str(ROOT / 'runs' / run)!r}"
    )


def read_rows(loss_csv: Path) -> list[dict[str, str]]:
    with loss_csv.open(newline="") as handle:
        return list(csv.DictReader(handle))


def numeric_column(rows: list[dict[str, str]], key: str) -> list[float] | None:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is None or value == "":
            return None
        try:
            number = float(value)
        except ValueError:
            return None
        if not math.isfinite(number):
            return None
        values.append(number)
    return values


def rolling_mean(values: list[float], window: int) -> list[float | None]:
    if window <= 0:
        raise ValueError("--window must be positive")
    out: list[float | None] = []
    running = 0.0
    for i, value in enumerate(values):
        running += value
        if i >= window:
            running -= values[i - window]
        out.append(running / min(i + 1, window))
    return out


def rolling_median(values: list[float], window: int) -> list[float | None]:
    if window <= 0:
        raise ValueError("--window must be positive")
    out: list[float | None] = []
    for i in range(len(values)):
        start = max(0, i + 1 - window)
        out.append(statistics.median(values[start : i + 1]))
    return out


def rolling_stat(values: list[float], window: int, stat: str) -> list[float | None]:
    if stat == "mean":
        return rolling_mean(values, window)
    if stat == "median":
        return rolling_median(values, window)
    raise ValueError(f"unsupported rolling stat: {stat}")


def align_series(
    x_values: list[float],
    source_steps: list[float],
    source_values: list[float | None],
) -> list[float | None]:
    by_step = {
        step: value
        for step, value in zip(source_steps, source_values)
    }
    return [by_step.get(step) for step in x_values]


def has_finite(values: list[float | None]) -> bool:
    return any(value is not None and math.isfinite(value) for value in values)


def nice_ticks(lo: float, hi: float, count: int = 5) -> list[float]:
    if lo == hi:
        pad = abs(lo) * 0.1 if lo else 1.0
        lo -= pad
        hi += pad
    return [lo + (hi - lo) * i / (count - 1) for i in range(count)]


def polyline(
    xs: list[float],
    ys: list[float | None],
    x_map,
    y_map,
    color: str,
    width: float,
    opacity: float = 1.0,
    dasharray: str | None = None,
) -> str:
    points = [
        f"{x_map(x):.2f},{y_map(y):.2f}"
        for x, y in zip(xs, ys)
        if y is not None and math.isfinite(y)
    ]
    if len(points) < 2:
        return ""
    dash = f' stroke-dasharray="{html.escape(dasharray)}"' if dasharray else ""
    return (
        f'<polyline fill="none" stroke="{color}" stroke-width="{width}" '
        f'stroke-opacity="{opacity}"{dash} points="{" ".join(points)}" />'
    )


def markers(
    xs: list[float],
    ys: list[float | None],
    x_map,
    y_map,
    color: str,
) -> str:
    points = [
        f'<circle cx="{x_map(x):.2f}" cy="{y_map(y):.2f}" r="3.2" '
        f'fill="white" stroke="{color}" stroke-width="1.7" />'
        for x, y in zip(xs, ys)
        if y is not None and math.isfinite(y)
    ]
    return "\n".join(points)


def render_panel(
    *,
    title: str,
    x_values: list[float],
    series: list[tuple[str, list[float | None], str, float, str | None]],
    x: int,
    y: int,
    width: int,
    height: int,
    subtitle: str | None = None,
) -> str:
    finite_values = [
        value
        for _, ys, _, _, _ in series
        for value in ys
        if value is not None and math.isfinite(value)
    ]
    if not finite_values:
        return ""

    xmin = min(x_values)
    xmax = max(x_values)
    ymin = min(finite_values)
    ymax = max(finite_values)
    if ymin == ymax:
        pad = abs(ymin) * 0.1 if ymin else 1.0
        ymin -= pad
        ymax += pad

    left = x + 74
    right = x + width - 26
    available_legend_width = max(240, right - left)
    legend_rows: list[list[tuple[str, list[float | None], str, float, str | None, float]]] = [[]]
    current_width = 0.0
    for entry in series:
        label = entry[0]
        entry_width = 64 + max(88, len(label) * 7.4)
        if legend_rows[-1] and current_width + entry_width > available_legend_width:
            legend_rows.append([])
            current_width = 0.0
        legend_rows[-1].append((*entry, entry_width))
        current_width += entry_width

    legend_row_height = 18
    legend_top = y + height - 6 - legend_row_height * len(legend_rows)
    top = y + 42
    bottom = legend_top - 34

    def x_map(value: float) -> float:
        if xmin == xmax:
            return (left + right) / 2
        return left + (value - xmin) / (xmax - xmin) * (right - left)

    def y_map(value: float) -> float:
        return bottom - (value - ymin) / (ymax - ymin) * (bottom - top)

    parts: list[str] = [
        f'<text x="{x}" y="{y + 18}" class="title">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" class="axis" />',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" class="axis" />',
    ]
    if subtitle:
        parts.insert(
            1,
            f'<text x="{x}" y="{y + 36}" class="subtitle">{html.escape(subtitle)}</text>',
        )

    for tick in nice_ticks(ymin, ymax):
        yy = y_map(tick)
        parts.append(
            f'<line x1="{left}" y1="{yy:.2f}" x2="{right}" y2="{yy:.2f}" class="grid" />'
        )
        parts.append(
            f'<text x="{left - 10}" y="{yy + 4:.2f}" text-anchor="end" class="tick">'
            f"{tick:.3g}</text>"
        )

    for tick in nice_ticks(xmin, xmax):
        xx = x_map(tick)
        parts.append(
            f'<text x="{xx:.2f}" y="{bottom + 24}" text-anchor="middle" class="tick">'
            f"{tick:.0f}</text>"
        )

    for label, ys, color, stroke_width, dasharray in series:
        parts.append(polyline(x_values, ys, x_map, y_map, color, stroke_width, dasharray=dasharray))
        if dasharray:
            parts.append(markers(x_values, ys, x_map, y_map, color))

    for row_idx, row in enumerate(legend_rows):
        legend_x = left
        legend_y = legend_top + row_idx * legend_row_height + 10
        for label, _ys, color, stroke_width, dasharray, entry_width in row:
            dash = f' stroke-dasharray="{html.escape(dasharray)}"' if dasharray else ""
            parts.append(
                f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 24}" '
                f'y2="{legend_y}" stroke="{color}" stroke-width="{stroke_width}"{dash} />'
            )
            parts.append(
                f'<text x="{legend_x + 32}" y="{legend_y + 4}" class="legend">'
                f"{html.escape(label)}</text>"
            )
            legend_x += entry_width

    return "\n".join(parts)


def write_svg(
    output: Path,
    run_name: str,
    rows: list[dict[str, str]],
    window: int,
    stat: str,
    validation_rows: list[dict[str, str]] | None = None,
) -> None:
    steps = numeric_column(rows, "step")
    losses = numeric_column(rows, "loss")
    if steps is None or losses is None:
        raise ValueError("loss.csv must contain numeric step and loss columns")

    validation_rows = validation_rows or []
    validation_steps = numeric_column(validation_rows, "step") if validation_rows else None
    if validation_steps is not None:
        x_values = sorted(set(steps + validation_steps))
    else:
        x_values = steps

    stat_label = f"rolling {stat}"
    loss_roll = rolling_stat(losses, window, stat)
    loss_series = [(f"train loss {stat_label}-{window}", align_series(x_values, steps, loss_roll), "#0f766e", 2.6, None)]

    if validation_steps is not None:
        validation_losses = numeric_column(validation_rows, "loss")
        if validation_losses is not None:
            val_loss_series = align_series(x_values, validation_steps, validation_losses)
            if has_finite(val_loss_series):
                loss_series.append(("validation loss", val_loss_series, "#f97316", 2.4, "5 5"))

    panels = [
        render_panel(
            title=f"{run_name}: loss ({window}-step {stat_label})",
            x_values=x_values,
            series=loss_series,
            x=44,
            y=52,
            width=1120,
            height=310,
        )
    ]

    denoise_series: list[tuple[str, list[float | None], str, float, str | None]] = []
    values = numeric_column(rows, "denoise_rmsd")
    if values is not None:
        rolled = align_series(x_values, steps, rolling_stat(values, window, stat))
        if has_finite(rolled):
            denoise_series.append((f"train denoised RMSD {stat_label}-{window}", rolled, "#2563eb", 2.4, None))

    if validation_steps is not None:
        values = numeric_column(validation_rows, "denoise_rmsd")
        if values is not None:
            aligned = align_series(x_values, validation_steps, values)
            if has_finite(aligned):
                denoise_series.append(("validation denoised RMSD", aligned, "#7c3aed", 2.3, "5 5"))

    if denoise_series:
        panels.append(
            render_panel(
                title=f"{run_name}: denoised prediction RMSD ({window}-step {stat_label})",
                x_values=x_values,
                series=denoise_series,
                x=44,
                y=400,
                width=1120,
                height=310,
            )
        )

    noise_series: list[tuple[str, list[float | None], str, float, str | None]] = []
    values = numeric_column(rows, "noise_sigma_mean")
    if values is not None:
        rolled = align_series(x_values, steps, rolling_stat(values, window, stat))
        if has_finite(rolled):
            noise_series.append((f"train mean sigma_i {stat_label}-{window}", rolled, "#64748b", 2.4, None))

    if validation_steps is not None:
        values = numeric_column(validation_rows, "noise_sigma_mean")
        if values is not None:
            aligned = align_series(x_values, validation_steps, values)
            if has_finite(aligned):
                noise_series.append(("validation mean sigma_i", aligned, "#475569", 2.3, "5 5"))

    if noise_series:
        panels.append(
            render_panel(
                title=f"{run_name}: sampled noise scale ({window}-step {stat_label})",
                subtitle="Plotted value is mean sigma_i. Expected coordinate RMSD of the raw Gaussian corruption is about sqrt(3) * sigma_i.",
                x_values=x_values,
                series=noise_series,
                x=44,
                y=748,
                width=1120,
                height=310,
            )
        )

    height = 420 + 348 * (len(panels) - 1)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{height}" viewBox="0 0 1200 {height}">
<style>
  .title {{ font: 700 18px sans-serif; fill: #111827; }}
  .subtitle {{ font: 13px sans-serif; fill: #475569; }}
  .axis {{ stroke: #111827; stroke-width: 1.2; }}
  .grid {{ stroke: #e5e7eb; stroke-width: 1; }}
  .tick {{ font: 12px sans-serif; fill: #374151; }}
  .legend {{ font: 13px sans-serif; fill: #111827; }}
</style>
<rect width="1200" height="{height}" fill="white" />
{chr(10).join(panel for panel in panels if panel)}
</svg>
"""
    output.write_text(svg)


def main() -> int:
    args = parse_args()
    run_dir = resolve_run_dir(args.run)
    loss_csv = run_dir / "loss.csv"
    if not loss_csv.exists():
        raise FileNotFoundError(f"missing {loss_csv}")

    rows = read_rows(loss_csv)
    if not rows:
        raise ValueError(f"{loss_csv} has no rows")

    validation_csv = run_dir / "validation.csv"
    validation_rows = read_rows(validation_csv) if validation_csv.exists() else []

    default_name = f"loss_rolling{args.window}.svg"
    if args.stat != "mean":
        default_name = f"loss_rolling{args.window}_{args.stat}.svg"
    output = args.output or run_dir / default_name
    if not output.is_absolute():
        output = run_dir / output
    write_svg(output, run_dir.name, rows, args.window, args.stat, validation_rows=validation_rows)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
