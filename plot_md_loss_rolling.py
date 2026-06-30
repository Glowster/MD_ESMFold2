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
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run",
        help="Run folder name under ./runs, or an explicit path to a run folder.",
    )
    parser.add_argument("--window", type=int, default=50)
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
        if i + 1 >= window:
            out.append(running / window)
        else:
            out.append(None)
    return out


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
) -> str:
    points = [
        f"{x_map(x):.2f},{y_map(y):.2f}"
        for x, y in zip(xs, ys)
        if y is not None and math.isfinite(y)
    ]
    if len(points) < 2:
        return ""
    return (
        f'<polyline fill="none" stroke="{color}" stroke-width="{width}" '
        f'stroke-opacity="{opacity}" points="{" ".join(points)}" />'
    )


def render_panel(
    *,
    title: str,
    x_values: list[float],
    series: list[tuple[str, list[float | None], str, float]],
    x: int,
    y: int,
    width: int,
    height: int,
) -> str:
    finite_values = [
        value
        for _, ys, _, _ in series
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
    top = y + 42
    bottom = y + height - 58

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

    legend_x = left
    legend_y = y + height - 18
    for label, ys, color, stroke_width in series:
        parts.append(
            f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 24}" '
            f'y2="{legend_y}" stroke="{color}" stroke-width="{stroke_width}" />'
        )
        parts.append(
            f'<text x="{legend_x + 32}" y="{legend_y + 4}" class="legend">'
            f"{html.escape(label)}</text>"
        )
        legend_x += 190
        parts.append(polyline(x_values, ys, x_map, y_map, color, stroke_width))

    return "\n".join(parts)


def write_svg(
    output: Path,
    run_name: str,
    rows: list[dict[str, str]],
    window: int,
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

    loss_roll = rolling_mean(losses, window)
    loss_series = [(f"train loss rolling-{window}", align_series(x_values, steps, loss_roll), "#0f766e", 2.6)]

    if validation_steps is not None:
        validation_losses = numeric_column(validation_rows, "loss")
        if validation_losses is not None:
            val_loss_series = align_series(x_values, validation_steps, validation_losses)
            if has_finite(val_loss_series):
                loss_series.append(("validation loss", val_loss_series, "#f97316", 2.4))

    panels = [
        render_panel(
            title=f"{run_name}: loss ({window}-step rolling average)",
            x_values=x_values,
            series=loss_series,
            x=44,
            y=52,
            width=1120,
            height=310,
        )
    ]

    rmsd_series: list[tuple[str, list[float | None], str, float]] = []
    for key, color in (("denoise_rmsd", "#2563eb"), ("noisy_rmsd", "#dc2626")):
        values = numeric_column(rows, key)
        if values is not None:
            rolled = align_series(x_values, steps, rolling_mean(values, window))
            if has_finite(rolled):
                rmsd_series.append((f"train {key} rolling-{window}", rolled, color, 2.4))

    if validation_steps is not None:
        for key, color in (("denoise_rmsd", "#7c3aed"), ("noisy_rmsd", "#ea580c")):
            values = numeric_column(validation_rows, key)
            if values is not None:
                aligned = align_series(x_values, validation_steps, values)
                if has_finite(aligned):
                    rmsd_series.append((f"validation {key}", aligned, color, 2.3))

    if rmsd_series:
        panels.append(
            render_panel(
                title=f"{run_name}: RMSD diagnostics ({window}-step rolling average)",
                x_values=x_values,
                series=rmsd_series,
                x=44,
                y=400,
                width=1120,
                height=310,
            )
        )

    height = 760 if rmsd_series else 420
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{height}" viewBox="0 0 1200 {height}">
<style>
  .title {{ font: 700 18px sans-serif; fill: #111827; }}
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

    output = args.output or run_dir / f"loss_rolling{args.window}.svg"
    if not output.is_absolute():
        output = run_dir / output
    write_svg(output, run_dir.name, rows, args.window, validation_rows=validation_rows)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
