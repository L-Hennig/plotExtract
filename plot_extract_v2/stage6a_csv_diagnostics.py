from __future__ import annotations

import csv
import io
import json
import math
import re
import statistics
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

EPS = 0.02  # tolerance in log10 units for "flat"
TIME0_TOL_HOURS = 0.05  # tolerance for detecting time == 0


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        s = str(value).strip()
        if s == "" or s.lower() in {"none", "nan", "null"}:
            return None
        v = float(s)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def _extract_condition_label(y_col_name: str, fallback_idx: int) -> str:
    # Prefer text in parentheses, e.g. "log10 CFU/mL (Condition 1)"
    m = re.search(r"\(([^)]+)\)", y_col_name)
    if m:
        return m.group(1).strip()
    return f"Condition {fallback_idx}"


def _parse_wide_csv(csv_text: str) -> List[Tuple[str, str, List[Tuple[float, float]]]]:
    """Parse wide-format CSV into curves.

    Returns list of (curve_id, y_col_name, points[(time_hours, y)]).

    Wide format expected: x1,y1,x2,y2,... with repeated time column per curve.
    """
    reader = csv.reader(io.StringIO(csv_text.strip()))
    rows = list(reader)
    if not rows:
        return []

    header = [h.strip() for h in rows[0]]
    if len(header) < 2 or len(header) % 2 != 0:
        return []

    curves: List[Tuple[str, str, List[Tuple[float, float]]]] = []
    for pair_idx in range(0, len(header), 2):
        x_name = header[pair_idx]
        y_name = header[pair_idx + 1]
        curve_idx = pair_idx // 2 + 1
        curve_id = _extract_condition_label(y_name, curve_idx)

        points: List[Tuple[float, float]] = []
        for row in rows[1:]:
            if pair_idx >= len(row) or pair_idx + 1 >= len(row):
                continue
            t = _to_float(row[pair_idx])
            y = _to_float(row[pair_idx + 1])
            if t is None or y is None:
                continue
            # snap near-zero times to 0
            if abs(t) <= TIME0_TOL_HOURS:
                t = 0.0
            points.append((t, y))

        # de-duplicate times, keeping the last observed value
        dedup: Dict[float, float] = {}
        for t, y in points:
            dedup[t] = y
        points_sorted = sorted(dedup.items(), key=lambda kv: kv[0])
        curves.append((curve_id, y_name, [(t, y) for t, y in points_sorted]))

    return curves


def _step_directions(times: List[float], values: List[float]) -> List[str]:
    dirs: List[str] = []
    for i in range(1, len(values)):
        d = values[i] - values[i - 1]
        if abs(d) <= EPS:
            dirs.append("flat")
        elif d > EPS:
            dirs.append("increase")
        else:
            dirs.append("decrease")
    return dirs


def _pct_direction(directions: List[str], target: str) -> float:
    denom = sum(1 for d in directions if d != "flat")
    if denom == 0:
        return 0.0
    return sum(1 for d in directions if d == target) / denom


def _largest_jump(times: List[float], values: List[float]) -> Dict[str, Any]:
    # returns abs jump info
    if len(values) < 2:
        return {
            "abs_jump": 0.0,
            "from_time": None,
            "to_time": None,
            "delta": None,
        }

    best_abs = -1.0
    best = (None, None, None)
    for i in range(1, len(values)):
        delta = values[i] - values[i - 1]
        abs_jump = abs(delta)
        if abs_jump > best_abs:
            best_abs = abs_jump
            best = (times[i - 1], times[i], delta)

    return {
        "abs_jump": float(best_abs if best_abs >= 0 else 0.0),
        "from_time": best[0],
        "to_time": best[1],
        "delta": best[2],
    }


def _first_direction_change_index(directions: List[str]) -> Optional[int]:
    last: Optional[str] = None
    for i, d in enumerate(directions):
        if d == "flat":
            continue
        if last is None:
            last = d
            continue
        if d != last:
            return i
    return None


def _choose_reference_times(curves_times: Dict[str, List[float]]) -> List[float]:
    if not curves_times:
        return []
    # choose curve with most points
    best_curve = max(curves_times.keys(), key=lambda cid: len(curves_times[cid]))
    return sorted(curves_times[best_curve])


@dataclass
class CurveDiagnostics:
    curve_id: str
    expected_n_points: int
    n_points: int
    times_present: List[float]
    times_missing: List[float]
    y_at_time0: Optional[float]
    baseline_delta_from_median: Optional[float]
    step_directions: List[str]
    pct_increasing: float
    pct_decreasing: float
    largest_jump: Dict[str, Any]
    first_direction_change_step_index: Optional[int]


def compute_csv_diagnostics_from_text(csv_text: str) -> Dict[str, Any]:
    """Compute deterministic diagnostics for a wide-format CSV string."""
    curves = _parse_wide_csv(csv_text)

    curves_times: Dict[str, List[float]] = {}
    curves_values: Dict[str, List[float]] = {}
    curves_y0: Dict[str, Optional[float]] = {}

    for curve_id, _y_name, points in curves:
        times = [t for t, _ in points]
        values = [y for _, y in points]
        curves_times[curve_id] = times
        curves_values[curve_id] = values

        y0 = None
        for t, y in points:
            if abs(t) <= TIME0_TOL_HOURS:
                y0 = y
                break
        curves_y0[curve_id] = y0

    reference_times = _choose_reference_times(curves_times)

    curve_diags: List[CurveDiagnostics] = []
    for curve_id in curves_times.keys():
        times = curves_times[curve_id]
        values = curves_values[curve_id]

        directions = _step_directions(times, values)
        pct_inc = _pct_direction(directions, "increase")
        pct_dec = _pct_direction(directions, "decrease")

        median_y = statistics.median(values) if values else None
        y0 = curves_y0[curve_id]
        baseline_delta = (y0 - median_y) if (y0 is not None and median_y is not None) else None

        times_missing: List[float] = []
        if reference_times:
            present = set(times)
            times_missing = [t for t in reference_times if t not in present]

        curve_diags.append(
            CurveDiagnostics(
                curve_id=curve_id,
                expected_n_points=len(reference_times),
                n_points=len(times),
                times_present=times,
                times_missing=times_missing,
                y_at_time0=y0,
                baseline_delta_from_median=baseline_delta,
                step_directions=directions,
                pct_increasing=pct_inc,
                pct_decreasing=pct_dec,
                largest_jump=_largest_jump(times, values),
                first_direction_change_step_index=_first_direction_change_index(directions),
            )
        )

    return {
        "assumptions": [
            "y_values_are_log10_units: true",
            f"time0_tolerance_hours: {TIME0_TOL_HOURS}",
            "reference_times_rule: curve_with_most_points",
        ],
        "reference_times": reference_times,
        "curves": [asdict(cd) for cd in curve_diags],
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Stage 6a: deterministic CSV diagnostics")
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--out", default="csv_diagnostics.json")
    args = parser.parse_args()

    with open(args.csv_path, "r", encoding="utf-8") as f:
        csv_text = f.read()

    diag = compute_csv_diagnostics_from_text(csv_text)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2)

    print(f"Diagnostics written to {args.out}")


if __name__ == "__main__":
    main()
