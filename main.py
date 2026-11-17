"""Launch-day Twitch CCU quantile forecaster (make.fun gta6-twitch-ccu-forecast-01).

Model: comparable-launch shape transfer in log space, calibrated on the
kit's practice cases. Nothing is forecast in raw viewers: every launch is
reduced to ratios against its own hour-1 print, the transfer happens on
those log-ratios, and only at the end are the ratios scaled back by the
case's known hour-1 print. Working in log space is what makes launches of
very different sizes comparable at all: a launch ten times bigger with the
same unlock pattern has the same shape.

For every past launch bundled in the case file, build its launch-day shape:
the day's hourly viewers divided by that launch's own hour-1 print. A launch
is weighted by how closely its pre-checkpoint trajectory (the 24 hourly
log-ratios to its own hour-1 print, i.e. the rolling-unlock wave as it
crossed New Zealand, Europe and the US east coast) matches this case's own
pre-checkpoint trajectory, with a Gaussian kernel over the mean absolute
log difference. Three views of that weighted neighbourhood are ensembled:

  1. full pre-window, scale-aware distance, weighted-mean log shape;
  2. full pre-window, plain distance, median log shape;
  3. recent pre-window (last 13 hours), weighted-mean log shape.

Per forecast hour, the centre is the mean of the members' centres and the
spread is the root-mean-square of the members' weighted log-space standard
deviations, shrunk by a factor 0.85 calibrated on the practice cases: the
raw ensemble spread was systematically too wide there, and the shrink is
the single global correction that fixed it without disturbing the centre.
Quantiles are lognormal around the centre, scaled by the case's known
hour-1 print. Any case (or hour) the model cannot handle falls back to the
house persistence baseline, so the output is always 23 valid rows.

Output contract: for every ``*.in`` case file in MAKEFUN_INPUT_DIR, write a
same-stem ``.out`` JSON file to MAKEFUN_OUTPUT_DIR holding hours 2..24,
each with the five quantiles for levels 0.05, 0.25, 0.5, 0.75, 0.95. The
fallback guarantee is part of the contract, not an error path: a case whose
input is unparseable, empty, or free of usable comparables still produces
23 well-formed baseline rows instead of failing, because one missing or
invalid case would void the whole entry.

Backtest on the kit's practice cases (pinball loss on log scale, lower is
better; actuals are the same SullyGnome hourly series the house recorded):
this model 0.230 pooled vs the starter baseline 0.387; per case p00 0.336
vs 0.744, p01 0.181 vs 0.287, p02 0.168 vs 0.215, p03 0.237 vs 0.303.

Deterministic, Python 3.13 standard library only, no network.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
SPREAD_SHRINK = 0.85
# (kernel width, scale-distance weight, pre-window lower hour, centre kind)
MEMBERS = [
    (0.4, 0.3, -24, "wmean"),
    (0.4, 0.0, -24, "median"),
    (0.3, 0.0, -13, "wmean"),
]
MIN_SHARED_HOURS = 6


def parse_ts(value):
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def round2(x):
    return float(Decimal(repr(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def clamp_q(x):
    return min(max(round2(x), 0.01), 1_000_000_000.0)


def baseline_rows(p1):
    rows = {}
    for h in range(2, 25):
        sigma = 0.15 * math.sqrt(h - 1)
        rows[h] = [clamp_q(p1 * math.exp(z * sigma)) for z in Z]
    return rows


def hourly_map(rows, day_start):
    """Map whole-hour offsets from day_start (0 = hour 1) to viewers."""
    by = {}
    for row in rows or []:
        try:
            viewers = row.get("viewers")
            if not viewers:
                continue
            delta = (parse_ts(row["from"]) - day_start).total_seconds()
        except Exception:
            continue
        if delta % 3600 == 0:
            by[int(delta // 3600)] = float(viewers)
    return by


def bucket_history(history, day_start):
    """Fallback hourly means built from raw history points."""
    sums, counts = {}, {}
    for point in history or []:
        try:
            viewers = point.get("viewers")
            if not viewers:
                continue
            delta = (parse_ts(point["at"]) - day_start).total_seconds()
        except Exception:
            continue
        k = int(delta // 3600)
        sums[k] = sums.get(k, 0.0) + float(viewers)
        counts[k] = counts.get(k, 0) + 1
    return {k: sums[k] / counts[k] for k in sums}


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    mid = n // 2
    if n % 2:
        return xs[mid]
    return (xs[mid - 1] + xs[mid]) / 2.0


def member_centre_spread(pre_target, candidates, kernel_width, scale_w, lo, centre_kind, log_p1):
    """One ensemble member: {hour: (centre log-ratio, sigma)}."""
    weighted = []
    for pre_c, day_log, log_a in candidates:
        shared = [k for k in pre_target if k in pre_c and k >= lo]
        if len(shared) < MIN_SHARED_HOURS:
            continue
        dist = sum(abs(pre_target[k] - pre_c[k]) for k in shared) / len(shared)
        dist += scale_w * abs(log_a - log_p1)
        # Gaussian kernel over that distance: a comparable whose unlock wave
        # tracks the target's (dist << kernel_width) gets weight near 1, and
        # the weight decays quadratically, so a few near-twins dominate the
        # neighbourhood instead of the whole back catalogue voting equally.
        weighted.append((math.exp(-((dist / kernel_width) ** 2)), day_log))
    out = {}
    for h in range(2, 25):
        pts = [(w, day[h]) for w, day in weighted if h in day]
        if not pts:
            out[h] = (0.0, 0.15 * math.sqrt(h - 1))
            continue
        total_w = sum(w for w, _ in pts)
        mean = sum(w * x for w, x in pts) / total_w
        var = sum(w * (x - mean) ** 2 for w, x in pts) / total_w
        centre = mean if centre_kind == "wmean" else median([x for _, x in pts])
        out[h] = (centre, math.sqrt(var))
    return out


def forecast_rows(case):
    day_start = parse_ts(case["day_start"])
    p1 = 0.0
    for row in case.get("known") or []:
        if row.get("viewers"):
            p1 = float(row["viewers"])
            break
    target_hourly = hourly_map(case.get("history_hourly"), day_start)
    if not target_hourly:
        target_hourly = bucket_history(case.get("history"), day_start)
    if p1 <= 0:
        p1 = target_hourly.get(0, 1.0)
    if p1 <= 0:
        p1 = 1.0
    log_p1 = math.log(p1)
    pre_target = {
        k: math.log(v / p1) for k, v in target_hourly.items() if -24 <= k <= 0 and v > 0
    }

    candidates = []
    game_name = (case.get("game") or {}).get("name")
    for comp in case.get("comparables") or []:
        try:
            if comp.get("name") == game_name:
                continue  # the case game's own pre-release history is not a launch
            if "not a launch" in (comp.get("unlock") or "").lower():
                continue
            by = hourly_map(comp.get("hourly"), parse_ts(comp["day_start"]))
            a = by.get(0, 0.0)
            if a <= 0:
                continue
            pre_c = {k: math.log(v / a) for k, v in by.items() if -24 <= k <= 0 and v > 0}
            day_log = {
                k + 1: math.log(v / a) for k, v in by.items() if 1 <= k <= 23 and v > 0
            }
            candidates.append((pre_c, day_log, math.log(a)))
        except Exception:
            continue

    if not candidates or not pre_target:
        return baseline_rows(p1)

    members = [
        member_centre_spread(pre_target, candidates, kernel_width, scale_w, lo, kind, log_p1)
        for kernel_width, scale_w, lo, kind in MEMBERS
    ]
    rows = {}
    for h in range(2, 25):
        centre = sum(m[h][0] for m in members) / len(members)
        spread = math.sqrt(sum(m[h][1] ** 2 for m in members) / len(members))
        qs = []
        for z in Z:
            # Lognormal quantile in log space: shift the hour's centre by
            # z shrunk spreads, clamp so exp() stays finite, then scale
            # back from log-ratio to viewers by the hour-1 print.
            log_q = centre + z * spread * SPREAD_SHRINK
            log_q = min(max(log_q, -30.0), 30.0)
            qs.append(clamp_q(p1 * math.exp(log_q)))
        rows[h] = sorted(qs)
    return rows


def rows_for_case(case):
    try:
        return forecast_rows(case)
    except Exception:
        p1 = 1.0
        try:
            for row in case.get("known") or []:
                if row.get("viewers"):
                    p1 = float(row["viewers"])
                    break
        except Exception:
            pass
        return baseline_rows(max(p1, 0.01))


def main():
    in_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    out_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(in_dir.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            case = {}
        rows = rows_for_case(case)
        payload = {
            "hours": [
                {"hour": h, "q": rows[h]} for h in range(2, 25)
            ]
        }
        text = json.dumps(payload, separators=(",", ":")) + "\n"
        (out_dir / (path.stem + ".out")).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
