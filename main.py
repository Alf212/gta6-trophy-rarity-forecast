"""GTA VI trophy rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Reads a case file holding every read the house made of Exophase's Grand Theft
Auto VI PS5 trophy page before the case's checkpoint, and writes, for each
trophy, five quantiles (5/25/50/75/95%) of its earn rate at the case's target,
in percent. Scored by pinball loss on the logit scale, so the program's job is
an honest predictive distribution per trophy, not just a point guess.

Model, per trophy, from the case's own history alone (no network, stdlib only):

  * The page only changes when Exophase refreshes it, so the history is first
    compressed to its distinct snapshots (a new snapshot when the players
    tracked or any rate moves). Everything is fitted on snapshots.
  * Trend model: a weighted least squares line of the trophy's logit rate
    against ln(1 + hours since release). Progress through a game is fast
    early and slows, so log-time is the natural clock; the fit is weighted
    toward recent snapshots and its slope is shrunk toward zero by its own
    standard error, so a noisy flat series stays flat.
  * Count model: a rate is earners / players tracked. The pool size and each
    trophy's earner count are extrapolated separately (log-log in the same
    log-time clock, earners never allowed to shrink) and divided. This sees
    dilution coming: when the tracked pool grows faster than a trophy's
    earners, the rate falls even while more players earn it. It only advises
    the median, at a lower weight: two free extrapolations divided compound
    each other's error, while the trend fit already sees the net rate.
  * The median is the weighted average of the two models' logits when both
    exist. Their disagreement widens the spread.
  * The spread (logit sd) combines a random-walk term growing with the
    square root of the horizon, the trend fit's own uncertainty at the
    target, the binomial noise of the last read's counts, and the two
    models' disagreement.
  * Stale page: if the tracked pool never moved during the whole history
    while the game is already out (the launch-night signature: Exophase's
    cache is stale and a refresh is overdue), the forecast assumes the pool
    jumps before the target and rates dilute toward earners / a much larger
    pool, with a wide, downward-skewed spread. A trophy every tracked player
    holds is exempt: a refresh does not dilute what everyone earns at once.

Deterministic: the output is a pure function of the case file.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader runs python3 -P -s, which keeps this file's directory off
# sys.path; put it back first so a sibling module would import.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.674490, 0.0, 0.674490, 1.644854]

# Random-walk diffusion of a logit rate, per square-root day of horizon.
DIFFUSION = 0.42
# Half-life (hours) of the recency weights on snapshots in the fits. Short
# enough that the slope is the recent regime's: a progression curve's slope
# in log-time steepens as the launch cohort advances, and it is the slope
# near the checkpoint that carries into the target week.
WEIGHT_HALF_LIFE_H = 18.0
# Assumed pool multiplier when the page proves stale across the launch.
STALE_POOL_GROWTH = 5.0


def clamp_rate(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float:
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def parse_time(text: str) -> float:
    """ISO 8601 (with Z) to epoch seconds."""
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def wls(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (a, b, resid_sd, sxx, xbar)."""
    sw = sum(ws)
    xbar = sum(w * x for x, w in zip(xs, ws)) / sw
    ybar = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - xbar) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return ybar, 0.0, 0.0, sxx, xbar
    sxy = sum(w * (x - xbar) * (y - ybar) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    a = ybar - b * xbar
    dof = max(1.0, sw - 2.0)
    var = sum(w * (y - (a + b * x)) ** 2 for x, y, w in zip(xs, ys, ws)) / dof
    return a, b, math.sqrt(max(var, 0.0)), sxx, xbar


class Case:
    def __init__(self, case: dict):
        self.trophies = case["trophies"]
        self.horizon_days = case["horizon_minutes"] / 1440.0
        game = case.get("game", {})
        release = game.get("release_date", "2026-11-19")
        self.release_ts = datetime.fromisoformat(release).replace(tzinfo=timezone.utc).timestamp()
        self.checkpoint_ts = parse_time(case["at"])
        self.target_ts = parse_time(case["target"])
        self.n_reads = len(case["history"])
        # Distinct snapshots: a new one whenever the page actually changed.
        self.snaps = []  # (ts, players, rates tuple)
        prev = None
        for read in case["history"]:
            key = (read["players_tracked"], tuple(read["rates"]))
            if key != prev:
                ts = parse_time(read.get("at") or read["slot"])
                self.snaps.append((ts, read["players_tracked"], read["rates"]))
                prev = key

    def x_of(self, ts: float) -> float:
        """The log-time clock: ln(1 + hours since release)."""
        age_h = max(0.0, (ts - self.release_ts) / 3600.0)
        return math.log1p(age_h)


def forecast_pool(case: Case):
    """Whether the tracked pool ever moved during the history.

    Returns (last_players, frozen): frozen says the pool never moved in the
    whole history — the stale-page signature. (The pool's own extrapolation
    lives in the per-trophy dilution guard, fitted on the same snapshots as
    the earners it is divided by.)
    """
    last_p = case.snaps[-1][1]
    # "Frozen" is a claim about the page, so it needs at least two reads to
    # prove the pool never moved; a single read says nothing either way.
    frozen = case.n_reads >= 2 and all(p == last_p for _, p, _ in case.snaps)
    return float(last_p), frozen


def forecast_trophy(case: Case, idx: int, pool_frozen: bool):
    """Five quantiles (percent) for one trophy."""
    snaps = case.snaps
    x_t = case.x_of(case.target_ts)
    t_last = snaps[-1][0]

    # This trophy's series over snapshots (skip reads that lacked it).
    series = []  # (x, ts, players, rate)
    for ts, players, rates in snaps:
        r = rates[idx]
        if r is not None:
            series.append((case.x_of(ts), ts, players, r))
    if not series:
        return [0.01, 0.01, 0.01, 0.01, 0.01]
    _, _, p_last, r_last = series[-1]
    earners_last = r_last * p_last / 100.0
    # The stale-page story only applies once the game is actually out.
    stale = pool_frozen and case.checkpoint_ts >= case.release_ts

    medians = []  # candidate median logits
    median_weights = []
    fit_sd = 0.0

    # --- trend model: logit rate against log-time -------------------------
    distinct_rates = {r for _, _, _, r in series}
    if len(series) >= 3 and len(distinct_rates) >= 2:
        xs = [s[0] for s in series]
        ys = [logit(s[3]) for s in series]
        ws = [0.5 ** ((t_last - s[1]) / 3600.0 / WEIGHT_HALF_LIFE_H) for s in series]
        a, b, resid_sd, sxx, xbar = wls(xs, ys, ws)
        # Shrink the slope by its own noise: var(b) ~ resid^2 / sxx.
        if sxx > 1e-9:
            var_b = resid_sd ** 2 / sxx
            shrink = (b * b) / (b * b + var_b) if (b * b + var_b) > 0 else 0.0
        else:
            shrink = 0.0
        b = max(-2.5, min(2.5, b * shrink))
        move = b * (x_t - xs[-1])
        move = max(-3.0, min(3.0, move))
        medians.append(logit(r_last) + move)
        median_weights.append(1.0)
        lever = 1.0 + (x_t - xbar) ** 2 / max(sxx, 1e-9)
        fit_sd = resid_sd * math.sqrt(min(lever, 9.0))

    # --- count model: earners and pool extrapolated apart -----------------
    if stale:
        if earners_last < 0.9995 * p_last:
            # Stale page: assume the pool jumps before the target and the
            # earners grow only a little; the rate dilutes.
            grown = earners_last * 1.25
            r_med = 100.0 * grown / (p_last * STALE_POOL_GROWTH)
            medians.append(logit(r_med))
            median_weights.append(1.0)
    elif len(series) >= 2:
        # Dilution guard: fit earners and pool on this trophy's own
        # snapshots (same x points, so the two fits are consistent) and
        # divide. It only advises the median (weight below), never drives
        # it: two free extrapolations divided compound each other's error,
        # while the trend fit above already sees the net rate. Its real
        # signal is structural — a pool outgrowing the earners.
        xs = [s[0] for s in series]
        ws = [0.5 ** ((t_last - s[1]) / 3600.0 / WEIGHT_HALF_LIFE_H) for s in series]
        es = [math.log1p(s[3] * s[2] / 100.0) for s in series]
        ps = [math.log(max(s[2], 1)) for s in series]
        ae, be, _, _, _ = wls(xs, es, ws)
        ap, bp, _, _, _ = wls(xs, ps, ws)
        be = min(max(be, 0.0), 4.0)
        bp = min(max(bp, 0.0), 3.0)
        e_target = max(math.expm1(ae + be * x_t), earners_last)
        p_target = max(math.exp(ap + bp * x_t), float(p_last))
        e_target = min(e_target, p_target)
        med_b = logit(100.0 * e_target / p_target)
        # Never let the guard alone drag the median more than 2 logits
        # from the last rate; beyond that it is extrapolation noise.
        med_b = max(logit(r_last) - 2.0, min(logit(r_last) + 2.0, med_b))
        medians.append(med_b)
        median_weights.append(0.3)

    if medians:
        med = sum(m * w for m, w in zip(medians, median_weights)) / sum(median_weights)
        disagree = (max(medians) - min(medians)) / 2.0 if len(medians) > 1 else 0.0
    else:
        med = logit(r_last)
        disagree = 0.0

    # --- spread ------------------------------------------------------------
    var = (DIFFUSION * math.sqrt(case.horizon_days)) ** 2
    var += fit_sd ** 2
    var += disagree ** 2
    var += 1.0 / max(earners_last, 1.0) + 1.0 / max(p_last - earners_last, 1.0)
    if stale:
        var += 0.9 ** 2
    sigma = min(max(math.sqrt(var), 0.25), 2.2)
    sigma_down = sigma * 1.25 if stale else sigma
    sigma_up = sigma * 0.9 if stale else sigma

    qs = []
    for z in Z:
        s = sigma_down if z < 0 else sigma_up
        qs.append(round4(min(100.0, max(0.0, inv_logit(med + z * s)))))
    # Enforce non-decreasing.
    for i in range(1, len(qs)):
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    return qs


def forecast_case(case: dict) -> dict:
    c = Case(case)
    _, pool_frozen = forecast_pool(c)
    rows = []
    for i, trophy in enumerate(c.trophies):
        rows.append({"id": trophy["id"], "q": forecast_trophy(c, i, pool_frozen)})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
