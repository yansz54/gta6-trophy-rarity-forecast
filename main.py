"""GTA VI trophy rarity forecaster (count-based convolution model).

Forecasts, for each of GTA VI's 55 PlayStation trophies, the percent of
Exophase-tracked players who will hold it at the case target, as five
quantiles (0.05, 0.25, 0.5, 0.75, 0.95). Scoring is mean pinball loss on
the logit scale; a prize needs a loss strictly below the best house entry.

Model (see METHOD.md for the full description):
  rate(T) = 100 * C(T) / N(T), modelling the two sides of the ratio
  separately instead of extrapolating the rate itself.
  - N(t): log N fitted linearly in log(1 + hours since release) over the
    case history's distinct reads (consecutive identical reads collapsed).
    Capped at 2,000,000 tracked players.
  - C(t): a pre-release cohort plus a convolution of the join rate with a
    per-trophy earn curve F (a lognormal in hours actually played, averaged
    over a lognormal distribution of play rates, median 2.2 h/day,
    sigma 0.75).
  - Each trophy's eventual share `a` and characteristic play-hours `tau`
    come from a prior table keyed to its kind, then fitted to the trophy's
    observed counts by a grid search with a prior penalty.
  - Spread: a seeded parametric bootstrap (80 draws, fixed seed); the
    N-growth parameters and each trophy's (a, tau) are resampled, the
    forecast recomputed, and the five quantiles read off the simulated
    logit rates, with a small floor spread added. A no-refresh branch is
    mixed in: with probability exp(-horizon / 5h) the answer read shows
    exactly the last observed rates.

Standard library only; no network, no wall-clock reads. Deterministic:
two runs on the same input are byte-identical.
"""

import json
import math
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard normal quantiles at the levels (kept for reference; the
# forecast's quantiles come from the bootstrap, not from these).
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

BOOTSTRAP_DRAWS = 80
BOOTSTRAP_SEED = 42
N_CAP = 2_000_000
# No-refresh branch: mean time between Exophase page refreshes.
REFRESH_MEAN_HOURS = 5.0
# Play-rate distribution: lognormal, median hours/day, sigma.
PLAY_RATE_MEDIAN = 2.2
PLAY_RATE_SIGMA = 0.75
# Sigma of the lognormal in play-hours needed (around tau).
EARN_SIGMA = 0.5
# Small floor spread added to the simulated logit rates (logit units).
FLOOR_SPREAD = 0.05
# Tau grid: 15 multipliers on a log grid around the prior tau.
TAU_GRID_N = 15
TAU_GRID_HALF_LOG = 1.2
# Prior penalty weight in the grid search (squared log-distance).
PRIOR_PENALTY = 2.0


# Prior table: kind -> (eventual share a, characteristic play-hours tau).
PRIORS = {
    "first": (0.999, 1.5),
    "story_ch1": (0.88, 18.0),
    "story_ch2": (0.82, 28.0),
    "story_ch3": (0.74, 45.0),
    "story_ch4": (0.66, 62.0),
    "story_final": (0.55, 75.0),
    "story_mission": (0.80, 25.0),
    "side_strand": (0.35, 55.0),
    "activity": (0.32, 40.0),
    "grind": (0.14, 110.0),
    "collectible": (0.16, 110.0),
    "money": (0.30, 65.0),
    "misc": (0.35, 40.0),
    "completion": (0.08, 220.0),
    "platinum": (0.05, 260.0),
}


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(x: float) -> float:
    return 100.0 / (1.0 + math.exp(-x))


def round4(x: float):
    """x rounded to 4 decimals, half up, as JavaScript's
    Math.round(x * 10000) / 10000 rounds it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(s: str) -> float:
    """ISO 8601 UTC timestamp -> epoch seconds (float)."""
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    from datetime import datetime
    return datetime.fromisoformat(s).timestamp()


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def classify_kind(trophy: dict) -> str:
    """Classify a trophy into a prior kind from tier, name, description."""
    tier = (trophy.get("tier") or "").lower()
    name = trophy.get("name") or ""
    desc = trophy.get("description") or ""
    d = desc.lower()
    n = name.lower()
    if tier == "platinum":
        return "platinum"
    if "chapter 1" in d or "chapter 1" in n:
        return "story_ch1"
    if "chapter 2" in d or "chapter 2" in n:
        return "story_ch2"
    if "chapter 3" in d or "chapter 3" in n:
        return "story_ch3"
    if "chapter 4" in d or "chapter 4" in n:
        return "story_ch4"
    if "final mission" in d and "for " not in d:
        return "story_final"
    if "final mission for" in d:
        return "side_strand"
    if "gone fishing" in d:
        return "first"
    if "100%" in d or "100 percent" in d or "100%" in n:
        return "completion"
    # Counted repeats are grinds; collectible sets are collectibles.
    if any(k in d for k in ("stunt", "challenge", "race", "win ", "defeat ",
                            "knock out", "headshot")):
        return "grind"
    if any(k in d for k in ("collect", "card", "spot", "vehicle",
                            "find all", "photograph")):
        return "collectible"
    if any(k in d for k in ("launder", "spend", "buy ", "money", "cash",
                            "earn $", "$")):
        return "money"
    if any(k in d for k in ("bowling", "golf", "darts", "fish", "tackle",
                            "bowl", "dance", "drink", "yoga", "meditat")):
        return "activity"
    if "stranger" in d or "strand" in d:
        return "side_strand"
    # A capitalised "Complete X." is a named story mission / heist.
    if desc.startswith("Complete ") and desc.rstrip().endswith("."):
        return "story_mission"
    return "misc"


def distinct_reads(history):
    """Collapse consecutive identical reads to one observation."""
    out = []
    prev = None
    for h in history:
        key = (h.get("players_tracked"), tuple(h.get("rates") or []))
        if key != prev:
            out.append(h)
            prev = key
    return out


def fit_n(reads, release_epoch):
    """Fit log N = alpha + beta * log(1 + hours since release) by least
    squares over the distinct reads. Returns (alpha, beta, beta_sd,
    n_distinct). With a single distinct read the slope is unidentified:
    beta falls back to the prior mean with a wide width."""
    xs = []
    ys = []
    for h in reads:
        t = parse_time(h["at"])
        hours = max(0.0, (t - release_epoch) / 3600.0)
        xs.append(math.log1p(hours))
        ys.append(math.log(max(1, h["players_tracked"])))
    n = len(xs)
    if n == 0:
        return (math.log(638.0), 1.0, 0.5, 0)
    if n == 1:
        # Slope unidentified: keep the level, wide prior width on slope.
        return (ys[0], 1.0, 0.5, 1)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return (my, 1.0, 0.5, n)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    beta = sxy / sxx
    alpha = my - beta * mx
    # Residual standard error for the slope.
    resid = [y - (alpha + beta * x) for x, y in zip(xs, ys)]
    dof = max(1, n - 2)
    s2 = sum(r * r for r in resid) / dof
    beta_se = math.sqrt(max(s2, 1e-6) / sxx) if sxx > 0 else 0.7
    beta_sd = max(beta_se, 0.15)
    return (alpha, beta, beta_sd, n)


def n_at(alpha, beta, hours):
    """Tracked players at `hours` since release (capped)."""
    val = math.exp(alpha + beta * math.log1p(max(0.0, hours)))
    return min(val, N_CAP)


def join_rate(alpha, beta, hours):
    """dN/dh at `hours` since release (players per hour)."""
    if beta <= 0:
        return 0.0
    return n_at(alpha, beta, hours) * beta / (1.0 + max(0.0, hours))


def earn_cdf(age_days, tau):
    """F(age): share of joiners who have earned by real-time age (days),
    for a trophy needing tau play-hours. The play-hours needed are
    lognormal around tau (sigma EARN_SIGMA); the play rate is lognormal
    (median PLAY_RATE_MEDIAN h/day, sigma PLAY_RATE_SIGMA). The ratio is
    lognormal, so the earn time in days is lognormal with
    mu = ln(tau / PLAY_RATE_MEDIAN), sigma = sqrt(EARN_SIGMA^2 +
    PLAY_RATE_SIGMA^2)."""
    if age_days <= 0:
        return 0.0
    sigma = math.sqrt(EARN_SIGMA ** 2 + PLAY_RATE_SIGMA ** 2)
    mu = math.log(tau / PLAY_RATE_MEDIAN)
    return norm_cdf((math.log(age_days) - mu) / sigma)


def cohort_age_for_share(share, tau):
    """The age (days) at which the earn curve reaches `share` (0..1).
    Inverts earn_cdf by bisection. Returns None if share is out of range."""
    if share <= 0.0:
        return 0.0
    if share >= 1.0:
        return 1e6
    sigma = math.sqrt(EARN_SIGMA ** 2 + PLAY_RATE_SIGMA ** 2)
    mu = math.log(tau / PLAY_RATE_MEDIAN)
    # Invert the normal CDF via the error function inverse approximation.
    # Use a bisection on age in days.
    lo, hi = 1e-9, 1e4
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if earn_cdf(mid, tau) < share:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def predict_count(trophy_idx, case, reads, alpha, beta, a, tau,
                  release_epoch, target_epoch):
    """Predicted earned count for one trophy at the target time."""
    trophies = case["trophies"]
    first = reads[0]
    t0 = parse_time(first["at"])
    n0 = first["players_tracked"]
    # Observed earned count at the first read.
    rates0 = first["rates"]
    r0 = rates0[trophy_idx] if rates0 and rates0[trophy_idx] is not None else 0.0
    c_obs = r0 / 100.0 * n0
    # Cohort: effective age at t0 chosen so the cohort's count matches
    # the observed count at t0.
    eventual = a * n0
    if eventual <= 0:
        return 0.0
    share0 = min(0.999999, max(0.0, c_obs / eventual))
    age0 = cohort_age_for_share(share0, tau)
    age_t = age0 + max(0.0, (target_epoch - t0) / 86400.0)
    c_cohort = eventual * earn_cdf(age_t, tau)
    # Joiners: convolution of the join rate with the earn curve.
    # Integrate from t0 to target in hourly steps (trapezoid rule).
    c_join = 0.0
    h0 = max(0.0, (t0 - release_epoch) / 3600.0)
    h1 = max(0.0, (target_epoch - release_epoch) / 3600.0)
    if h1 > h0:
        steps = max(1, int(math.ceil((h1 - h0) * 4)))  # 15-minute steps
        dh = (h1 - h0) / steps
        total = 0.0
        for i in range(steps + 1):
            h = h0 + i * dh
            # Age of joiners from s=h at the target, in days.
            age = max(0.0, (h1 - h) / 24.0)
            f = earn_cdf(age, tau)
            w = 0.5 if i in (0, steps) else 1.0
            total += w * join_rate(alpha, beta, h) * f
        c_join = total * dh
    return c_cohort + c_join


def fit_trophy(trophy_idx, case, reads, alpha, beta, release_epoch):
    """Fit (a, tau) for one trophy by grid search with a prior penalty.
    Returns (a, tau). The predicted count is linear in a, so a is solved
    exactly for each tau on the grid; tau is searched on a 15-point log
    grid around the prior tau."""
    kind = classify_kind(case["trophies"][trophy_idx])
    a0, tau0 = PRIORS[kind]
    # Observed counts at each distinct read.
    obs = []
    for h in reads:
        rates = h.get("rates") or []
        r = rates[trophy_idx] if trophy_idx < len(rates) and rates[trophy_idx] is not None else 0.0
        obs.append((parse_time(h["at"]), r / 100.0 * h["players_tracked"]))
    if not obs:
        return (a0, tau0)
    best = None
    # Tau grid: 15 multipliers, log-spaced, centred on the prior tau.
    for k in range(TAU_GRID_N):
        frac = (k / (TAU_GRID_N - 1.0)) * 2.0 - 1.0  # -1 .. +1
        tau = tau0 * math.exp(frac * TAU_GRID_HALF_LOG)
        # For this tau, the predicted count is linear in a:
        # C(t) = a * g(t) where g(t) is the count per unit a. Solve a by
        # least squares on the observed counts (with the cohort anchored
        # to the first read, the shape is set by tau; the level by a).
        # We compute g(t) = predict_count with a=1 (and cohort share
        # capped), then a = sum(obs*g)/sum(g*g), penalised toward a0.
        gs = []
        for (t_i, c_i) in obs:
            # Count per unit a at read time t_i (cohort + joiners).
            g = _count_per_unit_a(trophy_idx, case, reads, alpha, beta,
                                  tau, release_epoch, t_i)
            gs.append(g)
        num = sum(c * g for (_, c), g in zip(obs, gs))
        den = sum(g * g for g in gs)
        if den <= 0:
            a_fit = a0
        else:
            a_fit = num / den
        # Prior penalty (in log space for a, log space for tau).
        a_fit = min(0.999, max(0.001, a_fit))
        err = 0.0
        for (_, c_i), g in zip(obs, gs):
            err += (a_fit * g - c_i) ** 2
        # Scale the data error by the count magnitude so the penalty is
        # comparable across trophies.
        scale = max(1.0, sum(c for _, c in obs) / max(1, len(obs)))
        err = err / (scale * scale)
        pen = PRIOR_PENALTY * ((math.log(a_fit / a0)) ** 2 +
                               (math.log(tau / tau0)) ** 2)
        total = err + pen
        if best is None or total < best[0]:
            best = (total, a_fit, tau)
    return (best[1], best[2])


def _count_per_unit_a(trophy_idx, case, reads, alpha, beta, tau,
                      release_epoch, when_epoch):
    """Predicted earned count at `when_epoch` per unit of eventual share
    a (i.e. with a = 1), for the grid search. The cohort is anchored to
    the first read's observed count, so the shape is set by tau."""
    first = reads[0]
    t0 = parse_time(first["at"])
    n0 = first["players_tracked"]
    rates0 = first["rates"]
    r0 = rates0[trophy_idx] if rates0 and rates0[trophy_idx] is not None else 0.0
    c_obs = r0 / 100.0 * n0
    # With a = 1, eventual = n0. Cohort age at t0 for share c_obs / n0.
    share0 = min(0.999999, max(0.0, c_obs / n0)) if n0 > 0 else 0.0
    age0 = cohort_age_for_share(share0, tau)
    age_w = age0 + max(0.0, (when_epoch - t0) / 86400.0)
    c_cohort = n0 * earn_cdf(age_w, tau)
    c_join = 0.0
    h0 = max(0.0, (t0 - release_epoch) / 3600.0)
    h1 = max(0.0, (when_epoch - release_epoch) / 3600.0)
    if h1 > h0:
        steps = max(1, int(math.ceil((h1 - h0) * 4)))
        dh = (h1 - h0) / steps
        total = 0.0
        for i in range(steps + 1):
            h = h0 + i * dh
            age = max(0.0, (h1 - h) / 24.0)
            f = earn_cdf(age, tau)
            w = 0.5 if i in (0, steps) else 1.0
            total += w * join_rate(alpha, beta, h) * f
        c_join = total * dh
    return c_cohort + c_join


def forecast_case(case):
    """Forecast one case. Returns {"trophies": [{"id", "q": [...]}, ...]}."""
    trophies = case["trophies"]
    history = case.get("history") or []
    if not history:
        raise ValueError("case has no history")
    release_epoch = parse_time(case["game"]["release_date"] + "T00:00:00Z")
    target_epoch = parse_time(case["target"])
    horizon_hours = case["horizon_minutes"] / 60.0
    reads = distinct_reads(history)
    alpha, beta, beta_sd, _ = fit_n(reads, release_epoch)
    last = history[-1]
    last_rates = last["rates"] or []
    n_target = n_at(alpha, beta,
                    max(0.0, (target_epoch - release_epoch) / 3600.0))
    p_norefresh = math.exp(-horizon_hours / REFRESH_MEAN_HOURS)
    rng = random.Random(BOOTSTRAP_SEED)
    rows = []
    for idx, trophy in enumerate(trophies):
        r_last = last_rates[idx] if idx < len(last_rates) and last_rates[idx] is not None else 0.0
        a, tau = fit_trophy(idx, case, reads, alpha, beta, release_epoch)
        # Point forecast (median of the model, before the bootstrap).
        c_pt = predict_count(idx, case, reads, alpha, beta, a, tau,
                             release_epoch, target_epoch)
        rate_pt = 100.0 * c_pt / n_target if n_target > 0 else r_last
        # Bootstrap: 80 draws.
        draws = []
        for _ in range(BOOTSTRAP_DRAWS):
            # Resample the N slope (wide when unidentified) and (a, tau).
            b_s = rng.gauss(beta, beta_sd)
            a_s = a * math.exp(rng.gauss(0.0, 0.20))
            a_s = min(0.999, max(0.001, a_s))
            tau_s = tau * math.exp(rng.gauss(0.0, 0.20))
            n_s = n_at(alpha, b_s,
                       max(0.0, (target_epoch - release_epoch) / 3600.0))
            c_s = predict_count(idx, case, reads, alpha, b_s, a_s, tau_s,
                                release_epoch, target_epoch)
            rate_s = 100.0 * c_s / n_s if n_s > 0 else r_last
            # Small floor spread on the logit scale.
            lg = logit(rate_s) + rng.gauss(0.0, FLOOR_SPREAD)
            rate_s = inv_logit(lg)
            # No-refresh branch: with probability p the answer read shows
            # exactly the last observed rates.
            if rng.random() < p_norefresh:
                rate_s = r_last
            draws.append(rate_s)
        draws.sort()
        qs = []
        for level in LEVELS:
            j = min(int(level * BOOTSTRAP_DRAWS), BOOTSTRAP_DRAWS - 1)
            qs.append(round4(clamp(draws[j])))
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
