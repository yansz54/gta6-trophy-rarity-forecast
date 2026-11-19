#!/usr/bin/env python3
"""GTA VI trophy rarity forecaster (v3) — make.fun gta6-trophy-rarity-forecast-01.

Model (see METHOD.md for the full write-up, every seed and every draw):
  rate(t) = 100 * earned(t) / tracked(t), the two sides modelled separately.
  * tracked(t): saturating logistic in x = ln(1 + hours since release),
    N(x) = Nsat / (1 + exp(-k (x - x0))); x0 anchored to the first distinct
    read; (k, Nsat) fitted to distinct reads where the history has signal,
    otherwise from priors (k ~ 1.0, Nsat ~ 750k).
  * earned(t): a pre-release cohort plus a convolution of the join rate with
    a per-trophy earn curve — a lognormal in hours actually played, averaged
    over a lognormal play-rate distribution (median 2.5 h/day, sigma 0.75).
    Each trophy's eventual share s and characteristic play-hours h are
    fitted by grid search to its observed counts, with a prior penalty
    pulling them to the kind prior table.
  * refresh: Exophase refreshes on its own schedule (a renewal process, NOT
    memoryless). The refresh interval is lognormal (mean 3.8 h, sigma_log
    0.15); the chance of no refresh before the target is the survival
    ratio S(staleness + horizon) / S(staleness). Every bootstrap draw
    samples refresh times; a draw whose last refresh before the target is
    none reports the stale last-observed rates, otherwise the post-refresh
    (diluted) rates from the counts. So the median is the diluted rate
    whenever a refresh is the likely outcome.
  * spread: a bootstrap (120 draws) resampling growth, refresh timing and
    each trophy's (s, h); the five quantiles are read off the simulated
    rates, with a small logit floor spread added.
  * saturated trophies (observed rate >= 99.9%, i.e. the opening mission):
    forecast a tight band near 100% — the first mission is completed in
    minutes and every tracked player has played it.

Standard library only, no network, no wall-clock reads. All randomness
comes from random.Random with fixed integer seeds (see METHOD.md), so two
runs on the same case are byte-identical.
"""

import json
import math
import os
import random
import sys
from bisect import bisect_right
from datetime import datetime, timezone
from statistics import NormalDist

_ND = NormalDist()

# ---------------------------------------------------------------- constants
SEED = 20261119
BOOTSTRAP_DRAWS = 120

PLAY_RATE_MEDIAN = 2.2      # hours of play per day, median (brief sec. 3)
PLAY_RATE_SIGMA = 0.75      # lognormal sigma of the play-rate distribution
EARN_SIGMA = 0.6

REFRESH_MEAN_H = 3.8        # mean refresh interval, hours (lognormal)
REFRESH_SIGMA = 0.15        # lognormal sigma of the refresh interval

GROWTH_K_PRIOR = 1.0        # prior median of the logistic growth rate k
GROWTH_K_SIGMA = 0.12
NSAT_PRIOR = 750000.0       # prior median of the saturation level
NSAT_SIGMA = 0.35
GROWTH_K_FIT_SIGMA = 0.15   # resampling log-sigma of k around a fit
NSAT_FIT_SIGMA = 0.25       # resampling log-sigma of Nsat around a fit

TROPHY_S_SIGMA = 0.18
TROPHY_H_SIGMA = 0.22
FIT_PRIOR_SIGMA = 0.6       # log-sigma of the (s, h) prior penalty in the fit

FLOOR_SD = 0.06
SATURATED_MIN_RATE = 99.9   # observed rate at/above this => saturated rule
SATURATED_Q = [99.7, 99.9, 99.95, 99.99, 99.99]

Z_TAU = [-1.6448536269514722, -0.6744897501960817, 0.0,
         0.6744897501960817, 1.6448536269514722]

# Prior table: trophy kind -> (eventual share, characteristic play-hours).
# Fitted in v2 from GTA VI story data + GTA V mature rates + launch reads;
# re-checked against the practice data (see METHOD.md). The opening
# mission's hours were cut from 1.5 to 0.12: launch-night data shows new
# joiners earn essentially only the opening mission in their first hours,
# so its characteristic time must be minutes, not hours.
PRIOR = {
    "opening":    (0.999, 0.12),
    "chapter1":   (0.88, 18.0),
    "chapter2":   (0.82, 28.0),
    "chapter3":   (0.74, 45.0),
    "chapter4":   (0.66, 62.0),
    "final":      (0.55, 75.0),
    "story":      (0.80, 25.0),
    "strand":     (0.35, 55.0),
    "oneoff":     (0.32, 40.0),
    "grind":      (0.14, 110.0),
    "collectible": (0.16, 110.0),
    "money":      (0.30, 65.0),
    "misc":       (0.35, 40.0),
    "completion": (0.08, 220.0),
    "platinum":   (0.05, 260.0),
}

# ---------------------------------------------------------------- utilities
def parse_time(s):
    """ISO 8601 UTC string -> aware datetime."""
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_between(a, b):
    return (b - a).total_seconds() / 3600.0


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p):
    """Standard normal quantile (statistics.NormalDist, deterministic)."""
    if p <= 0.0:
        return -10.0
    if p >= 1.0:
        return 10.0
    return _ND.inv_cdf(p)


def lognorm_cdf(x, mean, sigma):
    """CDF of a lognormal with the given mean and log-sigma."""
    if x <= 0.0:
        return 0.0
    mu = math.log(mean) - 0.5 * sigma * sigma
    return norm_cdf((math.log(x) - mu) / sigma)


def lognorm_sf(x, mean, sigma):
    return 1.0 - lognorm_cdf(x, mean, sigma)


def lognorm_ppf(p, mean, sigma):
    """Quantile of a lognormal with the given mean and log-sigma."""
    mu = math.log(mean) - 0.5 * sigma * sigma
    return math.exp(mu + sigma * norm_ppf(p))


def logit(p):
    p = min(max(p, 0.0001), 0.9999)
    return math.log(p / (1.0 - p))


def inv_logit(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def clamp_rate(r):
    return min(max(r, 0.01), 99.99)


# ------------------------------------------------- earn curve (g grid)
# g(age_hours, h) = average earn probability for a player of wall-clock
# age `age_hours`, whose characteristic play-hours are h: the lognormal
# earn curve in play-hours, averaged over the play-rate distribution.
# Precomputed on a (log age x log h) grid with a fixed 9-point quadrature
# over play-rates; bilinear interpolation in (ln age, ln h).

def _build_g_grid():
    ages = []
    a = 0.05
    while a <= 5000.0:
        ages.append(a)
        a *= 1.16
    hs = []
    h = 0.1
    while h <= 500.0:
        hs.append(h)
        h *= 1.22
    # 9-point play-rate quadrature: quantiles (i+0.5)/9 of the lognormal.
    rates = []
    for i in range(9):
        z = norm_ppf((i + 0.5) / 9.0)
        rates.append(math.exp(math.log(PLAY_RATE_MEDIAN) + PLAY_RATE_SIGMA * z))
    grid = []
    for age in ages:
        row = []
        for h in hs:
            tot = 0.0
            for r in rates:
                play_h = r * age / 24.0
                if play_h <= 0.0:
                    continue
                tot += norm_cdf((math.log(play_h) - math.log(h)) / EARN_SIGMA)
            row.append(tot / len(rates))
        grid.append(row)
    return ages, hs, grid


AGES, HS, GGRID = _build_g_grid()
LOG_AGES = [math.log(a) for a in AGES]
LOG_HS = [math.log(h) for h in HS]


def g_interp(age, h):
    """Bilinear interpolation of g on the (ln age, ln h) grid."""
    if age <= 0.0:
        return 0.0
    if age >= AGES[-1]:
        age = AGES[-1]
    if h <= HS[0]:
        h = HS[0]
    if h >= HS[-1]:
        h = HS[-1]
    la, lh = math.log(age), math.log(h)
    ia = bisect_right(LOG_AGES, la) - 1
    if ia < 0:
        ia = 0
    elif ia > len(LOG_AGES) - 2:
        ia = len(LOG_AGES) - 2
    ih = bisect_right(LOG_HS, lh) - 1
    if ih < 0:
        ih = 0
    elif ih > len(LOG_HS) - 2:
        ih = len(LOG_HS) - 2
    a0, a1 = LOG_AGES[ia], LOG_AGES[ia + 1]
    h0, h1 = LOG_HS[ih], LOG_HS[ih + 1]
    fa = 0.0 if a1 == a0 else (la - a0) / (a1 - a0)
    fh = 0.0 if h1 == h0 else (lh - h0) / (h1 - h0)
    g00 = GGRID[ia][ih]
    g01 = GGRID[ia][ih + 1]
    g10 = GGRID[ia + 1][ih]
    g11 = GGRID[ia + 1][ih + 1]
    return (g00 * (1 - fa) * (1 - fh) + g01 * (1 - fa) * fh
            + g10 * fa * (1 - fh) + g11 * fa * fh)


def invert_g(target, h):
    """Wall-clock age at which g(age, h) == target (bisection; target in [0,1])."""
    if target <= 0.0:
        return 0.0
    if target >= 1.0:
        return AGES[-1]
    lo, hi = 0.0, AGES[-1]
    if g_interp(hi, h) < target:
        return hi
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if g_interp(mid, h) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ------------------------------------------------------- trophy kinds
def classify(trophy):
    name = (trophy.get("name") or "").lower()
    desc = (trophy.get("description") or "").lower()
    tier = trophy.get("tier") or ""
    if tier == "platinum":
        return "platinum"
    if "gone fishing" in desc:
        return "opening"
    if "complete chapter 1" in desc:
        return "chapter1"
    if "complete chapter 2" in desc:
        return "chapter2"
    if "complete chapter 3" in desc:
        return "chapter3"
    if "complete chapter 4" in desc:
        return "chapter4"
    if desc.strip() == "complete the final mission.":
        return "final"
    if "complete the final mission for" in desc:
        return "strand"
    if "attain 100% completion" in desc:
        return "completion"
    if "mission strands" in desc:
        return "strand"
    if "earn $" in desc or "spend $" in desc:
        return "money"
    # Counted repeats (grinds): "complete/win/rob/discover N ..." with N >= 5.
    import re
    m = re.search(r"\b(complete|win|rob|discover|register|waank it)\s+(\d+)\b", desc)
    if m and int(m.group(2)) >= 5:
        return "grind"
    # Set completion (collectibles): "all", "every", "each", "sets".
    if ("all " in desc or "every " in desc or "each " in desc
            or "sets" in desc or "set." in desc):
        return "collectible"
    # A capitalised "Complete <Name>." is a named story mission.
    if desc.startswith("complete ") and desc.rstrip(".")[9:10].isupper():
        return "story"
    # Specific one-off feats.
    if (desc.startswith(("try ", "beat ", "win ", "rob ", "complete ",
                         "enjoy ", "smash ", "follow-through", "bowl ",
                         "meditate ", "hunt ", "lose ", "take down ",
                         "make ", "discover ", "explore ", "attain "))
            or "kinda hot" in desc):
        return "oneoff"
    return "misc"


# ------------------------------------------------------- growth curve
def growth_x(t_rel_h):
    """x = ln(1 + hours since release)."""
    return math.log(max(1.0 + t_rel_h, 1e-6))


def growth_n(x, k, nsat, x0):
    return nsat / (1.0 + math.exp(-k * (x - x0)))


def anchor_x0(x_first, n_first, k, nsat):
    """x0 so that the curve passes through (x_first, n_first)."""
    ratio = nsat / max(n_first, 1.0) - 1.0
    if ratio <= 0.0:
        ratio = 1e-9
    return x_first + math.log(ratio) / k


def fit_growth(distinct, release_dt):
    """Return (k, nsat, x0, mode). Fits (k, nsat) to distinct reads when the
    history has signal (>= 2 distinct reads with different N); otherwise
    the prior medians. x0 is always anchored to the first distinct read."""
    t_first, n_first = distinct[0][0], distinct[0][1]
    x_first = growth_x(hours_between(release_dt, t_first))
    varied = len({d[1] for d in distinct}) > 1 and len(distinct) >= 2
    if not varied:
        k, nsat = GROWTH_K_PRIOR, NSAT_PRIOR
        return k, nsat, anchor_x0(x_first, n_first, k, nsat), "prior"
    ks = [math.exp(math.log(0.4) + i * (math.log(2.5) - math.log(0.4)) / 15.0)
          for i in range(16)]
    nsats = [math.exp(math.log(100000.0)
                      + i * (math.log(3000000.0) - math.log(100000.0)) / 15.0)
             for i in range(16)]
    best = None
    for k in ks:
        for nsat in nsats:
            if nsat <= max(d[1] for d in distinct):
                continue
            x0 = anchor_x0(x_first, n_first, k, nsat)
            sse = 0.0
            for (t, n, _e) in distinct:
                x = growth_x(hours_between(release_dt, t))
                pred = growth_n(x, k, nsat, x0)
                sse += (math.log(max(pred, 1.0)) - math.log(max(n, 1.0))) ** 2
            if best is None or sse < best[0]:
                best = (sse, k, nsat, x0)
    _sse, k, nsat, x0 = best
    return k, nsat, x0, "fit"


# ------------------------------------------------------- earned counts
def make_bins(t_start, t_end, n_curve, max_bins=110):
    """Time bins from t_start to t_end for the join convolution.
    Returns [(t_mid, dN)] with at most max_bins bins; dN from the N curve."""
    span_h = hours_between(t_start, t_end)
    if span_h <= 0:
        return []
    nbin = int(min(max_bins, max(1, math.ceil(span_h))))
    step = span_h / nbin
    bins = []
    prev_t = t_start
    prev_n = n_curve(prev_t)
    for i in range(1, nbin + 1):
        t = t_start
        # advance by step hours from t_start
        from datetime import timedelta
        t = t_start + timedelta(hours=step * i)
        n = n_curve(t)
        dn = max(n - prev_n, 0.0)
        if dn > 0:
            mid = t_start + timedelta(hours=step * (i - 0.5))
            bins.append((mid, dn))
        prev_t, prev_n = t, n
    return bins


def earned_at(t_eval, s, h, a0, t_first, n0, bins, n_curve):
    """Predicted earned count at t_eval for one trophy.
    a0 = effective wall-clock age of the pre-release cohort at t_first
    (solved so the cohort reproduces the first observed count)."""
    age0 = a0 + hours_between(t_first, t_eval)
    e = n0 * s * g_interp(age0, h)
    for (t_mid, dn) in bins:
        age = hours_between(t_mid, t_eval)
        if age <= 0:
            continue
        e += dn * s * g_interp(age, h)
    return e


def fit_trophy(trophy, distinct, release_dt, growth):
    """Grid-search (s, h) for one trophy: match its observed earned counts
    at the distinct reads, with a prior penalty to the kind prior.
    Returns (s, h)."""
    kind = classify(trophy)
    s0, h0 = PRIOR[kind]
    t_first, n0, rates0 = distinct[0]
    idx = trophy["_idx"]
    e0 = rates0[idx] / 100.0 * n0 if rates0[idx] is not None else 0.0
    k, nsat, x0 = growth
    n_curve = lambda t: growth_n(growth_x(hours_between(release_dt, t)),
                                 k, nsat, x0)
    # Coarse bins for the fit (rebuilt per distinct-read time below).
    s_vals = [math.exp(math.log(0.02) + i * (math.log(0.999) - math.log(0.02)) / 12.0)
              for i in range(13)]
    h_vals = [math.exp(math.log(0.5) + i * (math.log(300.0) - math.log(0.5)) / 12.0)
              for i in range(13)]
    # Precompute per-read bin structures once (shared across (s, h)).
    # With a single distinct read the error term is zero for every solvable
    # (s, h) (a0 is solved to reproduce the one observed count), so the fit
    # is just the prior penalty among solvable grid points — no inversion
    # or convolution needed.
    if len(distinct) == 1:
        best = None
        # The prior point itself is always a candidate: the log-spaced
        # grid is coarse near s = 1, and with one read the data cannot move
        # the fit, so the exact prior must be reachable.
        for s, h in [(s0, h0)] + [(s, h) for s in s_vals for h in h_vals]:
            if e0 > 0 and s <= e0 / max(n0, 1.0):
                continue
            if e0 > 0 and s * g_interp(AGES[-1], h) < e0 / max(n0, 1.0):
                continue
            pen = (((math.log(s) - math.log(s0)) / FIT_PRIOR_SIGMA) ** 2
                   + ((math.log(h) - math.log(h0)) / FIT_PRIOR_SIGMA) ** 2)
            if best is None or pen < best[0]:
                best = (pen, s, h)
        if best is None:
            return s0, h0
        return best[1], best[2]
    read_bins = []
    for (t, n, rates) in distinct:
        read_bins.append(make_bins(t_first, t, n_curve, max_bins=30))
    best = None
    for s, h in [(s0, h0)] + [(s, h) for s in s_vals for h in h_vals]:
        if e0 > 0 and s <= e0 / max(n0, 1.0):
            continue
        if e0 > 0:
            a0 = invert_g(e0 / max(n0, 1.0) / s, h)
        else:
            a0 = 0.0
        err = 0.0
        for ri, (t, n, rates) in enumerate(distinct):
            e_obs = rates[idx] / 100.0 * n if rates[idx] is not None else 0.0
            e_pred = earned_at(t, s, h, a0, t_first, n0,
                               read_bins[ri], n_curve)
            err += ((e_pred - e_obs) ** 2) / max(1.0, e_obs)
        pen = (((math.log(s) - math.log(s0)) / FIT_PRIOR_SIGMA) ** 2
               + ((math.log(h) - math.log(h0)) / FIT_PRIOR_SIGMA) ** 2)
        tot = err + pen
        if best is None or tot < best[0]:
            best = (tot, s, h)
    if best is None:
        return s0, h0
    return best[1], best[2]


# ------------------------------------------------------- refresh model
def sample_refresh_times(rng, staleness_h, horizon_h):
    """Sample the refresh times (hours after the checkpoint) of the renewal
    process up to the horizon. Returns the list of refresh offsets; an
    empty list means no refresh before the target. The first interval is
    the residual life given the page's age (staleness); later intervals
    are fresh draws from the interval distribution."""
    # Residual life: S(A + R) = U * S(A), solved by bisection.
    s_a = lognorm_sf(staleness_h, REFRESH_MEAN_H, REFRESH_SIGMA)
    u = rng.random()
    target_sf = u * s_a
    lo, hi = 0.0, 500.0
    for _ in range(35):
        mid = 0.5 * (lo + hi)
        if lognorm_sf(staleness_h + mid, REFRESH_MEAN_H, REFRESH_SIGMA) > target_sf:
            lo = mid
        else:
            hi = mid
    r = 0.5 * (lo + hi)
    times = []
    t = r
    while t <= horizon_h:
        times.append(t)
        # Fresh interval draw.
        u2 = rng.random()
        iv = lognorm_ppf(min(max(u2, 1e-9), 1 - 1e-9),
                         REFRESH_MEAN_H, REFRESH_SIGMA)
        t += max(iv, 0.05)
    return times


# ------------------------------------------------------- forecast a case
def forecast_case(case):
    from datetime import timedelta
    release_dt = parse_time(case["game"]["release_date"] + "T00:00:00Z")
    checkpoint = parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_h = hours_between(checkpoint, target)
    trophies = case["trophies"]
    for i, tr in enumerate(trophies):
        tr["_idx"] = i
    history = case["history"]

    # Distinct reads: consecutive identical (players, rates) collapse to
    # one observation, kept at its first time.
    distinct = []
    prev_key = None
    for rd in history:
        key = (rd["players_tracked"], tuple(rd["rates"]))
        if key != prev_key:
            distinct.append((parse_time(rd["at"]), rd["players_tracked"],
                             rd["rates"]))
            prev_key = key
    if not distinct:
        # No history at all: fall back to a degenerate single read at the
        # checkpoint (never exercised on real cases).
        distinct = [(checkpoint, 1, [0.01] * len(trophies))]

    t_first, n0, rates_first = distinct[0]
    t_last_distinct = distinct[-1][0]
    staleness_h = max(hours_between(t_last_distinct, checkpoint), 0.0)
    last_rates = distinct[-1][2]

    # Growth fit (point estimates) and per-trophy (s, h) fits.
    k_pt, nsat_pt, x0_pt, growth_mode = fit_growth(distinct, release_dt)
    growth_pt = (k_pt, nsat_pt, x0_pt)
    fitted = []
    for tr in trophies:
        fitted.append(fit_trophy(tr, distinct, release_dt, growth_pt))

    # Separate RNG streams (fixed seeds) so the draw order is unambiguous.
    rng_g = random.Random(SEED)
    rng_r = random.Random(SEED + 1)
    rng_t = random.Random(SEED + 2)

    k_sig = GROWTH_K_FIT_SIGMA if growth_mode == "fit" else GROWTH_K_SIGMA
    nsat_sig = NSAT_FIT_SIGMA if growth_mode == "fit" else NSAT_SIGMA

    draws = [[] for _ in trophies]  # draws[i] = list of rate draws
    for _b in range(BOOTSTRAP_DRAWS):
        # Growth draw.
        k_b = math.exp(math.log(k_pt) + k_sig * norm_ppf(rng_g.random()))
        nsat_b = math.exp(math.log(nsat_pt) + nsat_sig * norm_ppf(rng_g.random()))
        nsat_b = max(nsat_b, max(d[1] for d in distinct) * 1.5)
        x0_b = anchor_x0(growth_x(hours_between(release_dt, t_first)),
                         n0, k_b, nsat_b)
        n_curve_b = lambda t, _k=k_b, _n=nsat_b, _x=x0_b: growth_n(
            growth_x(hours_between(release_dt, t)), _k, _n, _x)
        # Refresh draw.
        refresh_offsets = sample_refresh_times(rng_r, staleness_h, horizon_h)
        if refresh_offsets:
            t_last_refresh = checkpoint + timedelta(hours=refresh_offsets[-1])
            bins_b = make_bins(t_first, t_last_refresh, n_curve_b)
            n_at_refresh = n_curve_b(t_last_refresh)
        else:
            t_last_refresh = None
            bins_b = []
            n_at_refresh = None
        # Trophy draws (id order: s first, then h, per trophy).
        for i, tr in enumerate(trophies):
            s_fit, h_fit = fitted[i]
            s_b = math.exp(math.log(s_fit) + TROPHY_S_SIGMA * norm_ppf(rng_t.random()))
            h_b = math.exp(math.log(h_fit) + TROPHY_H_SIGMA * norm_ppf(rng_t.random()))
            s_b = min(max(s_b, 0.005), 0.999)
            h_b = min(max(h_b, 0.1), 500.0)
            if t_last_refresh is None:
                # No refresh: the page still shows the last observed rates.
                r = last_rates[i] if last_rates[i] is not None else 0.01
                draws[i].append(float(r))
                continue
            e0 = rates_first[i] / 100.0 * n0 if rates_first[i] is not None else 0.0
            if e0 > 0 and s_b > e0 / max(n0, 1.0):
                a0_b = invert_g(e0 / max(n0, 1.0) / s_b, h_b)
            else:
                a0_b = 0.0
            e_b = earned_at(t_last_refresh, s_b, h_b, a0_b, t_first, n0,
                            bins_b, n_curve_b)
            rate_b = 100.0 * e_b / max(n_at_refresh, 1.0)
            draws[i].append(rate_b)

    rows = []
    for i, tr in enumerate(trophies):
        if last_rates[i] is not None and last_rates[i] >= SATURATED_MIN_RATE:
            rows.append({"id": tr["id"], "q": list(SATURATED_Q)})
            continue
        ds = sorted(draws[i])
        n = len(ds)
        qs = []
        for tau in (0.05, 0.25, 0.5, 0.75, 0.95):
            pos = tau * (n - 1)
            lo_i = int(math.floor(pos))
            hi_i = min(lo_i + 1, n - 1)
            frac = pos - lo_i
            qs.append(ds[lo_i] * (1 - frac) + ds[hi_i] * frac)
        # Small logit floor spread so no band is degenerate.
        # (logit takes a fraction, so the percent rate is divided by 100.)
        out = []
        for q, z in zip(qs, Z_TAU):
            out.append(clamp_rate(100.0 * inv_logit(logit(q / 100.0) + z * FLOOR_SD)))
        # Enforce non-decreasing.
        for j in range(1, 5):
            if out[j] < out[j - 1]:
                out[j] = out[j - 1]
        rows.append({"id": tr["id"],
                     "q": [round(v, 4) for v in out]})
    return {"trophies": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(f for f in os.listdir(in_dir) if f.endswith(".in"))
    for name in names:
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        fc = forecast_case(case)
        out_path = os.path.join(out_dir, case["id"] + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(fc, fh, separators=(",", ":"))
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
