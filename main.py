"""GTA VI trophy rarity forecaster — count-based convolution model.

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, the
percent of Exophase-tracked players who will have earned it at the case's
target, as five quantiles (0.05, 0.25, 0.5, 0.75, 0.95).

Why counts, not rates.  An earn rate is a ratio: earned players over
tracked players.  At a launch both move, and they move for different
reasons.  Tracked players jump whenever Exophase refreshes the page (the
house saw 638 tracked players from launch night until 04:30Z, then 3,868:
every rate but the one at 100% fell in that one refresh, because the new
players had earned almost nothing yet).  Earned counts only ever grow, as
players work through the story.  A forecaster that extrapolates the rate
itself — the house baselines do — mistakes dilution for players quitting,
and mistakes a flat page between refreshes for a flat world.  This program
models the two sides separately and divides at the end:

  * N(t), tracked players: a power-law fit of log N on log(1 + hours since
    release), fitted to the case history's distinct reads, capped by a prior
    ceiling.  The join rate n(t) = dN/dt is its derivative.
  * C_i(t), players holding trophy i: two parts.  The cohort already
    tracked at the first read (reviewers and early players, who had played
    before release) starts from its observed count and grows as its
    non-holders keep playing, asymptoting at the cohort's eventual share.
    Everyone who joins after release is a convolution of the join rate with
    a per-trophy earn curve F_i: a player who joins at time s has had
    (t - s) to play, and F_i is the share of joiners who have earned trophy
    i by then — a lognormal curve in hours actually played (a trophy needs
    a minimum amount of real play: nobody finishes a 22-mission chapter in
    two hours), averaged over a lognormal distribution of play rates (hours
    played per day).  Each trophy's eventual share a_i and characteristic
    play-hours tau_i come from a prior table keyed to the trophy's kind
    (chapter, side strand, collectible, grind, money, activity, misc, 100%,
    platinum), then fitted to the trophy's observed counts in the case
    history by a grid search with a prior penalty.  The predicted count is
    linear in a, so the search is fine in tau and exact in a.
  * rate_i(T) = 100 * C_i(T) / N(T).

Spread.  The quantiles come from a parametric bootstrap (seeded, so two
runs are byte-identical): the N-growth parameters and each trophy's
(a, tau) are resampled, the forecast is recomputed, and the five quantiles
are read off the simulated logit rates, with a small floor spread added.
On short horizons the dominant risk is a page refresh the house has not
shown yet, so the bootstrap mixes in a no-refresh branch — with
probability exp(-horizon / 5h) the answer read shows exactly the last
observed rates — which is what makes the lower quantiles reach down to
where the rates land after a refresh.

Standard library only.  No network, no randomness outside the seeded
generator, no wall-clock reads.  Runs in well under the 8-second cap.
"""

import json
import math
import os
import random
import re
import sys
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]

# Release day began 2026-11-19T00:00:00Z (game released 19 Nov 2026).
# The case's game.release_date is authoritative when present; this constant
# is only the fallback.  (An earlier draft had 1794681600 here, which is
# 2026-11-14 — four days early — and silently shifted every time axis.)
RELEASE_EPOCH = 1795046400
# Prior ceiling for tracked players at any target (Exophase, GTA VI week 1).
N_CAP = 2000000.0
# Bootstrap draws for the quantile spread.
N_BOOT = 80
BOOT_SEED = 20261119
# Mean interval between Exophase page refreshes (hours).  Launch night was
# flat from 23:50Z to 04:00Z and had refreshed by 04:30Z: one interval of
# roughly four and a half hours.
REFRESH_MEAN_H = 5.0

# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up (matches the starter's JS rounding)."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def parse_time(s):
    """ISO 8601 UTC (with trailing Z) to epoch seconds."""
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    from datetime import datetime

    return datetime.fromisoformat(s).timestamp()


def norm_ppf(p):
    """Standard normal quantile (Acklam's rational approximation)."""
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285984469534e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# Play-rate distribution: hours a player actually plays per day, lognormal.
# Median 2.2 h/day, sigma 0.75 — launch-week players play more than the
# year-round average, with a long tail of marathon players.  Nine fixed
# quantile points, equal weights.
RHO_POINTS = [math.exp(0.75 * norm_ppf((i + 0.5) / 9.0) + math.log(2.2))
              for i in range(9)]


# ---------------------------------------------------------------------------
# Trophy kinds and their priors.
#
# a     = eventual share of tracked players who ever earn the trophy
#         (Exophase-tracked players are self-selected enthusiasts, so these
#         run well above PSN-wide rates: GTA V's first-mission trophy sits
#         at 43% PSN-wide but ~93% among tracker users; its platinum at
#         0.1% PSN-wide but ~4-6% among tracker users).
# tau   = characteristic hours actually played to earn it (lognormal median).
# sigma = spread of the lognormal time-to-earn (hours played).
# Calibrated from GTA V (50-60 h story, 91-mission GTA VI story ~50-80 h,
# 100% ~200 h) and Red Dead Redemption 2, plus the launch-night reads.
# ---------------------------------------------------------------------------

TYPE_PRIORS = {
    "first":        {"a": 0.999, "tau": 1.5,   "s": 0.35},  # opening mission
    "story_ch1":    {"a": 0.88,  "tau": 18.0,  "s": 0.45},  # complete chapter 1
    "story_ch2":    {"a": 0.82,  "tau": 28.0,  "s": 0.45},  # complete chapter 2
    "story_ch3":    {"a": 0.74,  "tau": 45.0,  "s": 0.45},  # complete chapter 3
    "story_ch4":    {"a": 0.66,  "tau": 62.0,  "s": 0.45},  # complete chapter 4
    "story_final":  {"a": 0.55,  "tau": 75.0,  "s": 0.45},  # complete final mission
    "story_mission":{"a": 0.80,  "tau": 25.0,  "s": 0.50},  # a named story mission
    "side_strand":  {"a": 0.35,  "tau": 55.0,  "s": 0.55},  # a stranger's strand
    "activity":     {"a": 0.32,  "tau": 40.0,  "s": 0.55},  # one-off activities
    "grind":        {"a": 0.14,  "tau": 110.0, "s": 0.55},  # 250 stunts, 50 challenges
    "collectible":  {"a": 0.16,  "tau": 110.0, "s": 0.55},  # sets, spots, vehicles
    "money":        {"a": 0.30,  "tau": 65.0,  "s": 0.55},  # earn/spend/buy big
    "misc":         {"a": 0.35,  "tau": 40.0,  "s": 0.55},  # one-off moments
    "completion":   {"a": 0.08,  "tau": 220.0, "s": 0.50},  # 100% completion
    "platinum":     {"a": 0.05,  "tau": 260.0, "s": 0.50},  # the platinum
}

_CHAPTER_RE = re.compile(r"complete chapter (\d)", re.I)
# A plain story mission: "Complete Gone Fishing." — a capitalised mission
# name, not a chapter, not a final mission, not a strand.  Case-sensitive
# on purpose: "Complete an arcade game." and "Complete all 8 ... strands."
# start lowercase and are not story missions.
_MISSION_RE = re.compile(r"^Complete ([A-Z][^.]{2,})\.?$")


def classify(trophy, first_rate):
    """Kind of trophy, from its tier, name and description.

    Returns "story_mission" for a plain story mission; resolve_kinds()
    promotes the earliest such mission to "first" when no trophy is already
    "first" (the opening mission is usually caught earlier by its ~100%
    first-read rate).  The order matters: "final mission for X" is a strand,
    not the final mission, and a plain mission name must be caught before
    the activity keywords (the opening mission is "Gone Fishing").
    """
    name = trophy.get("name", "") or ""
    desc = trophy.get("description", "") or ""
    text = (name + " " + desc).lower()
    if trophy.get("tier") == "platinum":
        return "platinum"
    if "100% completion" in desc.lower() or "attain 100%" in desc.lower():
        return "completion"
    m = _CHAPTER_RE.search(desc)
    if m:
        n = int(m.group(1))
        return {1: "story_ch1", 2: "story_ch2", 3: "story_ch3", 4: "story_ch4"}.get(n, "misc")
    if "final mission for" in desc:
        return "side_strand"
    if "final mission" in desc:
        return "story_final"
    # A trophy already at ~100% in the first read is the opening beat.
    if first_rate is not None and first_rate >= 95.0:
        return "first"
    if _MISSION_RE.match(desc.strip()) and "strand" not in desc.lower():
        return "story_mission"
    # A counted repeat ("Win 20 races", "250 Stunts") is a grind, not a
    # one-off activity, even when the wording mentions races.
    if re.search(r"\b(win|complete|earn|do)\s+\d+\b", text) and \
            ("race" in text or "stunt" in text or "challenge" in text):
        return "grind"
    if "stunt" in text or "gold medal" in text or "challenges" in text:
        return "grind"
    if ("collectible card" in text or "waink" in text or "meditate" in text
            or "explore all" in text or "discover 100 vehicles" in text
            or "register 10 vehicles" in text):
        return "collectible"
    if "laundering" in text or "spend $" in text or "purchase all" in text:
        return "money"
    if ("race" in text or "fishing" in text or "arcade" in text
            or "shooting range" in text or "try each type of activity" in text):
        return "activity"
    if "mission strands" in desc:
        return "side_strand"
    return "misc"


def resolve_kinds(trophies, kinds):
    """Promote the earliest plain story mission to "first", but only when
    no trophy is already "first" (the opening mission is usually caught by
    its ~100% first-read rate, and a second "first" would be wrong: the
    other named story missions keep the "story_mission" prior).  Returns a
    new list of kinds."""
    kinds = list(kinds)
    if "first" in kinds:
        return kinds
    mission_idx = [i for i, k in enumerate(kinds) if k == "story_mission"]
    if mission_idx:
        first = min(mission_idx, key=lambda i: trophies[i]["id"])
        kinds[first] = "first"
    return kinds


# ---------------------------------------------------------------------------
# Earn curves and the convolution
# ---------------------------------------------------------------------------


def earn_share(age_days, tau_hours, sigma):
    """Share of joiners who earned the trophy age_days after joining.

    Lognormal in hours actually played, averaged over the play-rate
    distribution: F = E_rho[Phi((ln(rho * age_days) - ln(tau)) / sigma)].
    The lognormal's thin left tail is the point: a trophy that needs 18
    hours of play is essentially unearned after 2 hours, however fast the
    player, because the missions themselves take that long.
    """
    if age_days <= 0:
        return 0.0
    log_tau = math.log(tau_hours)
    total = 0.0
    for rho in RHO_POINTS:
        play = rho * age_days
        if play <= 0:
            continue
        total += norm_cdf((math.log(play) - log_tau) / sigma)
    return total / len(RHO_POINTS)


class NModel:
    """Tracked-players growth: log N linear in log(1 + hours since release)."""

    def __init__(self, points):
        # points: list of (t_hours, N), distinct reads, first-seen times.
        self.points = points
        xs = [math.log1p(max(0.0, t)) for t, _ in points]
        ys = [math.log(max(1.0, n)) for _, n in points]
        n = len(points)
        if n >= 2 and max(xs) > min(xs):
            mx = sum(xs) / n
            my = sum(ys) / n
            sxx = sum((x - mx) ** 2 for x in xs)
            sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            b1 = sxy / sxx if sxx > 0 else 0.9
            b0 = my - b1 * mx
            resid = [y - (b0 + b1 * x) for x, y in zip(xs, ys)]
            dof = max(1, n - 2)
            sig = math.sqrt(sum(r * r for r in resid) / dof)
            # Slope uncertainty from the fit, floored: a flat history says
            # nothing about growth, so keep the prior width.
            sig_b1 = math.sqrt(sig * sig / sxx) if sxx > 0 else 0.35
        else:
            b1, b0, sig, sig_b1 = 0.9, ys[0] if ys else 0.0, 0.30, 0.35
        # Clamp the slope to sane growth (never shrinking, never explosive).
        self.b1 = min(2.0, max(0.05, b1))
        self.b0 = b0
        self.sig = max(0.10, sig)
        self.sig_b1 = max(0.15, min(0.6, sig_b1))
        self.sig_b0 = max(0.10, sig)

    def n_at(self, t_hours, b0=None, b1=None):
        b0 = self.b0 if b0 is None else b0
        b1 = self.b1 if b1 is None else b1
        val = math.exp(b0 + b1 * math.log1p(max(0.0, t_hours)))
        return min(N_CAP, max(1.0, val))

    def joins_per_hour(self, t_hours, b0=None, b1=None):
        """Players joining in the hour starting at t_hours."""
        return max(0.0, self.n_at(t_hours + 1.0, b0, b1) - self.n_at(t_hours, b0, b1))


def w_at(row, age_h):
    """earn_share at a (possibly fractional) age in hours, linear interp."""
    if age_h <= 0:
        return 0.0
    if age_h >= len(row) - 1:
        return row[-1]
    lo = int(math.floor(age_h))
    frac = age_h - lo
    return row[lo] * (1.0 - frac) + row[lo + 1] * frac


# ---------------------------------------------------------------------------
# Per-case forecast
# ---------------------------------------------------------------------------

# Fine log-spaced multipliers for the (a, tau) fit.  The predicted count is
# linear in a, so a is searched exactly and tau on this grid.
TAU_MULTS = [0.05, 0.07, 0.10, 0.14, 0.20, 0.28, 0.40, 0.56, 0.80,
             1.0, 1.25, 1.6, 2.0, 2.5, 3.2]
A_MULTS = [0.4, 0.55, 0.7, 0.9, 1.0, 1.15, 1.35, 1.6, 1.9, 2.2, 2.6]


def distinct_reads(history):
    """Collapse consecutive identical reads (flat page between refreshes)."""
    out = []
    prev_key = None
    for read in history:
        key = (read.get("players_tracked"), tuple(read.get("rates") or ()))
        if key != prev_key:
            out.append(read)
            prev_key = key
    return out


def sample_times(reads, limit=10):
    """Evenly spaced subset of distinct reads, first and last included."""
    if len(reads) <= limit:
        return list(reads)
    idx = sorted(set([0, len(reads) - 1] +
                     [round(i * (len(reads) - 1) / (limit - 1)) for i in range(limit)]))
    return [reads[i] for i in idx]


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case.get("history") or []
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    n_tr = len(trophies)

    at_epoch = parse_time(case["at"])
    target_epoch = parse_time(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0
    horizon_hours = case["horizon_minutes"] / 60.0
    # Release instant: the case's game.release_date, else the constant.
    release_epoch = RELEASE_EPOCH
    gd = (case.get("game") or {}).get("release_date")
    if gd:
        release_epoch = parse_time(gd + "T00:00:00Z")
    target_h = max(0.0, (target_epoch - release_epoch) / 3600.0)
    max_age_h = int(math.ceil(target_h)) + 1

    reads = distinct_reads(history)
    # Points for the N model: first-seen time of each distinct read.
    points = []
    for r in reads:
        t_h = max(0.0, (parse_time(r["at"]) - release_epoch) / 3600.0)
        points.append((t_h, float(r["players_tracked"])))
    nmodel = NModel(points)
    last_n = float(reads[-1]["players_tracked"])

    # First-read rates, for the "opening beat" classification.
    first_rates = reads[0].get("rates") or [None] * n_tr

    # Per-trophy anchor: first read with a non-null rate.  That read's
    # population is the pre-release cohort; its count is the starting point
    # for that cohort's growth.  Also the last non-null rate (the
    # no-refresh answer on short horizons).
    anchors = []
    last_rates = []
    for i in range(n_tr):
        anchor = None
        last_rate = None
        for r, (t_h, n) in zip(reads, points):
            v = (r.get("rates") or [None] * n_tr)[i]
            if v is None:
                continue
            if anchor is None:
                anchor = (t_h, n, v * n / 100.0)
            last_rate = v
        anchors.append(anchor)
        last_rates.append(last_rate)

    # --- kinds and priors -------------------------------------------------
    kinds = resolve_kinds(
        trophies, [classify(tr, first_rates[i]) for i, tr in enumerate(trophies)])
    priors = [TYPE_PRIORS[k] for k in kinds]

    # --- per-trophy fit ---------------------------------------------------
    fit_reads = sample_times(reads, limit=10)
    fit_times = [max(0.0, (parse_time(r["at"]) - release_epoch) / 3600.0)
                 for r in fit_reads]
    obs_counts = []
    for r in fit_reads:
        n = float(r["players_tracked"])
        rates = r.get("rates") or [None] * n_tr
        obs_counts.append([None if v is None else v * n / 100.0 for v in rates])

    # W grid: earn_share at whole-hour ages, keyed by (tau, sigma).
    w_grid = {}
    for pr in priors:
        for m in TAU_MULTS:
            tau = round(pr["tau"] * m, 6)
            key = (tau, pr["s"])
            if key in w_grid:
                continue
            row = []
            for age_h in range(max_age_h + 1):
                row.append(earn_share(age_h / 24.0, tau, pr["s"]))
            w_grid[key] = row

    # Hourly joins under the base N fit, bins 0..ceil(target_h).
    n_bins = int(math.ceil(target_h)) + 1
    joins = [nmodel.joins_per_hour(float(h)) for h in range(n_bins)]

    def parts(i, t_h, tau, s):
        """(base, slope) with predicted count = base + a * slope.

        base  = cohort count if a were 0 (its observed count, decayed by
                the earn curve) — i.e. c0 * (1 - F).
        slope = cohort growth per unit a (n0 * F) + new-joiner convolution
                per unit a (sum joins * F).
        """
        anchor = anchors[i]
        row = w_grid[(tau, s)]
        if anchor is None:
            base, slope = 0.0, 0.0
        else:
            t_first, n0, c0 = anchor
            f = w_at(row, t_h - t_first) if t_h > t_first else 0.0
            base = c0 * (1.0 - f)
            slope = n0 * f
        if t_h > 0:
            upto = min(int(math.floor(t_h)), len(joins) - 1)
            g = 0.0
            for sbin in range(upto + 1):
                g += joins[sbin] * w_at(row, t_h - sbin)
            slope += g
        return base, slope

    def loglik(obs, pred):
        ll = 0.0
        for o, p in zip(obs, pred):
            if o is None:
                continue
            p = max(1e-6, p)
            ll += o * math.log(p) - p
        return ll

    best = []
    for i in range(n_tr):
        pr = priors[i]
        obs = [oc[i] for oc in obs_counts]
        best_ll = None
        best_a = pr["a"]
        best_tau = pr["tau"]
        for tm in TAU_MULTS:
            tau = round(pr["tau"] * tm, 6)
            ps = [parts(i, t, tau, pr["s"]) for t in fit_times]
            pen_tau = -0.5 * ((math.log(tm) / 0.7) ** 2)
            for am in A_MULTS:
                a = min(0.999, pr["a"] * am)
                pred = [b + a * sl for (b, sl) in ps]
                ll = loglik(obs, pred) + pen_tau - 0.5 * ((math.log(am) / 0.7) ** 2)
                if best_ll is None or ll > best_ll:
                    best_ll, best_a, best_tau = ll, a, tau
        best.append((best_a, best_tau, pr["s"]))

    # --- forecast at the target + bootstrap spread ----------------------
    rng = random.Random(BOOT_SEED)
    n_target = max(nmodel.n_at(target_h), last_n)
    floor_sd = 0.12 * math.sqrt(max(horizon_days, 0.05) / 4.52)
    floor_sd = max(0.05, floor_sd)
    p_no_refresh = math.exp(-horizon_hours / REFRESH_MEAN_H)

    rows = []
    for i, tr in enumerate(trophies):
        a, tau, s = best[i]
        base, slope = parts(i, target_h, tau, s)
        med_count = base + a * slope
        med_rate = 100.0 * med_count / n_target
        draws = []
        for _ in range(N_BOOT):
            if rng.random() < p_no_refresh and last_rates[i] is not None:
                # No refresh before the answer read: the page still shows
                # the last observed rates, exactly.
                draws.append(logit(last_rates[i]))
                continue
            b1 = rng.gauss(nmodel.b1, nmodel.sig_b1)
            b1 = min(2.0, max(0.05, b1))
            b0 = rng.gauss(nmodel.b0, nmodel.sig_b0)
            n_t = max(nmodel.n_at(target_h, b0, b1), last_n)
            a_d = min(0.999, a * math.exp(rng.gauss(0.0, 0.35)))
            tau_d = tau * math.exp(rng.gauss(0.0, 0.35))
            tau_d = min(600.0, max(0.5, tau_d))
            tau_key = min((v for (v, ss) in w_grid if ss == s),
                          key=lambda v: abs(math.log(v / tau_d)))
            row = w_grid[(tau_key, s)]
            # Cohort part with the drawn parameters.
            anchor = anchors[i]
            if anchor is None:
                c0_d = 0.0
            else:
                t_first, n0, c0 = anchor
                f_d = w_at(row, target_h - t_first) if target_h > t_first else 0.0
                cap_d = n0 * a_d
                c0_d = c0 if c0 >= cap_d else c0 + (cap_d - c0) * f_d
            total = 0.0
            upto = min(int(math.floor(target_h)), len(joins) - 1)
            for sbin in range(upto + 1):
                total += joins[sbin] * w_at(row, target_h - sbin)
            # Rescale joins by the drawn N curve: approximate by the ratio
            # of drawn to base N at the target (joins scale with N).
            scale = n_t / n_target if n_target > 0 else 1.0
            c_d = c0_d + a_d * total * scale
            r_d = 100.0 * c_d / n_t if n_t > 0 else med_rate
            draws.append(logit(r_d) + rng.gauss(0.0, floor_sd))
        draws.sort()
        qs = []
        for lv in LEVELS:
            pos = lv * (len(draws) - 1)
            lo = int(math.floor(pos))
            hi = min(len(draws) - 1, lo + 1)
            frac = pos - lo
            lg = draws[lo] * (1 - frac) + draws[hi] * frac
            qs.append(round4(clamp_pct(inv_logit(lg))))
        qs = sorted(qs)
        rows.append({"id": tr["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        out = json.dumps(forecast, separators=(",", ":")) + "\n"
        if len(out.encode("utf-8")) > 65536:
            raise ValueError(f"{path.name}: forecast over 65,536 bytes")
        (OUTPUT_DIR / f"{path.stem}.out").write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
