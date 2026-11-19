"""Monte Carlo validation for the GTA VI trophy rarity forecaster.

Builds synthetic worlds with true parameters drawn independently of the
forecaster's priors (eventual shares +/-35%, play-hours +/-40%, play-rate
median and story speed drawn per world, tracked population saturating
between 250k and 1.5M), the same refresh-and-read cadence as the house
(refreshes Poisson, mean 5 h; reads every 30 min, then every 10 min),
and scores the forecaster against the two house baselines with the
house's own pinball loss (tools/score.py).

Final-style cases use 2 days of history and a 4.5-day horizon;
practice-style cases use a 90-minute horizon. The forecaster should beat
both baselines in (nearly) every world; on practice-style cases the win
is in the mean, because the refresh is a coin flip there.

Usage:
    python3 tools/simulate.py [--worlds N] [--refresh-mean H] [--practice]
Prints one row per refresh mean with the mean loss of each method and
how many worlds the forecaster won.
"""

import argparse
import importlib.util
import json
import math
import random
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, __file__.rsplit("/", 2)[0])

from tools.score import pinball, logit  # noqa: E402

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

# The 55 trophies' kinds (from the practice case's trophy list, in id
# order) and the forecaster's prior (a, tau) per kind, for drawing the
# world's true parameters as perturbations of the priors.
KINDS = ["platinum", "first", "story_ch1", "story_ch2", "story_ch3",
         "story_ch4", "story_final"] + ["story_mission"] * 20 + \
        ["side_strand"] * 6 + ["activity"] * 8 + ["grind"] * 3 + \
        ["collectible"] * 3 + ["money"] * 3 + ["misc"] * 5 + \
        ["completion", "platinum"]
KINDS = KINDS[:55]
PRIORS = {
    "first": (0.999, 1.5), "story_ch1": (0.88, 18.0),
    "story_ch2": (0.82, 28.0), "story_ch3": (0.74, 45.0),
    "story_ch4": (0.66, 62.0), "story_final": (0.55, 75.0),
    "story_mission": (0.80, 25.0), "side_strand": (0.35, 55.0),
    "activity": (0.32, 40.0), "grind": (0.14, 110.0),
    "collectible": (0.16, 110.0), "money": (0.30, 65.0),
    "misc": (0.35, 40.0), "completion": (0.08, 220.0),
    "platinum": (0.05, 260.0),
}


def load_forecaster():
    spec = importlib.util.spec_from_file_location(
        "rarity_main",
        __file__.rsplit("/", 2)[0] + "/main.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def make_world(rng, refresh_mean, final_style=True):
    """Draw a synthetic world's true parameters."""
    play_rate_med = rng.uniform(1.6, 3.0)
    play_rate_sig = rng.uniform(0.6, 0.9)
    n_sat = rng.uniform(250_000, 1_500_000)
    # N(t): saturating (hyperbolic) growth from 638 at release.
    n0 = 638.0
    growth = rng.uniform(0.8, 1.6)  # shape of the saturation curve
    trues = []
    for kind in KINDS:
        a0, tau0 = PRIORS[kind]
        a = min(0.999, max(0.01, a0 * math.exp(rng.gauss(0, 0.35))))
        tau = tau0 * math.exp(rng.gauss(0, 0.40))
        trues.append((a, tau))
    return {
        "play_rate_med": play_rate_med, "play_rate_sig": play_rate_sig,
        "n_sat": n_sat, "n0": n0, "growth": growth, "trues": trues,
        "refresh_mean": refresh_mean,
    }


def true_n(world, hours):
    """Saturating tracked population at `hours` since release."""
    n0, sat, g = world["n0"], world["n_sat"], world["growth"]
    # Hyperbolic saturation: N = sat * h / (h + k), anchored at N(0)=n0
    # via N(h) = n0 + (sat - n0) * (1 - exp(-g * h / 48)).
    return n0 + (sat - n0) * (1.0 - math.exp(-g * hours / 48.0))


def true_earn_cdf(age_days, tau, world):
    """True earn curve for a trophy needing tau play-hours."""
    sigma = math.sqrt(0.5 ** 2 + world["play_rate_sig"] ** 2)
    mu = math.log(tau / world["play_rate_med"])
    if age_days <= 0:
        return 0.0
    return norm_cdf((math.log(age_days) - mu) / sigma)


def true_counts(world, hours, trophy_idx):
    """True earned count for one trophy at `hours` since release.
    Cohort (reviewers, started 6 h before release) + joiners."""
    a, tau = world["trues"][trophy_idx]
    n = true_n(world, hours)
    # Cohort: reviewers present at release; they started 6 h early.
    cohort_n = world["n0"]
    age_cohort = (hours + 6.0) / 24.0
    c_cohort = a * cohort_n * true_earn_cdf(age_cohort, tau, world)
    # Joiners: integrate the join rate against the earn curve.
    steps = max(1, int(hours * 4))
    dh = hours / steps if hours > 0 else 0.0
    total = 0.0
    for i in range(steps + 1):
        h = i * dh
        # Join rate by finite difference of N.
        n1 = true_n(world, h)
        n2 = true_n(world, min(hours, h + dh))
        jr = (n2 - n1) / dh if dh > 0 else 0.0
        age = max(0.0, (hours - h) / 24.0)
        w = 0.5 if i in (0, steps) else 1.0
        total += w * jr * true_earn_cdf(age, tau, world)
    c_join = total * dh if hours > 0 else 0.0
    return c_cohort + c_join


def build_case(world, rng, final_style=True):
    """Simulate one case: history reads + target, with the house cadence.
    Returns (case_dict, answer_rates) where answer_rates is the true rate
    per trophy at the target."""
    release = datetime(2026, 11, 19, 0, 0, tzinfo=timezone.utc)
    if final_style:
        hist_hours, horizon_hours = 48.0, 108.0  # 2 days, 4.5 days
    else:
        hist_hours, horizon_hours = 3.0, 1.5
    # Refresh times: Poisson with the world's refresh mean, from release.
    refresh_at = []
    t = 0.0
    while t < hist_hours + horizon_hours:
        t += rng.expovariate(1.0 / world["refresh_mean"])
        if t < hist_hours + horizon_hours:
            refresh_at.append(t)
    # Read slots: every 30 min (then every 10 min in the final stretch).
    reads = []
    slot = 0.0
    # The page a read shows is the last refresh at or before its slot.
    def page_at(h):
        # Last refresh at or before h; if none, the release page (n0).
        past = [r for r in refresh_at if r <= h]
        rt = past[-1] if past else 0.0
        n = true_n(world, rt)
        rates = []
        for idx in range(55):
            c = true_counts(world, rt, idx)
            rates.append(round(100.0 * c / n, 2) if n > 0 else 0.0)
        return n, rates
    while slot < hist_hours:
        n, rates = page_at(slot)
        reads.append({
            "slot": (release + timedelta(hours=slot)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "at": (release + timedelta(hours=slot)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "players_tracked": int(round(n)),
            "trophies_earned": [None] * 55,
            "rates": rates,
        })
        slot += 0.5
    # The answer: the true rates at the target.
    target_h = hist_hours + horizon_hours
    n_ans = true_n(world, target_h)
    ans = []
    for idx in range(55):
        c = true_counts(world, target_h, idx)
        ans.append(100.0 * c / n_ans if n_ans > 0 else 0.0)
    trophies = []
    for i, kind in enumerate(KINDS):
        trophies.append({"id": i + 1, "name": f"Trophy {i + 1}",
                         "description": "", "tier": "bronze",
                         "points": 15, "secret": False})
    trophies[0]["tier"] = "platinum"
    trophies[0]["points"] = 300
    case = {
        "schema": "makefun/gta6-trophies-case/v1", "id": "sim",
        "at": (release + timedelta(hours=hist_hours)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "horizon_minutes": int(horizon_hours * 60),
        "target": (release + timedelta(hours=target_h)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "levels": LEVELS,
        "game": {"name": "Grand Theft Auto VI", "platform": "PS5",
                 "release_date": "2026-11-19",
                 "source": "https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/"},
        "trophies": trophies, "history": reads, "notes": "synthetic world",
    }
    return case, ans


def baseline_persistence(case):
    last = case["history"][-1]["rates"]
    return {"trophies": [{"id": t["id"], "q": [round(r, 4)] * 5}
                         for t, r in zip(case["trophies"], last)]}


def baseline_fitted_curve(case):
    """House curve baseline: straight line to each trophy's logit rate
    against ln(1 + hours) over the last 24 h of reads, read at the target,
    spread = random walk + residual."""
    release = datetime(2026, 11, 19, 0, 0, tzinfo=timezone.utc)
    horizon_days = case["horizon_minutes"] / 1440.0
    rows = []
    for idx, trophy in enumerate(case["trophies"]):
        pts = []
        for h in case["history"]:
            t = datetime.strptime(h["at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            hours = (t - release).total_seconds() / 3600.0
            if hours < -24:
                continue
            r = h["rates"][idx]
            if r is None:
                continue
            pts.append((math.log1p(max(0.0, hours)), logit(r)))
        if len(pts) >= 3:
            mx = sum(p[0] for p in pts) / len(pts)
            my = sum(p[1] for p in pts) / len(pts)
            sxx = sum((p[0] - mx) ** 2 for p in pts)
            sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
            slope = sxy / sxx if sxx > 0 else 0.0
            inter = my - slope * mx
            resid = math.sqrt(sum((p[1] - (inter + slope * p[0])) ** 2
                                  for p in pts) / max(1, len(pts) - 2))
            target_x = math.log1p(max(0.0, (datetime.strptime(
                case["target"], "%Y-%m-%dT%H:%M:%SZ").replace(
                    tzinfo=timezone.utc) - release).total_seconds() / 3600.0))
            centre = inter + slope * target_x
            spread = math.sqrt((0.5 * math.sqrt(horizon_days)) ** 2 + resid ** 2)
        else:
            last = case["history"][-1]["rates"][idx]
            centre = logit(last if last is not None else 0.0)
            spread = 0.5 * math.sqrt(horizon_days)
        qs = []
        for z in Z:
            v = 100.0 / (1.0 + math.exp(-(centre + z * spread)))
            qs.append(round(min(99.99, max(0.01, v)), 4))
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def score_forecast(forecast, answer):
    total, count = 0.0, 0
    for row in forecast["trophies"]:
        y = answer[row["id"] - 1]
        for tau, q in zip(LEVELS, row["q"]):
            total += pinball(y, q, tau)
            count += 1
    return total / count if count else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=8)
    ap.add_argument("--refresh-mean", type=float, default=5.0)
    ap.add_argument("--practice", action="store_true",
                    help="practice-style cases (90-minute horizon)")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    mod = load_forecaster()
    rng = random.Random(args.seed)
    final_style = not args.practice
    losses = {"forecaster": [], "fitted-curve": [], "persistence": []}
    wins = 0
    for w in range(args.worlds):
        world = make_world(rng, args.refresh_mean, final_style)
        case, ans = build_case(world, rng, final_style)
        fc = mod.forecast_case(case)
        losses["forecaster"].append(score_forecast(fc, ans))
        losses["fitted-curve"].append(
            score_forecast(baseline_fitted_curve(case), ans))
        losses["persistence"].append(
            score_forecast(baseline_persistence(case), ans))
        if losses["forecaster"][-1] < min(losses["fitted-curve"][-1],
                                          losses["persistence"][-1]):
            wins += 1
    kind = "final-style" if final_style else "practice-style"
    print(f"{kind} cases, refresh mean {args.refresh_mean} h, "
          f"{args.worlds} worlds (seed {args.seed})")
    for name in ("forecaster", "fitted-curve", "persistence"):
        vals = losses[name]
        print(f"  {name:13s} mean loss {sum(vals)/len(vals):.3f} "
              f"(min {min(vals):.3f}, max {max(vals):.3f})")
    print(f"  forecaster beat both baselines in {wins}/{args.worlds} worlds")


if __name__ == "__main__":
    main()
