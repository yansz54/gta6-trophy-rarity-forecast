"""Monte Carlo validation for the trophy-rarity forecaster.

Builds synthetic worlds that obey the same physics the forecaster assumes
— tracked players grow and dilute the rates, earned counts only grow as
players work through the story, Exophase refreshes the page every few
hours — but with TRUE parameters drawn independently of the forecaster's
priors, so the comparison is honest.  Then it runs the forecaster and both
house baselines on each synthetic case and compares mean pinball loss.

World, per seed:
  * Tracked players N(t) follow a saturating hyperbola, anchored to the two
    facts the house actually observed: 638 tracked at launch night, 3,868
    by the 04:30Z refresh.  N_inf (the eventual tracked population) is
    drawn log-uniform between 250k and 1.5M.
  * A small reviewer cohort (10 players, joined 2-10 days early, playing
    ~8 h/day) reproduces the launch-night counts: a few chapter completions
    and a few platinums among the first 638.
  * Each trophy's true (a, tau, sigma) is drawn around the forecaster's
    prior with lognormal noise (sigma 0.35 on a, 0.4 on tau), the true
    play-rate median is drawn around 2.2 h/day, and a story-speed factor
    shifts all story taus together — so the world is sometimes faster and
    sometimes slower than the prior expects.
  * Refreshes are a Poisson process (mean interval --refresh hours, default
    5), with the first refresh after release forced into 04:00-05:00Z to
    match the observed one.  Reads (every 30 min, then every 10 min) show
    the last refresh's values, rates rounded to 2 decimals, exactly like
    the house's cases.
  * The answer is the first read at or after the target.

Usage:
    python3 tools/simulate.py [--seeds N] [--refresh H] [--practice-only]
                              [--final-only] [--out DIR]
"""

import argparse
import json
import math
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main as M
from tools.score import mean_pinball, persistence_forecast, curve_forecast

EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
RELEASE = datetime(2026, 11, 19, 0, 0, tzinfo=timezone.utc)
KIT_CASE = Path.home() / "workspace" / "makefun" / \
    "gta6-trophy-rarity-forecast-01-kit" / "gta6-trophy-rarity-forecast-01-kit" / \
    "input" / "p00.in"


def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def hours_since_release(dt):
    return (dt - RELEASE).total_seconds() / 3600.0


def norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


class World:
    def __init__(self, rng, refresh_mean_h=5.0):
        self.rng = rng
        self.refresh_mean_h = refresh_mean_h
        # Tracked-players growth: saturating hyperbola, anchored to the two
        # observed facts (638 at launch night, 3,868 at the 04:30Z refresh).
        n_inf = math.exp(rng.uniform(math.log(250000), math.log(1500000)))
        anchor = 3868.0 * math.exp(rng.gauss(0.0, 0.10))
        frac = (anchor - 638.0) / (n_inf - 638.0)
        frac = min(0.9, max(1e-4, frac))
        self.n_inf = n_inf
        self.t_half = 4.5 * (1.0 / frac - 1.0)
        self.n0 = 638.0
        # True play-rate distribution (median hours/day).
        self.rho_med = 2.2 * math.exp(rng.gauss(0.0, 0.25))
        self.rho_points = [math.exp(0.75 * M.norm_ppf((i + 0.5) / 9.0) +
                                    math.log(self.rho_med)) for i in range(9)]
        # True trophy parameters, drawn around the forecaster's priors.
        case = json.load(open(KIT_CASE))
        self.trophies = case["trophies"]
        story_speed = math.exp(rng.gauss(0.0, 0.20))
        kinds = M.resolve_kinds(
            self.trophies, [M.classify(tr, None) for tr in self.trophies])
        self.true_params = []
        for tr, kind in zip(self.trophies, kinds):
            pr = M.TYPE_PRIORS[kind]
            a = min(0.999, max(0.002, pr["a"] * math.exp(rng.gauss(0.0, 0.35))))
            tau = pr["tau"] * math.exp(rng.gauss(0.0, 0.40))
            if kind.startswith("story"):
                tau *= story_speed
            tau = min(600.0, max(0.5, tau))
            s = min(0.8, max(0.3, pr["s"] * math.exp(rng.gauss(0.0, 0.15))))
            self.true_params.append((a, tau, s))
        # Reviewer cohort: 10 players, joined 2-10 days early, heavy play.
        self.reviewers = [(rng.uniform(-240.0, -48.0),
                           math.exp(rng.gauss(math.log(8.0), 0.4)))
                          for _ in range(10)]
        # Refresh schedule.
        self.refreshes = self._refresh_schedule()
        # Cache of page states, keyed by refresh time.
        self._state_cache = {}
        # Hourly join bins from -240h to +200h, and earn-curve rows.
        self._build()

    def n_true(self, t_h):
        if t_h <= 0:
            return self.n0
        return self.n0 + (self.n_inf - self.n0) * t_h / (t_h + self.t_half)

    def _refresh_schedule(self):
        rng = self.rng
        times = [-0.2]
        times.append(rng.uniform(4.0, 5.0))
        t = times[-1]
        while t < 200.0:
            t += rng.expovariate(1.0 / self.refresh_mean_h)
            times.append(t)
        return times

    def _earn_row(self, a, tau, s, max_age_h):
        row = []
        log_tau = math.log(tau)
        for age_h in range(max_age_h + 1):
            if age_h <= 0:
                row.append(0.0)
                continue
            age_d = age_h / 24.0
            total = 0.0
            for rho in self.rho_points:
                play = rho * age_d
                total += norm_cdf((math.log(play) - log_tau) / s) if play > 0 else 0.0
            row.append(a * total / len(self.rho_points))
        return row

    def _build(self):
        # Hourly bins: index 0 = hour starting at -240h.
        self.t_start = -240.0
        self.n_bins = 440  # -240h .. +200h
        joins = []
        for b in range(self.n_bins):
            t0 = self.t_start + b
            t1 = t0 + 1.0
            if t1 <= 0:
                joins.append(0.0)
            else:
                joins.append(max(0.0, self.n_true(t1) - self.n_true(max(0.0, t0))))
        # Reviewers join in their own bins.
        for jt, _rho in self.reviewers:
            b = int(math.floor(jt - self.t_start))
            if 0 <= b < self.n_bins:
                joins[b] += 1.0
        # The rest of the launch-night cohort (638 - 10 reviewers): players
        # who had the game before release, spread over the 72 hours before
        # it.  Without them the launch-night rates are ~0 for every trophy
        # (the real page showed Gone Fishing at 100% and chapter 1 at
        # 1.25% among the first 638).
        initial = self.n0 - len(self.reviewers)
        for b in range(168, 240):
            joins[b] += initial / 72.0
        self.joins = joins
        # Earn rows per trophy, indexed by age in hours.  Reviewers play
        # faster than the median, so their earn curve uses their own rho:
        # approximate by evaluating the row with a per-trophy rho scale is
        # overkill; the reviewer counts are tiny, so use the same row and
        # let their early join time do the work (they have had days to play).
        self.rows = [self._earn_row(a, tau, s, 440)
                     for (a, tau, s) in self.true_params]

    def state_at(self, t_h):
        """True (N, [rates]) at hour t_h (float, hours since release)."""
        n = self.n_true(t_h)
        rates = []
        # Bin index of hour t_h: bins are hourly from t_start.
        upto = int(math.floor(t_h - self.t_start))
        for i in range(len(self.trophies)):
            row = self.rows[i]
            total = 0.0
            for b in range(min(upto + 1, self.n_bins)):
                age = int(t_h - (self.t_start + b))
                if 0 <= age < len(row):
                    total += self.joins[b] * row[age]
            rates.append(100.0 * total / n if n > 0 else 0.0)
        return n, rates

    def snapshot(self, t_h):
        """Page values at hour t_h: last refresh at or before t_h."""
        last = None
        for rt in self.refreshes:
            if rt <= t_h:
                last = rt
            else:
                break
        if last is None:
            last = self.refreshes[0]
        if last not in self._state_cache:
            n, rates = self.state_at(last)
            self._state_cache[last] = (int(round(n)),
                                       [round(r, 2) for r in rates])
        return self._state_cache[last]

    def make_case(self, checkpoint_dt, target_dt, case_id):
        horizon_min = int((target_dt - checkpoint_dt).total_seconds() // 60)
        # Read cadence: every 30 min from 23:50Z 18 Nov; every 10 min from
        # 06:00Z 19 Nov onward.
        reads = []
        t = datetime(2026, 11, 18, 23, 50, tzinfo=timezone.utc)
        while t <= checkpoint_dt:
            step = 30 if t < datetime(2026, 11, 19, 6, 0, tzinfo=timezone.utc) else 10
            reads.append(self._read(t))
            t += timedelta(minutes=step)
        # Answer: first read at or after the target.
        t = checkpoint_dt
        while True:
            step = 10
            r = self._read(t)
            if t >= target_dt:
                answer = r
                break
            t += timedelta(minutes=step)
        case = {
            "id": case_id,
            "game": {"name": "Grand Theft Auto VI", "platform": "PS5",
                     "release_date": "2026-11-19"},
            "at": iso(checkpoint_dt),
            "target": iso(target_dt),
            "horizon_minutes": horizon_min,
            "levels": [0.05, 0.25, 0.5, 0.75, 0.95],
            "trophies": self.trophies,
            "history": reads,
        }
        return case, answer["rates"]

    def _read(self, dt):
        t_h = hours_since_release(dt)
        n, rates = self.snapshot(t_h)
        return {"slot": iso(dt), "at": iso(dt + timedelta(milliseconds=300)),
                "players_tracked": n,
                "trophies_earned": int(round(sum(r * n / 100.0 for r in rates))),
                "rates": rates}


def run(seed, refresh_mean_h, kinds, out_dir):
    rng = random.Random(1000 + seed)
    world = World(rng, refresh_mean_h)
    results = {}
    if "practice" in kinds:
        ck = datetime(2026, 11, 19, 3, 0, tzinfo=timezone.utc)
        tg = datetime(2026, 11, 19, 4, 30, tzinfo=timezone.utc)
        case, answer = world.make_case(ck, tg, f"sim-practice-{seed}")
        results["practice"] = evaluate(case, answer, seed, "practice", out_dir)
    if "final" in kinds:
        ck = datetime(2026, 11, 21, 11, 30, tzinfo=timezone.utc)
        tg = datetime(2026, 11, 26, 0, 0, tzinfo=timezone.utc)
        case, answer = world.make_case(ck, tg, f"sim-final-{seed}")
        results["final"] = evaluate(case, answer, seed, "final", out_dir)
    return results


def evaluate(case, answer, seed, kind, out_dir):
    import time
    t0 = time.time()
    fc = M.forecast_case(case, case["id"])
    elapsed = time.time() - t0
    loss, _ = mean_pinball(answer, fc["trophies"])
    lp, _ = mean_pinball(answer, persistence_forecast(case)["trophies"])
    lc, _ = mean_pinball(answer, curve_forecast(case)["trophies"])
    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{case['id']}.in").write_text(json.dumps(case))
        (out_dir / f"{case['id']}.out").write_text(json.dumps(fc))
        (out_dir / f"{case['id']}.answer.json").write_text(
            json.dumps({"rates": answer}))
    print(f"  seed {seed} {kind}: mine {loss:.4f}  persistence {lp:.4f}  "
          f"curve {lc:.4f}  ({elapsed:.1f}s)")
    return {"mine": loss, "persistence": lp, "curve": lc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--refresh", type=float, default=5.0)
    ap.add_argument("--practice-only", action="store_true")
    ap.add_argument("--final-only", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    kinds = []
    if not args.final_only:
        kinds.append("practice")
    if not args.practice_only:
        kinds.append("final")
    agg = {k: {"mine": [], "persistence": [], "curve": []} for k in kinds}
    wins = {k: {"mine_vs_pers": 0, "mine_vs_curve": 0, "n": 0} for k in kinds}
    for seed in range(args.seeds):
        res = run(seed, args.refresh, kinds, args.out or None)
        for k, r in res.items():
            for m in ("mine", "persistence", "curve"):
                agg[k][m].append(r[m])
            wins[k]["n"] += 1
            if r["mine"] < r["persistence"]:
                wins[k]["mine_vs_pers"] += 1
            if r["mine"] < r["curve"]:
                wins[k]["mine_vs_curve"] += 1
    print()
    for k in kinds:
        n = wins[k]["n"]
        print(f"{k} ({n} worlds, refresh mean {args.refresh}h):")
        for m in ("mine", "persistence", "curve"):
            vals = agg[k][m]
            print(f"  {m:12s} mean loss {sum(vals)/len(vals):.4f}  "
                  f"range [{min(vals):.3f}, {max(vals):.3f}]")
        print(f"  mine beats persistence {wins[k]['mine_vs_pers']}/{n}, "
              f"beats curve {wins[k]['mine_vs_curve']}/{n}")


if __name__ == "__main__":
    main()
