#!/usr/bin/env python3
"""Pinball scorer (the house's formula) + practice-answer reconstruction.

Reconstruction (brief section 5a): the 04:30 UTC read is 3,868 players
tracked and 4,276 trophies earned. Base: new joiners earned only the
opening mission (which stays at 100%); every other trophy's post-refresh
rate is its pre-refresh earned count divided by 3,868. Sensitivity band:
the remaining 103 non-opening trophies are spread over the non-opening
trophies in proportion to their pre-refresh earned counts.

Usage:
  python3 tools/score.py <outdir> [base|band|both]
  python3 tools/score.py --baseline <indir>   (score the house persistence
                                                baseline on the practice)
"""
import json
import math
import os
import sys

KIT_INPUT = os.path.expanduser(
    "~/workspace/makefun/gta6-trophy-rarity-forecast-01-kit/"
    "gta6-trophy-rarity-forecast-01-kit/input")
NEW_PLAYERS = 3868
OLD_PLAYERS = 638


def logit_pct(x):
    p = min(max(x, 0.01), 99.99) / 100.0
    return math.log(p / (1.0 - p))


def pinball(y, q, tau):
    d = logit_pct(y) - logit_pct(q)
    return 2.0 * max(tau * d, (tau - 1.0) * d)


def reconstruct(case, band=False):
    """Return {trophy_id: true rate} for a practice case."""
    last = case["history"][-1]
    rates = last["rates"]
    # Pre-refresh earned counts.
    earned = [r / 100.0 * OLD_PLAYERS if r is not None else 0.0 for r in rates]
    # Opening mission: the trophy at 100% (id 2, "Just Like Paradise").
    opening = max(range(len(rates)), key=lambda i: rates[i] or 0)
    total_non_opening = sum(e for i, e in enumerate(earned) if i != opening)
    out = {}
    for i, tr in enumerate(case["trophies"]):
        if i == opening:
            out[tr["id"]] = 100.0
            continue
        e = earned[i]
        if band:
            e += 103.0 * e / max(total_non_opening, 1.0)
        out[tr["id"]] = 100.0 * e / NEW_PLAYERS
    return out


def score_forecast(forecast, truth, levels=(0.05, 0.25, 0.5, 0.75, 0.95)):
    """Return (loss, typical_miss_pp, coverage_90, loss_by_level dict)."""
    rows = {r["id"]: r["q"] for r in forecast["trophies"]}
    tot = 0.0
    n = 0
    by_level = {t: [] for t in levels}
    miss = 0.0
    cover = 0
    for tid, y in truth.items():
        q = rows[tid]
        miss += abs(y - q[2])
        if q[0] <= y <= q[4]:
            cover += 1
        for tau, qv in zip(levels, q):
            l = pinball(y, qv, tau)
            by_level[tau].append(l)
            tot += l
            n += 1
    loss = tot / max(n, 1)
    return (loss, miss / max(len(truth), 1), cover / max(len(truth), 1),
            {t: sum(v) / max(len(v), 1) for t, v in by_level.items()})


def persistence_forecast(case):
    """The house baseline: last rates, random-walk spread."""
    last = case["history"][-1]["rates"]
    d_days = case["horizon_minutes"] / 1440.0
    z = [-1.6448536269514722, -0.6744897501960817, 0.0,
         0.6744897501960817, 1.6448536269514722]
    rows = []
    for tr, r in zip(case["trophies"], last):
        if r is None:
            r = 0.01
        lg = logit_pct(r)
        qs = []
        for zi in z:
            v = 100.0 / (1.0 + math.exp(-(lg + zi * 0.5 * math.sqrt(d_days))))
            qs.append(round(min(max(v, 0.01), 99.99), 4))
        rows.append({"id": tr["id"], "q": qs})
    return {"trophies": rows}


def main():
    args = sys.argv[1:]
    if args and args[0] == "--baseline":
        in_dir = args[1] if len(args) > 1 else KIT_INPUT
        for cid in ("p00", "p01"):
            case = json.load(open(os.path.join(in_dir, cid + ".in")))
            for band in (False, True):
                truth = reconstruct(case, band=band)
                fc = persistence_forecast(case)
                loss, miss, cov, by = score_forecast(fc, truth)
                print(f"persistence {cid} band={band}: loss={loss:.4f} "
                      f"miss={miss:.3f} cov={cov:.3f} "
                      f"by_level={[round(by[t], 3) for t in (0.05, 0.25, 0.5, 0.75, 0.95)]}")
        return 0
    out_dir = args[0]
    which = args[1] if len(args) > 1 else "both"
    for cid in ("p00", "p01"):
        case = json.load(open(os.path.join(KIT_INPUT, cid + ".in")))
        fc = json.load(open(os.path.join(out_dir, cid + ".out")))
        bands = [False, True] if which == "both" else [which == "band"]
        for band in bands:
            truth = reconstruct(case, band=band)
            loss, miss, cov, by = score_forecast(fc, truth)
            print(f"{cid} band={band}: loss={loss:.4f} miss={miss:.4f} "
                  f"cov={cov:.3f} "
                  f"by_level={[round(by[t], 3) for t in (0.05, 0.25, 0.5, 0.75, 0.95)]}")
            # median check: is the reported median the stale last rate?
            last = case["history"][-1]["rates"]
            rows = {r["id"]: r["q"] for r in fc["trophies"]}
            stale_med = sum(1 for i, tr in enumerate(case["trophies"])
                            if abs(rows[tr["id"]][2] - last[i]) < 1e-9)
            print(f"   medians equal to last observed rate: {stale_med}/55")
    return 0


if __name__ == "__main__":
    sys.exit(main())
