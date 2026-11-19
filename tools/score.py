"""Scoring for the trophy-rarity forecaster.

Implements the house score exactly as competition.yaml describes it —
mean pinball loss on the logit scale — plus the two house baselines
(persistence and the fitted-curve forecaster) and the validity checks.

    python3 tools/score.py <case.in> <forecast.out> <answer.json>

<answer.json> is {"rates": [percent, ...]} in trophy order, i.e. the
answer read's rates.  Prints the mean pinball loss for this forecast and
for both baselines, plus any validity failures.
"""

import json
import math
import sys

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]
RELEASE_EPOCH = 1795046400  # 2026-11-19T00:00:00Z (game release day)


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


def pinball_loss(y, q, tau):
    d = logit(y) - logit(q)
    return 2.0 * max(tau * d, (tau - 1.0) * d)


def mean_pinball(answer_rates, forecast_rows):
    """Mean pinball loss over trophies and levels."""
    total = 0.0
    n = 0
    per_trophy = []
    for y, row in zip(answer_rates, forecast_rows):
        if y is None:
            per_trophy.append(None)
            continue
        losses = [pinball_loss(y, q, tau) for q, tau in zip(row["q"], LEVELS)]
        per_trophy.append(sum(losses) / len(losses))
        total += sum(losses)
        n += len(losses)
    return total / n if n else float("nan"), per_trophy


def parse_time(s):
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    from datetime import datetime

    return datetime.fromisoformat(s).timestamp()


def last_rates(history, n_tr):
    rates = history[-1].get("rates") or [None] * n_tr
    return rates


def persistence_forecast(case):
    """House baseline 1: q50 = last rate, spread 0.5*sqrt(days) on logits."""
    n_tr = len(case["trophies"])
    rates = last_rates(case["history"], n_tr)
    days = case["horizon_minutes"] / 1440.0
    sd = 0.5 * math.sqrt(days)
    rows = []
    for i, tr in enumerate(case["trophies"]):
        base = rates[i] if rates[i] is not None else 0.5
        lg = logit(base)
        rows.append({"id": tr["id"],
                     "q": [round(min(99.99, max(0.01, inv_logit(lg + z * sd))), 4)
                           for z in Z]})
    return {"trophies": rows}


def curve_forecast(case):
    """House baseline 2: least-squares logit line vs ln(1+hours) over the
    last 24h of reads, extrapolated to the target; same spread."""
    n_tr = len(case["trophies"])
    release_epoch = RELEASE_EPOCH
    gd = (case.get("game") or {}).get("release_date")
    if gd:
        release_epoch = parse_time(gd + "T00:00:00Z")
    target_h = max(0.0, (parse_time(case["target"]) - release_epoch) / 3600.0)
    cutoff = parse_time(case["at"]) - 24 * 3600
    reads = [r for r in case["history"] if parse_time(r["at"]) >= cutoff]
    if len(reads) < 2:
        reads = case["history"][-2:]
    days = case["horizon_minutes"] / 1440.0
    sd = 0.5 * math.sqrt(days)
    rows = []
    for i, tr in enumerate(case["trophies"]):
        xs, ys = [], []
        for r in reads:
            v = (r.get("rates") or [None] * n_tr)[i]
            if v is None:
                continue
            t_h = max(0.0, (parse_time(r["at"]) - release_epoch) / 3600.0)
            xs.append(math.log1p(t_h))
            ys.append(logit(v))
        if len(xs) >= 2 and max(xs) > min(xs):
            mx = sum(xs) / len(xs)
            my = sum(ys) / len(ys)
            sxx = sum((x - mx) ** 2 for x in xs)
            sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            b1 = sxy / sxx if sxx > 0 else 0.0
            b0 = my - b1 * mx
            lg = b0 + b1 * math.log1p(target_h)
        elif ys:
            lg = ys[-1]
        else:
            lg = logit(0.5)
        rows.append({"id": tr["id"],
                     "q": [round(min(99.99, max(0.01, inv_logit(lg + z * sd))), 4)
                           for z in Z]})
    return {"trophies": rows}


def validate(case, forecast):
    """Validity checks mirroring the kit: ids, order, quantiles, size."""
    errors = []
    rows = forecast.get("trophies")
    if not isinstance(rows, list):
        return ["trophies is not a list"]
    if len(rows) != len(case["trophies"]):
        errors.append(f"row count {len(rows)} != trophy count {len(case['trophies'])}")
    for row, tr in zip(rows, case["trophies"]):
        if row.get("id") != tr["id"]:
            errors.append(f"row id {row.get('id')} != trophy id {tr['id']}")
        q = row.get("q")
        if not isinstance(q, list) or len(q) != 5:
            errors.append(f"trophy {tr['id']}: q is not 5 numbers")
            continue
        if any(v is None or not (0.01 <= v <= 99.99) for v in q):
            errors.append(f"trophy {tr['id']}: q out of [0.01, 99.99]: {q}")
        if any(q[j] > q[j + 1] for j in range(4)):
            errors.append(f"trophy {tr['id']}: q not non-decreasing: {q}")
    return errors


def main():
    case = json.load(open(sys.argv[1]))
    forecast = json.load(open(sys.argv[2]))
    answer = json.load(open(sys.argv[3]))
    answer_rates = answer["rates"]
    errs = validate(case, forecast)
    print("validity:", "OK" if not errs else errs)
    loss, per = mean_pinball(answer_rates, forecast["trophies"])
    print(f"forecast mean pinball loss: {loss:.4f}")
    for name, fc in [("persistence", persistence_forecast(case)),
                     ("fitted-curve", curve_forecast(case))]:
        l2, _ = mean_pinball(answer_rates, fc["trophies"])
        print(f"{name} mean pinball loss: {l2:.4f}")
    worst = sorted(((per[i], case["trophies"][i]["name"]) for i in range(len(per))
                    if per[i] is not None), reverse=True)[:5]
    print("worst trophies (loss, name):")
    for l, nm in worst:
        print(f"  {l:.3f}  {nm}")


if __name__ == "__main__":
    main()
