"""Score a trophy-rarity forecast with the house's pinball loss.

The house scores every case, each of its trophies and each of the five
levels tau, as one item. Its pinball loss is

    2 * max(tau * d, (tau - 1) * d),  d = logit(y) - logit(q),

where y is the trophy's earn rate in the answering read, q the forecast
quantile at tau for it, and logit(x) = ln(p / (1 - p)) with p = x / 100
after x is clamped to 0.01 .. 99.99. The loss is the mean over every item
of every case, written to 6 decimals; lower is better and 0 is the best.

Usage:
    python3 tools/score.py <forecast.out> <answer.json>
where answer.json holds the answering read's rates, either as
{"rates": [y1, y2, ...]} in trophy-id order or as
{"trophies": [{"id": n, "rate": y}, ...]}. Prints the mean pinball loss,
the typical miss (mean |y - median| in percentage points) and the 90%
coverage (share of trophies with q05 <= y <= q95).
"""

import json
import math
import sys

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def pinball(y: float, q: float, tau: float) -> float:
    d = logit(y) - logit(q)
    return 2.0 * max(tau * d, (tau - 1.0) * d)


def load_answer(path: str) -> dict:
    """Return {trophy_id: rate} from an answer file."""
    data = json.load(open(path, encoding="utf-8"))
    if "rates" in data:
        return {i + 1: r for i, r in enumerate(data["rates"])}
    out = {}
    for row in data.get("trophies", []):
        out[row["id"]] = row["rate"]
    return out


def score(forecast_path: str, answer_path: str) -> dict:
    forecast = json.load(open(forecast_path, encoding="utf-8"))
    answer = load_answer(answer_path)
    total = 0.0
    count = 0
    miss = 0.0
    covered = 0
    scored = 0
    for row in forecast["trophies"]:
        tid = row["id"]
        if tid not in answer:
            continue
        y = answer[tid]
        qs = row["q"]
        for tau, q in zip(LEVELS, qs):
            total += pinball(y, q, tau)
            count += 1
        miss += abs(y - qs[2])
        if qs[0] <= y <= qs[4]:
            covered += 1
        scored += 1
    return {
        "loss": total / count if count else float("nan"),
        "typical_miss": miss / scored if scored else float("nan"),
        "coverage_90": covered / scored if scored else float("nan"),
        "trophies": scored,
        "items": count,
    }


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__.strip().split("\n\n")[0])
        print("usage: python3 tools/score.py <forecast.out> <answer.json>")
        sys.exit(2)
    result = score(sys.argv[1], sys.argv[2])
    print(f"loss {result['loss']:.6f}")
    print(f"typical miss {result['typical_miss']:.4f} points")
    print(f"90% coverage {result['coverage_90']:.3f}")
    print(f"trophies {result['trophies']} items {result['items']}")


if __name__ == "__main__":
    main()
