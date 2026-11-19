# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/yansz54/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/yansz54/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts how many GTA VI players earn each trophy in week one: for every
one of the 55 trophies it outputs the 5th, 25th, 50th, 75th and 95th
percentile of the share of tracked players who will have earned it by the
target time.

The key insight is that the trophy page only refreshes every few hours,
so the answer is the **diluted post-refresh** rate, not the last observed
rate. The forecaster models the refresh as a renewal process (lognormal
intervals, mean 3.8 h), player growth as a saturating logistic in log
time, and each trophy with a lognormal earn curve in play-hours
convolved with join times, then bootstraps 120 draws over growth,
refresh timing and per-trophy parameters to get honest, refresh-aware
quantiles. Practice loss (reconstructed answers) is 0.113 mean, below
the rank-20 cutoff of 0.127; see `METHOD.md` for the full model,
validation, and every seed needed to reproduce the forecasts
byte-identically.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=output python3 main.py
```

Reads one case per JSON file from `$MAKEFUN_INPUT_DIR` (default
`/input`) and writes `$MAKEFUN_OUTPUT_DIR/<id>.out` (default
`/output/<id>.out`). Python 3.13, standard library only, no network,
deterministic, ~0.9 s per case.

## Validate it

```sh
python3 tools/score.py output both   # practice loss vs reconstructed answers
python3 tools/simulate.py 12         # adversarial Monte Carlo, 12 worlds
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=djwcs0np&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
