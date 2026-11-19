# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/yansz54/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/yansz54/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast how many GTA VI players earn each trophy in week one. A
count-based convolution model — not rate extrapolation: it models the
tracked-player count N(t) and the earned count C(t) separately and
divides at the end, rate(T) = 100 · C(T) / N(T). N(t) is a power-law fit
of tracked players against hours since release; C(t) is a pre-release
cohort plus a convolution of the join rate with a lognormal earn curve
in hours actually played, with per-trophy priors (eventual share and
characteristic play-hours) fitted to the observed counts. Spread comes
from a seeded parametric bootstrap (80 draws) mixed with a no-refresh
branch for the page refresh the house has not shown yet. See METHOD.md
for the full method and the Monte Carlo validation (28 synthetic worlds:
0.30 vs 1.16 fitted-curve vs 1.90 persistence).

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` (Python 3, standard library only) reads every `*.in` case in
the input directory and writes one `<id>.out` forecast per case: five
quantiles (0.05, 0.25, 0.5, 0.75, 0.95) of each trophy's earn rate at
the case target. Deterministic: two runs on the same input are
byte-identical. Runtime is about 1.4 s per final-style case (cap 8 s).

`tools/score.py` scores a forecast with the house's pinball loss;
`tools/simulate.py` runs the Monte Carlo validation.

---

<sub>Entered on [make.fun](https://make.fun/?ref=djwcs0np&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
