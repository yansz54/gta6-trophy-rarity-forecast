# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/yansz54/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/yansz54/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A count-based convolution forecaster for GTA VI trophy rarity in week one. It models tracked players and earned counts separately, so dilution from new players never looks like players quitting.

## Run it

```sh
MAKEFUN_INPUT_DIR=/input MAKEFUN_OUTPUT_DIR=/output python3 main.py
```

`main.py` (Python 3 standard library only, no network, deterministic) reads every `*.in` case in the input directory and writes one `*.out` forecast per case: five quantiles (0.05, 0.25, 0.5, 0.75, 0.95) of the percent of Exophase-tracked players who will hold each trophy at the case target. Runs in ~1.5 s per final-style case, well under the 8 s cap, and two runs are byte-identical.

## How it works

An earn rate is a ratio — earned players over tracked players — and at a launch both sides move for different reasons. Exophase's page is flat between refreshes and then jumps: launch night showed 638 tracked players until 04:30Z, then 3,868, and every rate but the one at 100% fell in that one refresh, because the new players had earned almost nothing yet. A forecaster that extrapolates the rate itself mistakes dilution for players quitting. This one does not:

- **Tracked players N(t)** — a power-law fit of log N on log(1 + hours since release), fitted to the case history's distinct reads. Its derivative is the join rate.
- **Earned counts C(t)** — two parts. The cohort already tracked at the first read (reviewers and early players) starts from its observed count and grows toward its eventual share. Everyone who joins after release is a convolution of the join rate with a per-trophy earn curve: a lognormal in hours actually played, averaged over a lognormal play-rate distribution. A trophy that needs 18 hours of play is essentially unearned after 2 hours, however fast the player, because the missions themselves take that long.
- **Priors, then fit** — each trophy's eventual share and characteristic play-hours come from a prior table keyed to its kind (chapter, story mission, side strand, collectible, grind, money, activity, misc, 100%, platinum), calibrated from GTA V and Red Dead Redemption 2 plus the launch-night reads, then fitted to the trophy's observed counts by a grid search with a prior penalty.
- **Spread** — a seeded parametric bootstrap over the N-growth parameters and each trophy's (share, hours), plus a no-refresh branch: with probability exp(−horizon / 5 h) the answer read still shows the last observed rates, which is what makes the lower quantiles reach down to where rates land after a refresh.

Full method, calibration sources, and validation: [METHOD.md](METHOD.md).

## Forecast table (practice cases)

The program's actual output on the kit's two practice cases (checkpoint 03:00Z / 03:30Z, target 04:30Z / 05:00Z, 90-minute horizon). Full outputs: [`data/forecasts/`](data/forecasts/). The medians sit at the last observed rates — a refresh in the next 90 minutes is the minority outcome — while the lower quantiles reach the diluted values a refresh would produce (chapter 1: 1.25% → 0.18% after the observed 04:30Z refresh to 3,868 tracked players).

| trophy | last read | q05 | q25 | q50 | q75 | q95 |
|---|---|---|---|---|---|---|
| Vice Got a Price! (platinum) | 0.63 | 0.11 | 0.63 | 0.63 | 0.63 | 0.63 |
| Just Like Paradise (opening mission) | 100.00 | 15.50 | 72.59 | 99.99 | 99.99 | 99.99 |
| Paradise Lost (chapter 1) | 1.25 | 0.18 | 0.80 | 1.25 | 1.25 | 1.25 |
| T**s Ass Drug Money (chapter 2) | 1.10 | 0.12 | 1.10 | 1.10 | 1.10 | 1.10 |
| Vice City, Baby! (chapter 3) | 1.10 | 0.12 | 0.82 | 1.10 | 1.10 | 1.10 |
| You Gotta Know When to Get Off (final mission) | 1.10 | 0.16 | 1.10 | 1.10 | 1.10 | 1.10 |
| A Long Road (100% completion) | 0.63 | 0.08 | 0.29 | 0.63 | 0.63 | 0.63 |

Week-one central projection (model's median q50 across the 12 Monte Carlo worlds in METHOD.md — a simulation-based expectation, not the graded forecast): opening mission ~77%, chapter 1 ~13%, chapter 2 ~8%, chapter 3 ~2.5%, chapter 4 ~1.5%, final mission ~0.15%, 100% and platinum ~0.01%.

## Validation

Monte Carlo (`tools/simulate.py`): synthetic worlds with true parameters drawn independently of the forecaster's priors, the same refresh-and-read cadence as the house, scored with the house's own pinball loss on the logit scale. Final-style cases (2 days of history, 4.5-day horizon), 28 worlds across refresh intervals of 3, 5 and 8 hours:

| method | mean pinball loss |
|---|---|
| **this forecaster** | **0.30** |
| fitted-curve baseline (house) | 1.16 |
| persistence baseline (house) | 1.90 |

It beat both house baselines in 28 of 28 worlds. On the real practice case p00, scored against the observed 04:30Z refresh (3,868 tracked players): 1.07 vs 1.82 for both baselines. Validation harness and scorer: [`tools/score.py`](tools/score.py), [`tools/simulate.py`](tools/simulate.py).

---

<sub>Entered on [make.fun](https://make.fun/?ref=djwcs0np&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
