# Method — GTA VI trophy rarity, week one

`main.py` forecasts, for each of GTA VI's 55 PlayStation trophies, the percent
of Exophase-tracked players who will hold it at the case target, as five
quantiles (0.05, 0.25, 0.5, 0.75, 0.95). Scoring is mean pinball loss on the
logit scale, lower is better, and a prize needs a loss strictly below the
best house entry. This note explains the model, where its numbers come from,
and how it was validated.

## 1. Why counts, not rates

An earn rate is a ratio: earned players over tracked players. At a launch
both sides move, and they move for different reasons.

- **Tracked players jump in steps.** Exophase refreshes the page every few
  hours; between refreshes the page is frozen. Launch night showed 638
  tracked players from 23:50Z until 04:00Z, then 3,868 by the 04:30Z refresh.
  Every rate but the one at 100% fell in that one refresh, because the new
  players had earned almost nothing yet.
- **Earned counts only grow**, as players work through the story.

A forecaster that extrapolates the rate itself — both house baselines do —
mistakes dilution for players quitting, and mistakes a flat page between
refreshes for a flat world. This program models the two sides separately
and divides at the end: rate(T) = 100 · C(T) / N(T).

## 2. Tracked players N(t)

log N is fitted linearly in log(1 + hours since release) over the case
history's distinct reads (consecutive identical reads collapsed — a flat
page is one observation, not seven). The join rate n(t) is the derivative.
The fit is capped at 2,000,000 tracked players. When the history is flat
the slope is unidentified, so the fit keeps a wide prior width — that
uncertainty is exactly what the bootstrap below turns into spread.

## 3. Earned counts C(t)

Two parts.

**The pre-release cohort.** The players already tracked at the first read
are reviewers and early players who had played before release. Their count
starts at the observed count and grows toward the cohort's eventual share
as its non-holders keep playing.

**Everyone who joins after release** is a convolution of the join rate
with a per-trophy earn curve F:

    C(t) = ∫ n(s) · F(t − s) ds

F is the share of joiners who have earned the trophy by a given age: a
lognormal in *hours actually played*, averaged over a lognormal
distribution of play rates (median 2.2 hours/day, sigma 0.75 — launch-week
players play more than the year-round average, with a long tail of
marathon players). The lognormal's thin left tail is the point: a trophy
that needs 18 hours of play is essentially unearned after 2 hours, however
fast the player, because the missions themselves take that long. (An
earlier Weibull version of the curve failed exactly here: it let 4% of the
fastest players finish a 22-mission chapter in under two hours of play.)

## 4. Priors, then fit

Each trophy's eventual share `a` (among tracked players) and
characteristic play-hours `tau` come from a prior table keyed to its kind,
then fitted to the trophy's observed counts in the case history by a grid
search with a prior penalty. The predicted count is linear in `a`, so `a`
is searched exactly and `tau` on a fine log grid (15 multipliers).

| kind | a | tau (h) | what it is |
|---|---|---|---|
| first | 0.999 | 1.5 | opening mission |
| story_ch1 | 0.88 | 18 | complete chapter 1 (22 missions) |
| story_ch2 | 0.82 | 28 | complete chapter 2 |
| story_ch3 | 0.74 | 45 | complete chapter 3 (26 missions) |
| story_ch4 | 0.66 | 62 | complete chapter 4 (25 missions) |
| story_final | 0.55 | 75 | complete the final mission |
| story_mission | 0.80 | 25 | a named story mission / heist |
| side_strand | 0.35 | 55 | a stranger's strand |
| activity | 0.32 | 40 | one-off activities |
| grind | 0.14 | 110 | 250 stunts, 50 challenges, 20 races |
| collectible | 0.16 | 110 | card sets, spots, 100 vehicles |
| money | 0.30 | 65 | launder / spend / buy big |
| misc | 0.35 | 40 | one-off moments |
| completion | 0.08 | 220 | 100% completion |
| platinum | 0.05 | 260 | the platinum |

Kinds are classified at runtime from the trophy's tier, name and
description (chapter regex, "final mission for X" is a strand not the
final mission, a capitalised "Complete X." is a story mission, counted
repeats are grinds), so the program generalises to any trophy list, not
just this one.

### Calibration sources

- **GTA VI story**: 91 missions across 5 chapters (22 / 11–13 / 26 / 25 /
  7); main story ~50–80 hours (reviewers 50–90 h), 100% completion
  ~200 hours. Chapter boundaries map to cumulative play of roughly
  16 / 24 / 43 / 61 / 66 hours.
- **GTA V, mature rates**: first-mission trophy 43% PSN-wide but ~93%
  among tracker users; final mission 8.7% PSN-wide; 100% completion 0.2%;
  platinum 0.1% PSN-wide but ~4–6% among tracker users. Tracked players
  are self-selected enthusiasts, so eventual shares run well above
  PSN-wide rates — the priors use tracker-user levels, not PSN-wide ones.
- **Launch-night reads** (the case history itself): 638 tracked players,
  943 trophies earned, rates flat 23:50Z–04:00Z, then 3,868 tracked and
  4,276 trophies by 04:30Z. Chapter 1 at 1.10% among the first 638 (7
  players — reviewers), the opening mission at 100%.

## 5. Spread (the quantiles)

A seeded parametric bootstrap (80 draws, fixed seed — two runs are
byte-identical): the N-growth parameters and each trophy's (a, tau) are
resampled, the forecast is recomputed, and the five quantiles are read off
the simulated logit rates, with a small floor spread added.

On short horizons the dominant risk is a page refresh the house has not
shown yet, so the bootstrap mixes in a **no-refresh branch**: with
probability exp(−horizon / 5 h) the answer read shows exactly the last
observed rates. That is what makes the lower quantiles reach down to where
the rates land after a refresh (chapter 1: 1.25% → ~0.2% after the
observed 04:30Z refresh), while the median stays at the last observed rate
— the refresh in the next 90 minutes is the minority outcome.

## 6. Validation

### Monte Carlo (`tools/simulate.py`)

Synthetic worlds with true parameters drawn independently of the
forecaster's priors (eventual shares ±35%, play-hours ±40%, play-rate
median and story speed drawn per world, tracked population saturating
between 250k and 1.5M), the same refresh-and-read cadence as the house
(refreshes Poisson, mean 5 h; reads every 30 min, then every 10 min),
scored with the house's own pinball loss. Final-style cases (2 days of
history, 4.5-day horizon):

| refresh mean | worlds | this forecaster | fitted-curve (house) | persistence (house) |
|---|---|---|---|---|
| 3 h | 8 | **0.29** | 1.00 | 1.89 |
| 5 h | 12 | **0.30** | 1.16 | 1.85 |
| 8 h | 8 | **0.33** | 1.15 | 1.87 |

It beat both house baselines in **28 of 28 worlds**. Practice-style cases
(90-minute horizon): mean 0.30 vs 0.46 for both baselines — the refresh is
a coin flip there, so the win is in the mean, not in every world.

### Real practice case p00

Scored against the observed 04:30Z refresh (3,868 tracked players;
per-trophy answer rates reconstructed from the launch-night counts, with
the opening mission the only one new players complete within 4.5 hours):
**1.07** vs 1.82 for both house baselines. (The house quotes ~1.51 for the
baselines on practice; the difference is the reconstructed answer, not the
scorer — the scorer is the house's formula, implemented in
`tools/score.py`.)

### Checks

- Validity: one row per trophy, ids in case order, five non-decreasing
  quantiles in [0.01, 99.99], output under 65,536 bytes.
- Determinism: two runs on the same input are byte-identical.
- Runtime: ~0.15 s per practice case, ~1.4 s per final-style case (cap 8 s).
- Standard library only; no network, no wall-clock reads.

## 7. Week-one central projection

The model's median q50 across the 12 Monte Carlo final worlds (5 h
refresh). A simulation-based expectation, **not** the graded forecast —
the graded forecast is whatever `main.py` outputs from the final case.

| trophy | median q50 (%) | trophy | median q50 (%) |
|---|---|---|---|
| Vice Got a Price! (platinum) | 0.01 | Hot Together | 1.49 |
| Just Like Paradise (opening) | 77.29 | Retail Therapy | 1.39 |
| Paradise Lost (ch 1) | 13.04 | Whip It Good | 0.03 |
| T**s Ass Drug Money (ch 2) | 7.62 | Gunslinger | 2.59 |
| Vice City, Baby! (ch 3) | 2.45 | Catch Me If You Can | 1.13 |
| Top Of The World, Ma (ch 4) | 1.53 | Challenging Tank | 1.39 |
| You Gotta Know When to Get Off (final) | 0.15 | Sneakthief | 2.01 |
| Trust | 13.25 | Maximum Withdrawal | 1.60 |
| Break the Bank | 8.98 | A Classic Blitz Play | 1.64 |
| Family Matters | 8.76 | The Goodtime State | 0.04 |
| DWNFALL | 5.71 | Pest Control | 1.35 |
| Hi-Jacked | 7.64 | In The Beginning… | 1.40 |
| Snow Storm | 9.35 | Capital Offense | 1.47 |
| A Long Road (100%) | 0.01 | Let’s Go Bowling | 1.92 |
| Cunning Stunts | 0.02 | Personal Best | 1.55 |
| Master of None | 1.45 | Actions Speak Louder | 1.66 |
| The Shootist | 0.03 | Bad Luck, I Guess | 0.99 |
| Top Fun | 1.94 | The Truth Is Out There | 1.79 |
| Earn Cash, Not Kudos | 0.03 | Ten Pin | 2.43 |
| Baited My Tackle | 2.02 | Dime on the Dollar | 0.62 |
| Stick Fiddler | 1.50 | Treat Yo Self | 1.04 |
| Road to Rehabilitation | 0.68 | Little Boxes | 0.30 |
| This Was No Boat Accident | 0.75 | On the Spot | 0.04 |
| The Lone Rider | 0.73 | | |
| Strangers & Freaks | 0.74 | | |
| Supply & Demand | 0.98 | | |
| Spin On This! | 0.62 | | |
| Positive Mental Attitude | 0.03 | | |
| Fair Game | 1.14 | | |
| Don't Touch My Deck | 0.04 | | |
| You WAiNKer! | 0.04 | | |
| One Last Thing | 0.04 | | |

## 8. Limitations (honest)

- The N-growth model is a power law; the true tracked population
  saturates (hyperbolic). Over a 4.5-day horizon the over-extrapolation is
  ~5–15% and the bootstrap's slope spread covers it, but a saturating
  N-model would be strictly better if the history were longer.
- The play-rate distribution (median 2.2 h/day) is a prior, not measured:
  no public data breaks out Exophase-tracked players' daily play hours.
  The fit corrects it per trophy from the observed counts, but only where
  the history has signal.
- Secret trophies (19 of 55) have hidden descriptions; they get the `misc`
  prior and are fitted purely from their counts.
- The refresh interval (mean 5 h) is estimated from a single observed
  interval (flat 23:50Z–04:00Z, refreshed by 04:30Z). The final is
  insensitive to it (28/28 wins at 3, 5 and 8 h); the practice cases are
  not, which is why their win is in the mean rather than every world.
