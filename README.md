[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Alf212/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Alf212/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# GTA VI trophy rarity forecast

An entry for make.fun's [gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
bounty: forecast, for every PlayStation 5 trophy of Grand Theft Auto VI, the
share of Exophase-tracked players who will have earned it at
2026-11-26T00:00:00Z, one week after launch — as five quantiles per trophy
(5%, 25%, 50%, 75%, 95%), scored by pinball loss on the logit scale.

## How it runs

```
python3 main.py        # reads $MAKEFUN_INPUT_DIR (default /input),
                       # writes $MAKEFUN_OUTPUT_DIR (default /output)
```

`main.py` (Python 3.13, standard library only, no network) reads every
`*.in` case file in the input directory and writes one `<id>.out` forecast
per case. It is deterministic: two runs on the same case write the same
bytes.

## The forecast

Each case holds every read the house made of the game's Exophase trophy
page before the checkpoint. The page only changes when Exophase refreshes
it, so the history is first compressed to its distinct snapshots. Then,
per trophy:

- **Trend model** — a weighted least squares line of the logit earn rate
  against ln(1 + hours since release), weighted toward recent snapshots,
  with the slope shrunk toward zero by its own standard error. Log-time is
  the natural clock for progress through a game: fast early, slowing later.
- **Count model** — a rate is earners ÷ players tracked, so the pool size
  and each trophy's earner count are extrapolated separately and divided.
  This sees dilution coming: when the tracked pool grows faster than a
  trophy's earners, the rate falls even as more players earn it.
- The **median** averages the two models' logits; their disagreement
  widens the spread. The **spread** combines a random-walk term growing
  with √horizon, the trend fit's uncertainty at the target, and the
  binomial noise of the last read's counts.
- **Stale page** — if the tracked pool never moved during the whole
  history while the game is already out (the launch-night signature of a
  stale Exophase cache), the forecast assumes the pool jumps before the
  target and rates dilute toward earners ÷ a much larger pool, with a
  wide, downward-skewed spread. A trophy every tracked player already
  holds is exempt: a refresh cannot dilute what everyone earns at once.

All estimates are the program's own, computed from the case's data alone;
the uncertainty bands are deliberately honest, because the pinball loss
pays for overconfidence at every trophy outside the band.

## Layout

- `main.py` — the whole entry: case parsing, the two models, the spread.
- `LICENSE` — MIT.
