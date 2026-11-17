# gta6-twitch-ccu-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/koala-builds/gta6-twitch-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/koala-builds/gta6-twitch-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

A quantile forecaster for a game launch's first day on Twitch. Given one case file (a launch's Twitch category: hourly history up to the checkpoint at day_start + 1h, the known hour-1 viewers, and the full launch-day curves of comparable past launches), it writes quantile forecasts (levels 0.05 / 0.25 / 0.5 / 0.75 / 0.95) for hours 2..24 of launch day.

The model transfers launch-day shapes from comparable launches. Each comparable's day is normalised by its own hour-1 print, and each comparable is weighted by how closely its pre-checkpoint trajectory (the 24 hourly log-ratios to its own hour-1 print: the rolling-unlock wave crossing New Zealand, Europe and the US east coast) matches this case's own pre-checkpoint trajectory, under a Gaussian kernel on the mean absolute log difference. Three views of that weighted neighbourhood are ensembled (a scale-aware weighted mean over the full pre-window, a median over the full pre-window, and a weighted mean over the last 13 pre-hours); the per-hour spread is the root-mean-square of the members' weighted log-space standard deviations, shrunk by 0.85. Quantiles are lognormal around the transferred shape, scaled by the case's known hour-1 print. Anything the model cannot handle falls back to the house persistence baseline, so the output is always exactly 23 valid rows.

Backtested on the competition kit's four practice cases (pinball loss on log scale, lower is better; actuals are the same SullyGnome hourly series the house recorded): pooled 0.230 against the starter baseline's 0.387, and better than the baseline on every case (p00 0.336 vs 0.744, p01 0.181 vs 0.287, p02 0.168 vs 0.215, p03 0.237 vs 0.303). Deterministic, Python 3.13 standard library only, no network.

## Run it

```sh
python3 main.py
```

The program reads every `*.in` case file in `$MAKEFUN_INPUT_DIR` (default `/input`) and writes one `<id>.out` file per case to `$MAKEFUN_OUTPUT_DIR` (default `/output`).

---

<sub>Entered on [make.fun](https://make.fun/?ref=koala-builds&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
