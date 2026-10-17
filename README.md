# tsanomaly

Point-anomaly detection for a single time series, written against the Python
standard library only. I built it in September 2026, with three other small
exercises (bayesab, minietl, gbdt), to understand the algorithms from the
inside before using statsmodels or scikit-learn for the same thing.

It decomposes a seasonal series the classical way (centered moving-average
trend, period-averaged seasonal component) and scores what is left over with
a z-score, a median/MAD modified z-score, or Tukey's IQR fences. There is a
causal rolling z-score for series with no fixed period, and a `demo` command
that injects a known set of anomalies into a synthetic series and reports
precision, recall and F1 against them, so the tool can be checked without a
real dataset on hand.

Python 3.9+, no runtime dependencies (`csv`, `statistics`, `math`, `random`,
`argparse`, `dataclasses`). Tests use `pytest`.

## Why decompose first

A plain z-score over a raw series with a daily or weekly cycle does not work
well: the seasonal swing itself looks like spread, so the threshold either
misses real anomalies or flags half the peaks and troughs. Subtracting the
trend and the typical shape of the cycle first leaves a residual that should
look like noise around zero when nothing unusual happened, which is what the
threshold detectors are built to catch.

## Run it

```bash
pip install -e ".[dev]"
```

### `detect`: decompose a CSV column and flag anomalies

```bash
tsanomaly detect --input readings.csv --value-col reading --timestamp-col hour \
  --period 24 --method zscore
```

Real output, run against a synthetic 5-day hourly temperature series (daily
sinusoidal cycle, Gaussian noise, one injected +12 spike at hour 74):

```
method=zscore points=120 anomalies=1
stats: {'mean': -0.007313585069445144, 'std': 1.1422518471888226, 'threshold': 3.0}
  [74] t=74 value=35.1790 residual=8.7531
```

`--format json` gives machine-readable output. `--trend-window N` instead of
`--period N` handles a series that drifts but has no seasonal cycle (plain
moving-average trend, no seasonal step). `--method {zscore,modified_zscore,iqr}`
picks the scorer; defaults are a z-score threshold of 3.0, a modified-z
threshold of 3.5 and an IQR fence multiplier of 1.5, overridable with
`--threshold` and `--k`.

### `rolling-detect`: no decomposition, streaming-style

```bash
tsanomaly rolling-detect --input readings.csv --value-col reading --window 20 --threshold 3.0
```

Each point is compared against the mean and std of the `window` points
strictly before it, never including itself, so it works online and needs no
period.

### `demo`: synthetic ground truth, scored

```bash
tsanomaly demo --length 200 --period 24 --anomalies 6 --seed 42 --method zscore
```

```
synthetic demo: 200 points, period=24, method=zscore
  injected anomalies : [46, 73, 93, 137, 170, 174]
  detected anomalies : [46, 73, 93, 137, 170, 174]
  precision=1.000 recall=1.000 f1=1.000  (tp=6 fp=0 fn=0)
```

## How it works

1. Trend: a centered moving average of length `period` (or `--trend-window`).
   Even windows use the usual "2xM" centering so the estimate lands on an
   integer index rather than a half-step.
2. Seasonal (period mode only): detrend, then average the detrended values at
   each position within the period (every hour-0, every hour-1, and so on),
   normalized so the seasonal indices sum to zero.
3. Residual: `value - trend - seasonal`, or `value - trend` in trend-only mode.
4. Detection on the residuals:
   - `zscore` flags `|residual - mean| / std > threshold`.
   - `modified_zscore` is Iglewicz and Hoaglin's median/MAD version, which
     holds up better when the anomalies are big enough to drag the mean and
     std around.
   - `iqr` is Tukey's fences, `[Q1 - k*IQR, Q3 + k*IQR]`, with
     linear-interpolation percentiles matching numpy's default.
   - `rolling_zscore` (the `rolling-detect` command) is a trailing-window
     z-score on the raw values, no decomposition.

## Known issues

Things I found by reading the code back and running it, not a wish list.

- The first and last `period // 2` points never get a residual, because the
  centered moving average needs a full window on both sides. With
  `--period 24` that is 12 points at each end that can never be flagged, and
  `--trend-window` has the same edge.
- Plain `zscore` masks itself once anomalies are dense: the spikes inflate the
  std they are measured against. `tsanomaly demo --length 500 --period 24
  --anomalies 20 --seed 7` gets recall 0.80 with `zscore` and 1.00 with
  `modified_zscore` or `iqr`. The default should probably be
  `modified_zscore`.
- The seasonal index at each position is a plain mean of the detrended
  values, so a large spike leaks a little into the seasonal estimate for its
  slot. R's `decompose()` has the same weakness; STL would not.
- When the MAD is zero (`modified_zscore`) or a rolling window has zero
  variance (`rolling-detect`), any point that differs from the median or mean
  at all is flagged, including floating-point noise.
- The CSV loader needs every cell in the value column to parse as a float; a
  blank cell is an error, not a missing value. Rows are taken in file order
  and timestamps are passed through as strings, never parsed or sorted.
- The `detect` output above came from a file I did not commit; the `demo`
  outputs are reproducible from the seed.
- Not implemented: multiplicative decomposition, STL, multivariate series,
  and change-point or level-shift detection. Only point anomalies.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

39 tests: the moving-average math (including a hand-verified even-window
centering case), decomposition on a noise-free synthetic signal, every
detector's edge cases (`None` skipping, zero-variance and zero-MAD series,
hand-computed IQR fences), the synthetic generator's determinism, and CLI
end-to-end runs including a subprocess-level `python -m tsanomaly.cli` check.

## License

MIT, see [LICENSE](LICENSE).
