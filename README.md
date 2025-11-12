# Horse Quant Project (HQP)

_Reproducible horse‑racing probability modelling & backtesting pipeline_

> **Pipeline**: `ingest → validate → split → ratings → features → train/evaluate → calibration → market (join) → edges → backtest`

---

## Highlights
- **End‑to‑end CLI** (Typer) with auditable, timestamped artifacts under `reports/` and `models/`.
- **Leakage‑aware**: race‑grouped splits, chronological ordering; market probabilities normalised per race.
- **Flexible interop**: commands accept file paths _or_ DataFrames (inside library code).
- **Calibration‑first**: Platt / Isotonic post‑hoc options with before/after Brier diagnostics.
- **Backtesting realism**: edge thresholds, odds caps, portfolio limits, Kelly/flat staking.

---

## Quickstart

### 1) Install
```bash
# Requires Python 3.11+
pipx install poetry  # or: pip install --user poetry
poetry install --with dev
poetry run hqp --help
```

### 2) Smoke test (no external data)
```bash
# Create a tiny synthetic dataset and run the full pipeline in quick mode
poetry run hqp ingest --dry-run --out data/interim/base.parquet
poetry run hqp run-all --quick
```
This will produce timestamped outputs under `reports/` and `models/` plus sample backtest results.

### 3) Typical full run (with your configs)
```bash
poetry run hqp ingest --config configs/ingest.yaml --out data/interim/base.parquet
poetry run hqp validate --base data/interim/base.parquet --schema configs/schema.yaml
poetry run hqp split --base data/interim/base.parquet --config configs/split.yaml
poetry run hqp ratings --base data/interim/base.parquet --out data/ratings/ratings.parquet
poetry run hqp features --base data/interim/base.parquet \
  --ratings data/ratings/ratings.parquet --out data/features/features.parquet
poetry run hqp train --features data/features/features.parquet --config configs/model_lgbm.yaml
poetry run hqp evaluate-model --run-dir models/artifacts/<RUN_TS> --features data/features/features.parquet
poetry run hqp odds --base data/interim/base.parquet --out data/market/odds.parquet
poetry run hqp market --features data/features/features.parquet --odds-path data/market/odds.parquet \
  --out data/market/market_join.parquet
poetry run hqp edge --market data/market/market_join.parquet --run-dir models/artifacts/<RUN_TS> \
  --out data/market/edges.parquet
poetry run hqp backtest --edges data/market/edges.parquet --config configs/backtest.yaml
```

> Or one‑shot: `poetry run hqp run-all` (uses sensible defaults; `--quick` for tiny smoke run).

---

## Data expectations
**Keys** (must exist by the time you train/backtest):
- `race_id`, `horse_id` (string‑casted internally for safe joins)
- `race_datetime` (or `race_dt` / `off_time`), and label column (`obs__is_winner` / `won` / `is_winner`)

**Odds** (for market join):
- Decimal candidates searched (first present wins): `decimal_odds`, `mkt_odds_decimal`, `odds_decimal`, `sp_decimal`, `isp_decimal`, `bsp_decimal`, `obs__bsp`, `bsp`, `betfair_sp`, `bfsp`, `sp`, `ltp_5min`.
- Fractional candidates: `fractional_odds`, `odds_fractional`, `sp_fractional`, `sp_frac`, `fractional`, `obs__sp_frac`, `obs__fractional_odds`.

**Normalisation math** (per race):
- Raw implied from decimal odds: `p_i = 1 / odds_i`.
- Overround removed by racewise normalisation: `p̂_i = p_i / Σ_j p_j` so `Σ_i p̂_i = 1`.
- **Edge** definition: `edge = model_prob − p̂_i` (same race & runner).

---

## CLI Reference (selected)
Run `poetry run hqp COMMAND --help` for all options.

### Ingest
```bash
poetry run hqp ingest --config configs/ingest.yaml --out data/interim/base.parquet
# or: --dry-run to emit a minimal synthetic base
```
Writes the canonical runner‑level parquet.

### Validate
```bash
poetry run hqp validate --base data/interim/base.parquet --schema configs/schema.yaml
```
Runs structural/semantic checks; writes JSON under `reports/validate/<ts>/`.

### Split (chronological, race‑grouped)
```bash
poetry run hqp split --base data/interim/base.parquet --config configs/split.yaml
```
Keeps all runners of a race together; prevents future leakage.

### Ratings → Features
```bash
poetry run hqp ratings --base data/interim/base.parquet --out data/ratings/ratings.parquet
poetry run hqp features --base data/interim/base.parquet --ratings data/ratings/ratings.parquet \
  --out data/features/features.parquet
```
Ratings: tries your `hqp.ratings.*` modules, falls back to **Empirical‑Bayes** priors (time‑safe). Features: calls `hqp.features.build.*`.

### Train / Model
```bash
poetry run hqp train --features data/features/features.parquet --config configs/model_lgbm.yaml
# alias: poetry run hqp model ...
```
- Auto‑clamps `training.n_splits` to available groups to avoid `n_splits > n_groups`.
- Artifacts under `models/artifacts/<ts>/` (model, feature importances when available).

### Evaluate model
```bash
poetry run hqp evaluate-model --run-dir models/artifacts/<RUN_TS> --features data/features/features.parquet
```
Writes diagnostics to `reports/model/<ts>/`.

### Market odds & join
```bash
poetry run hqp odds --base data/interim/base.parquet --out data/market/odds.parquet
poetry run hqp market --features data/features/features.parquet --odds-path data/market/odds.parquet \
  --out data/market/market_join.parquet --decimal-col decimal_odds --prefix mkt
poetry run hqp evaluate-market --market data/market/market_join.parquet
```
Normalises implied probabilities within race; strict hygiene on joins.

### Edges & Backtest
```bash
poetry run hqp edge --market data/market/market_join.parquet \
  --run-dir models/artifacts/<RUN_TS> --out data/market/edges.parquet
poetry run hqp backtest --edges data/market/edges.parquet --config configs/backtest.yaml
poetry run hqp init-backtest-configs  # writes baseline/relaxed templates
poetry run hqp backtest-compare --edges data/market/edges.parquet
poetry run hqp backtest-sweep --edges data/market/edges.parquet \
  --edge-min 0.02 --edge-max 0.20 --edge-step 0.02 --odds-grid "8,10,12,15,20"
```
Backtest prints a concise one‑liner from `summary.json` (ROI, hit rate, avg odds, etc.).

### Calibration flow (post‑hoc)
```bash
# 1) EB baseline predictions (from market join)
poetry run hqp predict-eb --market data/market/market_join.parquet --out reports/model/<TS>/pred_eb.parquet

# 2) Fit calibrator on predictions (auto‑merges labels if missing)
poetry run hqp fit-calibrator --pred reports/model/<TS>/pred_eb.parquet \
  --out models/calibration/<TS>.pkl --method platt  # or: isotonic

# 3) Apply calibrator (optionally keep both columns)
poetry run hqp apply-calibrator --pred reports/model/<TS>/pred_eb.parquet \
  --calibrator models/calibration/<TS>.pkl --out reports/model/<TS>/pred_eb_cal.parquet \
  --col-in model_prob --col-out model_prob_cal

# 4) Build edges from any predictions parquet
poetry run hqp edge-from-pred --market data/market/market_join.parquet \
  --pred reports/model/<TS>/pred_eb_cal.parquet --probs-col model_prob_cal \
  --out data/market/edges_from_pred.parquet

# 5) Or backtest directly from predictions in one step
poetry run hqp backtest-from-pred --market data/market/market_join.parquet \
  --pred reports/model/<TS>/pred_eb_cal.parquet --probs-col model_prob_cal \
  --config configs/backtest.yaml
```

### One‑shot report
```bash
poetry run hqp report --run-dir models/artifacts/<RUN_TS> --out reports/$(date +%Y%m%d_%H%M%S)/report.md
```

### One‑command pipeline
```bash
poetry run hqp run-all
# add --quick for stub configs & faster dry‑runs
```
This also demonstrates a calibrated EB strategy end‑to‑end (predict → calibrate → edges → backtest).

---

## Configuration
All major stages read YAML configs. Examples:
- `configs/model_lgbm.yaml` / `configs/model_lgbm_tiny.yaml`
- `configs/split.yaml`, `configs/backtest.yaml`
- `poetry run hqp init-backtest-configs` writes:
  - `configs/backtest_baseline.yaml`
  - `configs/backtest_relaxed.yaml`

> The trainer auto‑adjusts `training.n_splits` downwards when unique groups are scarce (e.g., tiny demos) and writes a suffixed `*.autoadj.yaml` for auditability.

---

## Reproducibility & Safety
- **Determinism**: seeds pinned in configs; stable sorts for time order; group‑aware CV.
- **No silent fallbacks**: commands raise explicit Typer errors where correctness matters (keys, columns, odds parsing).
- **Artifact policy**: each run writes to timestamped folders via `YYYYMMDD_HHMMSS` helper.

---

## Directory layout (typical)
```
configs/
  ingest.yaml
  schema.yaml
  split.yaml
  model_lgbm.yaml
  backtest.yaml
  backtest_baseline.yaml
  backtest_relaxed.yaml
data/
  interim/base.parquet
  ratings/ratings.parquet
  features/features.parquet
  market/
    odds.parquet
    market_join.parquet
    edges.parquet
models/
  artifacts/<TS>/
  calibration/<TS>.pkl
reports/
  validate/<TS>/
  model/<TS>/
  backtest/<TS>/
  <TS>/report.md
```

---

## Model & evaluation notes
- **Calibration**: `calibration` command merges labels if needed; outputs reliability diagrams & Brier deltas.
- **Market sanity**: `evaluate-market` inspects overround/coverage.
- **Leakage control**: race‑grouped splits, chronological processing; joins only by `(race_id, horse_id)`.

---

## Troubleshooting
- _“No recognizable odds column found”_: rebuild with `hqp odds` and check column candidates above.
- _Joined implied probs all NaN_: verify `(race_id, horse_id)` types match; CLI casts to string on join inputs.
- _`n_splits > n_groups`_: trainer will auto‑clamp and write `*.autoadj.yaml`; increase data or lower `n_splits` manually.
- _Predictions missing `won`_: provide `--labels` parquet or ensure base/features include one of `won/obs__is_winner/is_winner`.

---

## Contributing
PRs and issues welcome. Please run:
```bash
poetry run ruff check . && poetry run mypy . && poetry run pytest -q
```
Add/refresh unit tests for new logic; keep CLI side‑effects auditable.

---

## License
Released under the MIT License — free for personal, academic, and commercial use.

---

## Citation
If you use HQP in academic or blog posts, it would be appreciated if you could link back to this repository and cite the pipeline components you rely on (calibration, backtesting, etc.).

