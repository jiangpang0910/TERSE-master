# K-Fold Plan (MLE Perspective) for This Repo

I reviewed `trainers/train.py` and `trainers/test.py` and mapped where K-Fold should be introduced.

Important context from current code:
- `train.py` already imports `KFold` but does not use it yet.
- Training currently loops as: `scenario -> run -> load_data -> train_model -> save checkpoint -> compute metrics`.
- Testing currently loads one checkpoint per `scenario/run` and evaluates on `trg_test_dl`.

The key design decision is: **K-Fold should be inside each scenario/run training cycle**, not outside scenario selection.

---

## 1) Where K-Fold should go

### Correct place in `train.py` flow
Inside this block:
- For each `(src_id, trg_id)` scenario
- For each `run_id`
- After `load_data(src_id, trg_id)` and before `train_model()`

Why:
- A scenario is the domain-adaptation unit (`src -> trg`), so folds must be created per scenario to avoid mixing scenario definitions.
- `run_id` is currently used for randomness/reproducibility; folds should be nested under that.
- If K-Fold is put outside scenario loop, you risk data leakage and invalid domain split semantics.

### Correct place in `test.py` flow
Inside the same scenario/run loop where checkpoints are loaded:
- add an inner loop over folds
- load checkpoint for each fold
- evaluate fold checkpoint
- aggregate fold metrics per scenario/run, then across runs

Why:
- Train and test indexing must match (`scenario/run/fold`).
- Fold-level evaluation lets you report both variability and mean performance.

---

## 2) MLE-style objective and constraints

Before adding K-Fold, define what K-Fold is validating in this DA setup:

1. **No leakage into target test set**  
   `trg_test_dl` remains untouched and never used for fold splitting.

2. **Split only training side**  
   K-Fold should split the trainable subset (typically source training data, or source+target-train if algorithm explicitly uses unlabeled target train data in adaptation).

3. **Keep scenario meaning intact**  
   Folding should not alter `src_id` and `trg_id` domain identity.

4. **Reproducibility**  
   Fold creation should be deterministic per `(scenario, run_id, fold_id)` with controlled seed behavior.

---

## 3) Recommended implementation steps (no code, just plan)

### Step A - Confirm foldable dataset unit
Inspect what `load_data()` constructs in `AbstractTrainer`:
- Identify the dataset object currently feeding `src_train_dl` (and any adaptation loaders).
- Confirm whether you can index samples directly (needed for KFold indices).

Deliverable:
- A clear statement like: "KFold is applied to source training dataset indices."

### Step B - Add K-Fold experiment config
Add conceptual config knobs:
- `use_kfold: bool`
- `num_folds: int` (e.g., 5)
- `kfold_shuffle: bool` (usually true)
- `kfold_seed: int`

Rule:
- If `use_kfold=False`, preserve current behavior exactly.

### Step C - Build fold loop in training
Within each scenario/run:
1. Generate fold splits from chosen dataset indices.
2. For each `fold_id`:
   - Build fold-specific train/val loaders.
   - Train model using fold train split.
   - Validate/log fold metrics.
   - Save fold-specific checkpoint.

Checkpoint/log naming should include fold, for example:
- `.../{src}_to_{trg}_run_{run}_fold_{fold}/checkpoint.pt`

### Step D - Decide what "best model" means under K-Fold
You need one of these policies:
1. **Per-fold best**: keep best checkpoint per fold (most common).
2. **Global best across folds**: less common; can bias reporting.
3. **Fold ensemble at test**: average predictions from all fold models.

Recommended first version:
- Save and test per-fold best, then average metrics.

### Step E - Update testing logic to consume fold checkpoints
In `test.py`:
- For each scenario/run, iterate folds, load each fold checkpoint, evaluate.
- Aggregate:
  - per-run fold mean/std
  - per-scenario mean/std across runs

This keeps statistical reporting clean and comparable to your current tables.

### Step F - Reporting and tables
Add a fold-aware reporting layout:
- Row granularity options:
  - `(scenario, run, fold, metrics...)` raw rows
  - plus aggregated rows `(scenario, run, fold_mean, fold_std)`
  - plus final `(scenario, all_runs_mean, all_runs_std)`

Do not remove current summary style; extend it.

### Step G - Sanity checks before trusting results
Run these checks:
1. Fold splits are disjoint and cover full train set.
2. Target test set size is constant across folds.
3. Metrics are stable when rerunning same seed.
4. Runtime increase ~= `num_folds` multiplier (expected).

---

## 4) Practical MLE decisions you should make explicitly

1. **KFold vs StratifiedKFold**  
   - Classification with class imbalance: prefer stratified splitting.  
   - Regression: standard KFold is fine.

2. **When adaptation uses target-train data**  
   - Keep target-test isolated.
   - Decide whether target-train stays fixed across folds or is also fold-partitioned (usually fixed for UDA protocols).

3. **Compute budget**  
   Full cost becomes: `num_scenarios * num_runs * num_folds`.
   Consider reducing `num_runs` when `num_folds` is high.

4. **Comparison fairness**  
   Compare baseline and K-Fold under equal total training budget if you want strict fairness.

---

## 5) Minimal migration strategy (lowest risk)

1. Add K-Fold as optional (`use_kfold=False` default).
2. First enable on one scenario and one run only.
3. Verify checkpoint paths and table formats.
4. Then scale to all scenarios/runs.
5. Keep old single-split path available for quick debugging.

---

## 6) What success looks like

You know K-Fold is correctly integrated when:
- Each scenario/run produces multiple fold checkpoints.
- `test.py` can evaluate all folds without path hacks.
- Final reports include fold-level variability, not just run-level variability.
- No evidence of target-test leakage.

---

## 7) Quick mental model

Current:
- `scenario -> run -> one training split -> one checkpoint`

Target (with K-Fold):
- `scenario -> run -> fold_1..K -> K checkpoints -> aggregated metrics`

That is the cleanest way to adapt your current architecture without breaking the training/testing contract.