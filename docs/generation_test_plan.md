# Generation Test Plan

## Scope

This test plan validates the hairstyle generation LoRA after each training stage.

The manifests, predictions, and reports referenced below are generated locally and are not committed to this repository.

## Test datasets

The paths below refer to local benchmark manifests produced by `scripts/build_generation_benchmarks.py`.

- `dataset_build/processed/celeba_dialog_hq_generation/benchmarks/smoke_eval_quick.jsonl`
  - `12` samples
  - fast sanity check
- `dataset_build/processed/celeba_dialog_hq_generation/benchmarks/recon_eval_stratified.jsonl`
  - `96` samples
  - quantitative reconstruction benchmark
- `dataset_build/processed/celeba_dialog_hq_generation/benchmarks/edit_eval_requests.jsonl`
  - `48` samples
  - qualitative prompt-following benchmark

## Test procedure

### 1. Smoke test

Run on every new checkpoint:

```bash
python scripts/run_generation_benchmark.py \
  --benchmark-manifest dataset_build/processed/celeba_dialog_hq_generation/benchmarks/smoke_eval_quick.jsonl \
  --output-dir output/benchmark/smoke_stage_check \
  --lora-path <checkpoint_dir>

python scripts/evaluate_generation_benchmark.py \
  --predictions-manifest output/benchmark/smoke_stage_check/predictions.jsonl \
  --report-dir output/benchmark/smoke_stage_check/report
```

Pass criteria:

- no inference crash
- no blank or heavily corrupted result
- face and cloth remain structurally intact
- hairstyle direction changes are visible

### 2. Recon test

Run for checkpoints that pass smoke:

```bash
python scripts/run_generation_benchmark.py \
  --benchmark-manifest dataset_build/processed/celeba_dialog_hq_generation/benchmarks/recon_eval_stratified.jsonl \
  --output-dir output/benchmark/recon_stage_check \
  --lora-path <checkpoint_dir>

python scripts/evaluate_generation_benchmark.py \
  --predictions-manifest output/benchmark/recon_stage_check/predictions.jsonl \
  --report-dir output/benchmark/recon_stage_check/report
```

Primary metrics:

- `hair_mae`
- `hair_psnr`
- `hair_ssim`

Preservation metrics:

- `face_preserve_mae`
- `cloth_preserve_mae`

### 3. Edit test

Run for checkpoints that pass recon:

```bash
python scripts/run_generation_benchmark.py \
  --benchmark-manifest dataset_build/processed/celeba_dialog_hq_generation/benchmarks/edit_eval_requests.jsonl \
  --output-dir output/benchmark/edit_stage_check \
  --lora-path <checkpoint_dir>
```

Manual review checklist:

- prompt-following on bangs and silhouette
- color consistency
- identity preservation
- unnatural forehead exposure or hairline collapse
- repeated memorized fringe pattern across identities

## Overfit alarm rules

Treat any checkpoint as suspicious when:

- `hair_mae` improves but `face_preserve_mae` clearly worsens
- smoke images begin to share the same texture or fringe shape
- stage 2 outputs are less stable than stage 1 on the same samples
- prompt-following improves only for one narrow style family

## Promotion rule

Promote checkpoints in this order:

1. `stage1/best`
2. `stage2/best` if both recon and edit quality improve
3. never promote `final` automatically without smoke and recon confirmation
