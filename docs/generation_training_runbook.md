# Generation Training Runbook

## Goal

Prepare and run the hairstyle-specialized generation LoRA on the current MirrAI stack:

- base model: `runwayml/stable-diffusion-inpainting`
- control: `lllyasviel/control_v11p_sd15_canny`
- identity conditioning: `IP-Adapter face`
- trainable weights: `UNet LoRA only`

This runbook assumes the hair segmentation model is already trained and frozen, and only the generation model is being optimized.

## Repository policy

This repository keeps code, configs, and runbooks only. The following are local-only artifacts and are intentionally not committed:

- `dataset_build/processed/...` manifests and intermediate data
- `output/training/...` and `output/benchmark/...`
- `sd_lora_final_bundle_*`
- `pretrained_models/generation_lora_*`
- generated benchmark reports and `.docx` deliverables

## Current resource fit

- available budget: about `$19.87`
- current serverless endpoint: `RTX 4090 24GB`
- current observed worker cost anchor: about `$0.59/hour`
- network volume: `50GB`

Serverless is fine for:

- smoke tests
- short benchmark inference
- final qualitative checks

Serverless is not appropriate for training because the current endpoint has a `10 minute` execution timeout. Long training should run on a `1x RTX 4090 Pod` with the same volume attached.

## RunPod safety settings

Use these settings when creating the training Pod:

- `GPU Pod`, not Serverless
- `1 x RTX 4090`
- `Secure` cloud
- data center must match the network volume
- attach the existing `50GB` network volume at `/runpod-volume`
- expose `22/tcp`
- optional `8888/http`
- disable any auto-stop or idle timeout policy on the Pod if the template exposes one

Why:

- `Secure` avoids spot-style interruption risk
- the network volume keeps checkpoints and logs safe even if the Pod is restarted
- detached launch plus checkpoint resume keeps training recoverable after SSH disconnects

## Dataset footprint

Local dataset size on disk:

- raw `CelebA-Dialog HQ`: about `2.96 GB`
- processed generation set: about `16.98 GB`
- combined local footprint: about `19.94 GB`

This fits inside the `50GB` volume, but keep at least `15GB+` free for:

- Hugging Face cache
- checkpoints
- benchmark predictions
- logs and TensorBoard data

Recommended layout on volume:

```text
/runpod-volume/
  hf-cache/
  hair_swap_generation/
    output/
  datasets/
    celeba_dialog_hq_generation/
```

## Training curriculum

### Stage 1

- manifest: `style_train_all.jsonl`
- objective: broad hairstyle adaptation without over-committing to noisy pseudo labels
- max steps: `2600`
- lr: `1e-4`
- rank: `16`
- batch: `1`
- grad accumulation: `4`

### Stage 2

- manifest: `style_train_highconf.jsonl`
- objective: sharpen style fidelity with higher-confidence labels only
- max steps: `1100`
- lr: `7e-5`
- init: best LoRA from stage 1

### Why this is overfit-safe

- only UNet LoRA is trainable
- base model, ControlNet, VAE, text encoder, image encoder stay frozen
- prompt dropout, face dropout, and control dropout are enabled
- mask jitter adds small geometry noise
- min-SNR weighting stabilizes the noise schedule
- weighted sampling uses confidence and curriculum metadata
- validation runs every `250` steps
- early stopping patience is `3`

## Expected runtime and budget

On a single `RTX 4090 24GB`, expect roughly:

- Stage 1: `3 to 4.5 hours`
- Stage 2: `1.2 to 2 hours`
- smoke eval: `10 to 20 minutes`

Conservative total:

- `4.5 to 6.75 hours`

Budget estimate using the current `$0.59/hour` anchor:

- about `$2.66 to $3.98`

Even with pod pricing above the current serverless rate, the remaining `$19.87` budget is still comfortable for one full training pass plus re-run margin.

## Run commands

### 1. Build benchmarks

```bash
python scripts/build_generation_benchmarks.py
```

Outputs:

- `smoke_eval_quick.jsonl`
- `recon_eval_stratified.jsonl`
- `edit_eval_requests.jsonl`

These benchmark manifests are expected to stay on your local workspace or attached volume, not in git.

### 2. Train stages and run smoke eval

```bash
bash scripts/run_generation_training.sh
```

Useful overrides:

```bash
DATA_ROOT=/runpod-volume/datasets/celeba_dialog_hq_generation \
WORK_ROOT=/runpod-volume/hair_swap_generation \
RUN_FULL_RECON_EVAL=1 \
bash scripts/run_generation_training.sh
```

### 3. Safe detached launch

Recommended for RunPod:

```bash
apt-get update && apt-get install -y git tmux

cd /workspace/hair_swap_model

DATA_ROOT=/runpod-volume/datasets/celeba_dialog_hq_generation \
WORK_ROOT=/runpod-volume/hair_swap_generation \
AUTO_RESUME=1 \
RUN_FULL_RECON_EVAL=0 \
bash scripts/start_generation_training_safe.sh
```

This launcher:

- starts training in a detached `tmux` session
- writes logs to `/runpod-volume/hair_swap_generation/logs`
- writes run metadata to `/runpod-volume/hair_swap_generation/state`
- preserves checkpoints on the network volume
- auto-resumes from the latest checkpoint if you relaunch

Checkpoints, logs, and evaluation outputs created here are local runtime artifacts and should not be committed back into this repository.

### 4. Status check

```bash
WORK_ROOT=/runpod-volume/hair_swap_generation \
bash scripts/check_generation_training_status.sh
```

### 5. Reattach to live session

```bash
tmux attach -t hairgen_train
```

### 6. Stop safely

```bash
WORK_ROOT=/runpod-volume/hair_swap_generation \
bash scripts/stop_generation_training.sh
```

## Benchmark strategy

### Smoke benchmark

- file: `smoke_eval_quick.jsonl`
- size: `12`
- purpose: cold-start sanity check, LoRA load check, basic identity preservation

### Recon benchmark

- file: `recon_eval_stratified.jsonl`
- size: `96`
- purpose: quantitative masked reconstruction evaluation

### Edit benchmark

- file: `edit_eval_requests.jsonl`
- size: `48`
- purpose: qualitative prompt-following review for hair-related requests

## Metrics to watch

Use `scripts/evaluate_generation_benchmark.py` after inference.

Primary:

- `hair_mae`
- `hair_psnr`
- `hair_ssim`

Preservation:

- `face_preserve_mae`
- `cloth_preserve_mae`
- `overall_mae`

## Stop and rollback rules

Stop the run early when any of the following happens:

- validation loss worsens for `3` checks in a row
- `hair_mae` improves but `face_preserve_mae` rises noticeably
- generated samples start repeating the same texture or fringe pattern across identities
- stage 2 looks worse than stage 1 on smoke images

Promote checkpoints in this order:

1. `stage1/best`
2. `stage2/best`
3. `stage2/final` only if validation stayed stable until the end

## Disconnect and recovery behavior

If your SSH session drops:

- training continues inside `tmux`
- logs continue writing to the network volume
- checkpoints stay under the stage output directories

If the Pod itself restarts:

- reconnect
- go back to `/workspace/hair_swap_model`
- run the same detached launch command again
- `AUTO_RESUME=1` will continue from the latest stage checkpoint automatically

## Post-train deployment check

The inference stack now supports runtime LoRA loading:

- `handler_sd.py` accepts `lora_path`
- `handler_sd.py` accepts `lora_scale`
- `pipeline_sd_inpainting.py` applies and swaps LoRA weights at runtime

That means the same serverless endpoint can be reused for:

- best checkpoint smoke tests
- A/B comparison between stage 1 and stage 2
- final acceptance review
