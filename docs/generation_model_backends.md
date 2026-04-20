# Generation Model Backends

The hair inpainting pipeline now separates generation backends from mask,
prompt, face-protection, and compositing logic.

## Backends

| key | model | default canvas | note |
| --- | --- | ---: | --- |
| `sd15_controlnet` | `runwayml/stable-diffusion-inpainting` + `lllyasviel/control_v11p_sd15_canny` | 512 | Existing baseline with IP-Adapter and runtime LoRA |
| `sdxl_inpaint` | `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` | 1024 | SDXL inpainting candidate |
| `flux_fill` | `black-forest-labs/FLUX.1-Fill-dev` | 1024 | FLUX.1 Fill candidate; ignores negative prompt and strength |
| `powerpaint` | `Sanster/PowerPaint-V1-stable-diffusion-inpainting` | 512 | Diffusers-compatible PowerPaint candidate |

## RunPod Request

```json
{
  "input": {
    "image": "<base64>",
    "hairstyle_text": "short bob cut, natural salon hair",
    "color_text": "brown",
    "generation_backend": "sdxl_inpaint",
    "top_k": 1,
    "return_base64": true
  }
}
```

## Matrix Test

```bash
python scripts/run_runpod_inpaint_backend_matrix.py \
  --backends sdxl_inpaint,flux_fill,powerpaint \
  --images-dir images \
  --top-k 1
```

Outputs are saved under `output/runpod_inpaint_backend_matrix/`, grouped by
backend, source image, and style prompt.
