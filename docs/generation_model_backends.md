# Generation Model Backends

The hair inpainting pipeline now separates generation backends from mask,
prompt, face-protection, and compositing logic.

## Backends

| key | model | default canvas | note |
| --- | --- | ---: | --- |
| `sd15_controlnet` | `runwayml/stable-diffusion-inpainting` + `lllyasviel/control_v11p_sd15_canny` | 512 | Baseline with IP-Adapter and runtime LoRA |

## RunPod Request

```json
{
  "input": {
    "image": "<base64>",
    "hairstyle_text": "short bob cut, natural salon hair",
    "color_text": "brown",
    "generation_backend": "sd15_controlnet",
    "top_k": 1,
    "return_base64": true
  }
}
```

## Matrix Test

```bash
python scripts/run_runpod_inpaint_backend_matrix.py \
  --backends sd15_controlnet \
  --images-dir images \
  --top-k 1
```

Outputs are saved under `output/runpod_inpaint_backend_matrix/`, grouped by
backend, source image, and style prompt.
