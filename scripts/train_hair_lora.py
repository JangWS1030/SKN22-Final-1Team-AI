from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionControlNetInpaintPipeline,
    StableDiffusionPipeline,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, compute_snr
from diffusers.utils import convert_unet_state_dict_to_peft
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPImageProcessor, CLIPTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.manifest_paths import normalize_manifest_rows


logger = get_logger(__name__, log_level="INFO")


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return normalize_manifest_rows(rows, manifest_path=path)


def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path, default=None)
    known, _ = bootstrap.parse_known_args()

    defaults: Dict[str, object] = {}
    if known.config:
        defaults = json.loads(known.config.read_text(encoding="utf-8"))

    parser = argparse.ArgumentParser(description="Train a hairstyle-specialized inpainting LoRA.")
    parser.add_argument("--config", type=Path, default=known.config)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--validation-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--pretrained-model-name-or-path", type=str, default=None)
    parser.add_argument("--controlnet-model-name-or-path", type=str, default=None)
    parser.add_argument("--ip-adapter-repo-id", type=str, default="h94/IP-Adapter")
    parser.add_argument("--ip-adapter-weight-name", type=str, default="ip-adapter-plus-face_sd15.bin")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--max-train-steps", type=int, default=2000)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--validation-max-samples", type=int, default=96)
    parser.add_argument("--validation-steps", type=int, default=250)
    parser.add_argument("--checkpointing-steps", type=int, default=250)
    parser.add_argument("--checkpoints-total-limit", type=int, default=3)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--initial-lora-path", type=Path, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lr-scheduler", type=str, default="cosine")
    parser.add_argument("--lr-warmup-steps", type=int, default=100)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-weight-decay", type=float, default=1e-2)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--report-to", type=str, default="tensorboard")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--enable-xformers-memory-efficient-attention", action="store_true")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--use-8bit-adam", action="store_true")
    parser.add_argument("--snr-gamma", type=float, default=5.0)
    parser.add_argument("--noise-offset", type=float, default=0.05)
    parser.add_argument("--prediction-type", type=str, default=None)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.08)
    parser.add_argument("--controlnet-conditioning-scale", type=float, default=0.25)
    parser.add_argument("--hair-loss-weight", type=float, default=4.0)
    parser.add_argument("--prompt-dropout-p", type=float, default=0.10)
    parser.add_argument("--face-dropout-p", type=float, default=0.10)
    parser.add_argument("--control-dropout-p", type=float, default=0.05)
    parser.add_argument("--mask-jitter-px", type=int, default=6)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--default-prompt", type=str, default="portrait photo, realistic hairstyle, preserved identity")
    parser.add_argument("--ip-adapter-enabled", type=lambda value: str(value).lower() in {"1", "true", "yes"}, default=True)
    parser.add_argument("--local-rank", type=int, default=-1)
    if defaults:
        parser.set_defaults(**defaults)
    args = parser.parse_args()
    if defaults:
        for key, value in defaults.items():
            if hasattr(args, key) and getattr(args, key) is None:
                setattr(args, key, value)

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    missing_required = [
        flag
        for attr_name, flag in [
            ("manifest", "--manifest"),
            ("output_dir", "--output-dir"),
            ("pretrained_model_name_or_path", "--pretrained-model-name-or-path"),
            ("controlnet_model_name_or_path", "--controlnet-model-name-or-path"),
        ]
        if getattr(args, attr_name, None) is None
    ]
    if missing_required:
        parser.error(f"the following arguments are required: {', '.join(missing_required)}")
    return args


def pil_to_normalized_tensor(image: Image.Image) -> torch.Tensor:
    array = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return tensor * 2.0 - 1.0


def pil_to_control_tensor(image: Image.Image) -> torch.Tensor:
    array = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def pil_to_mask_tensor(image: Image.Image) -> torch.Tensor:
    array = (np.array(image.convert("L"), dtype=np.uint8) > 127).astype(np.float32)
    return torch.from_numpy(array).unsqueeze(0)


def apply_mask_jitter(mask_image: Image.Image, jitter_px: int) -> Image.Image:
    if jitter_px <= 0:
        return mask_image
    mask = (np.array(mask_image.convert("L"), dtype=np.uint8) > 127).astype(np.uint8)
    if mask.max() == 0:
        return mask_image
    shift = random.randint(-jitter_px, jitter_px)
    if shift == 0:
        return mask_image
    kernel_size = max(3, abs(shift) * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    jittered = cv2.dilate(mask, kernel, iterations=1) if shift > 0 else cv2.erode(mask, kernel, iterations=1)
    return Image.fromarray((jittered * 255).astype(np.uint8), mode="L")


class HairInpaintDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        rows: List[Dict],
        *,
        tokenizer: CLIPTokenizer,
        face_processor: CLIPImageProcessor,
        resolution: int,
        default_prompt: str,
        prompt_dropout_p: float,
        mask_jitter_px: int,
        training: bool,
    ) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.face_processor = face_processor
        self.resolution = resolution
        self.default_prompt = default_prompt
        self.prompt_dropout_p = prompt_dropout_p if training else 0.0
        self.mask_jitter_px = mask_jitter_px if training else 0

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.rows[index]
        prompt = str(
            row.get("train_prompt")
            or row.get("caption_target_hair_enriched")
            or row.get("caption_target")
            or self.default_prompt
        ).strip()
        if self.prompt_dropout_p > 0 and random.random() < self.prompt_dropout_p:
            prompt = self.default_prompt

        target = Image.open(row["target_image_path"]).convert("RGB").resize((self.resolution, self.resolution), Image.Resampling.LANCZOS)
        source = Image.open(row["source_image_path"]).convert("RGB").resize((self.resolution, self.resolution), Image.Resampling.LANCZOS)
        mask = Image.open(row["mask_path"]).convert("L").resize((self.resolution, self.resolution), Image.Resampling.NEAREST)
        control = Image.open(row["control_image_path"]).convert("RGB").resize((self.resolution, self.resolution), Image.Resampling.LANCZOS)
        face = Image.open(row["face_crop_path"]).convert("RGB")
        if self.mask_jitter_px > 0:
            mask = apply_mask_jitter(mask, self.mask_jitter_px)

        tokenized = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        face_pixel_values = self.face_processor(images=face, return_tensors="pt").pixel_values[0]

        return {
            "sample_id": row["sample_id"],
            "pixel_values": pil_to_normalized_tensor(target),
            "conditioning_values": pil_to_normalized_tensor(source),
            "mask_values": pil_to_mask_tensor(mask),
            "control_values": pil_to_control_tensor(control),
            "face_pixel_values": face_pixel_values,
            "input_ids": tokenized.input_ids[0],
            "sample_weight": float(row.get("sample_weight") or 1.0),
        }


def collate_fn(examples: List[Dict[str, object]]) -> Dict[str, object]:
    return {
        "sample_ids": [example["sample_id"] for example in examples],
        "pixel_values": torch.stack([example["pixel_values"] for example in examples]).float(),
        "conditioning_values": torch.stack([example["conditioning_values"] for example in examples]).float(),
        "mask_values": torch.stack([example["mask_values"] for example in examples]).float(),
        "control_values": torch.stack([example["control_values"] for example in examples]).float(),
        "face_pixel_values": torch.stack([example["face_pixel_values"] for example in examples]).float(),
        "input_ids": torch.stack([example["input_ids"] for example in examples]),
        "sample_weight": torch.tensor([example["sample_weight"] for example in examples], dtype=torch.float32),
    }


@dataclass
class TrainComponents:
    pipe: StableDiffusionControlNetInpaintPipeline
    noise_scheduler: DDPMScheduler
    tokenizer: CLIPTokenizer
    face_processor: CLIPImageProcessor
    unet: torch.nn.Module
    vae: torch.nn.Module
    text_encoder: torch.nn.Module
    controlnet: torch.nn.Module
    weight_dtype: torch.dtype


def unwrap_model(accelerator: Accelerator, model: torch.nn.Module) -> torch.nn.Module:
    model = accelerator.unwrap_model(model)
    return model._orig_mod if is_compiled_module(model) else model


def load_initial_lora(unet: torch.nn.Module, input_dir: Path) -> None:
    lora_state_dict, _ = StableDiffusionPipeline.lora_state_dict(str(input_dir))
    unet_state_dict = {k.replace("unet.", ""): v for k, v in lora_state_dict.items() if k.startswith("unet.")}
    set_peft_model_state_dict(unet, convert_unet_state_dict_to_peft(unet_state_dict), adapter_name="default")


def save_lora_weights(output_dir: Path, unet: torch.nn.Module) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    StableDiffusionPipeline.save_lora_weights(
        save_directory=str(output_dir),
        unet_lora_layers=get_peft_model_state_dict(unet),
        safe_serialization=True,
    )


def broadcast_mask(mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    while mask.ndim < like.ndim:
        mask = mask.unsqueeze(-1)
    return mask.to(dtype=like.dtype)


def compute_batch_loss(
    batch: Dict[str, object],
    *,
    args: argparse.Namespace,
    accelerator: Accelerator,
    components: TrainComponents,
    training: bool,
) -> torch.Tensor:
    device = accelerator.device
    weight_dtype = components.weight_dtype
    pixel_values = batch["pixel_values"].to(device=device, dtype=weight_dtype)
    conditioning_values = batch["conditioning_values"].to(device=device, dtype=weight_dtype)
    mask_values = batch["mask_values"].to(device=device, dtype=torch.float32)
    control_values = batch["control_values"].to(device=device, dtype=weight_dtype)
    face_pixel_values = batch["face_pixel_values"].to(device=device, dtype=weight_dtype)
    input_ids = batch["input_ids"].to(device=device)
    sample_weight = batch["sample_weight"].to(device=device, dtype=torch.float32)

    if training and args.control_dropout_p > 0:
        drop = (torch.rand(control_values.shape[0], device=device) < args.control_dropout_p).to(dtype=control_values.dtype)
        control_values = control_values * (1.0 - drop).view(-1, 1, 1, 1)

    masked_image = conditioning_values * (mask_values < 0.5).to(dtype=weight_dtype)
    latents = components.vae.encode(pixel_values).latent_dist.sample()
    latents = latents * components.vae.config.scaling_factor
    masked_image_latents = components.vae.encode(masked_image).latent_dist.sample()
    masked_image_latents = masked_image_latents * components.vae.config.scaling_factor

    noise = torch.randn_like(latents)
    if args.noise_offset > 0:
        noise = noise + args.noise_offset * torch.randn((latents.shape[0], latents.shape[1], 1, 1), device=device, dtype=latents.dtype)

    timesteps = torch.randint(0, components.noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
    noisy_latents = components.noise_scheduler.add_noise(latents, noise, timesteps)
    encoder_hidden_states = components.text_encoder(input_ids, return_dict=False)[0]

    if args.prediction_type is not None:
        components.noise_scheduler.register_to_config(prediction_type=args.prediction_type)
    if components.noise_scheduler.config.prediction_type == "epsilon":
        target = noise
    elif components.noise_scheduler.config.prediction_type == "v_prediction":
        target = components.noise_scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(f"Unknown prediction type {components.noise_scheduler.config.prediction_type}")

    latent_model_input = components.noise_scheduler.scale_model_input(noisy_latents, timesteps)
    down_block_res_samples, mid_block_res_sample = components.controlnet(
        latent_model_input,
        timesteps,
        encoder_hidden_states=encoder_hidden_states,
        controlnet_cond=control_values,
        conditioning_scale=float(args.controlnet_conditioning_scale),
        return_dict=False,
    )

    latent_mask = F.interpolate(mask_values, size=latents.shape[-2:], mode="nearest").to(dtype=weight_dtype)
    latent_model_input = torch.cat([latent_model_input, latent_mask, masked_image_latents], dim=1)

    added_cond_kwargs = None
    if args.ip_adapter_enabled:
        image_embeds = components.pipe.prepare_ip_adapter_image_embeds(face_pixel_values, None, device, latents.shape[0], False)
        if training and args.face_dropout_p > 0:
            drop = (torch.rand(latents.shape[0], device=device) < args.face_dropout_p).float()
            image_embeds = [embed * (1.0 - broadcast_mask(drop, embed)) for embed in image_embeds]
        added_cond_kwargs = {"image_embeds": image_embeds}

    model_pred = components.unet(
        latent_model_input,
        timesteps,
        encoder_hidden_states=encoder_hidden_states,
        down_block_additional_residuals=down_block_res_samples,
        mid_block_additional_residual=mid_block_res_sample,
        added_cond_kwargs=added_cond_kwargs,
        return_dict=False,
    )[0]

    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
    hair_weight = 1.0 + F.interpolate(mask_values, size=loss.shape[-2:], mode="nearest") * (args.hair_loss_weight - 1.0)
    loss = (loss * hair_weight).mean(dim=(1, 2, 3))

    if args.snr_gamma is not None:
        snr = compute_snr(components.noise_scheduler, timesteps)
        snr_weight = torch.stack([snr, args.snr_gamma * torch.ones_like(snr)], dim=1).min(dim=1)[0]
        if components.noise_scheduler.config.prediction_type == "epsilon":
            snr_weight = snr_weight / snr
        else:
            snr_weight = snr_weight / (snr + 1)
        loss = loss * snr_weight

    return (loss * sample_weight).mean()


def evaluate_validation(
    *,
    accelerator: Accelerator,
    args: argparse.Namespace,
    components: TrainComponents,
    dataloader: torch.utils.data.DataLoader,
) -> float:
    components.unet.eval()
    losses: List[float] = []
    with torch.no_grad():
        for batch in dataloader:
            loss = compute_batch_loss(batch, args=args, accelerator=accelerator, components=components, training=False)
            gathered = accelerator.gather(loss.detach().unsqueeze(0))
            losses.extend(float(value) for value in gathered.cpu().tolist())
    components.unet.train()
    return float(np.mean(losses)) if losses else float("inf")


def build_components(args: argparse.Namespace, accelerator: Accelerator) -> TrainComponents:
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", cache_dir=args.cache_dir)
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", cache_dir=args.cache_dir)
    controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path, cache_dir=args.cache_dir)
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        controlnet=controlnet,
        cache_dir=args.cache_dir,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.load_ip_adapter(args.ip_adapter_repo_id, subfolder="models", weight_name=args.ip_adapter_weight_name)
    pipe.set_progress_bar_config(disable=True)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    pipe.to(accelerator.device)
    pipe.unet.to(accelerator.device, dtype=weight_dtype)
    pipe.vae.to(accelerator.device, dtype=weight_dtype)
    pipe.text_encoder.to(accelerator.device, dtype=weight_dtype)
    pipe.controlnet.to(accelerator.device, dtype=weight_dtype)
    pipe.unet.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.controlnet.requires_grad_(False)
    if getattr(pipe, "image_encoder", None) is not None:
        pipe.image_encoder.to(accelerator.device, dtype=weight_dtype)
        pipe.image_encoder.requires_grad_(False)

    pipe.unet.to(memory_format=torch.channels_last)
    pipe.controlnet.to(memory_format=torch.channels_last)
    pipe.unet.add_adapter(
        LoraConfig(
            r=args.rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
    )
    if args.initial_lora_path:
        load_initial_lora(pipe.unet, args.initial_lora_path)
    if args.gradient_checkpointing:
        pipe.unet.enable_gradient_checkpointing()
    if args.enable_xformers_memory_efficient_attention and is_xformers_available():
        pipe.unet.enable_xformers_memory_efficient_attention()
        pipe.controlnet.enable_xformers_memory_efficient_attention()
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    if accelerator.mixed_precision == "fp16":
        cast_training_params(pipe.unet, dtype=torch.float32)

    return TrainComponents(
        pipe=pipe,
        noise_scheduler=noise_scheduler,
        tokenizer=tokenizer,
        face_processor=CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14", cache_dir=args.cache_dir),
        unet=pipe.unet,
        vae=pipe.vae,
        text_encoder=pipe.text_encoder,
        controlnet=pipe.controlnet,
        weight_dtype=weight_dtype,
    )


def write_summary(output_dir: Path, summary: Dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_config = ProjectConfiguration(project_dir=str(args.output_dir), logging_dir=str(Path(args.output_dir) / args.logging_dir))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=project_config,
    )

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    if args.seed is not None:
        set_seed(args.seed)
    if accelerator.is_main_process:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    components = build_components(args, accelerator)
    train_rows = load_jsonl(args.manifest)
    if args.max_train_samples is not None:
        train_rows = train_rows[: args.max_train_samples]
    validation_rows = load_jsonl(args.validation_manifest)[: args.validation_max_samples] if args.validation_manifest else []

    train_dataset = HairInpaintDataset(
        train_rows,
        tokenizer=components.tokenizer,
        face_processor=components.face_processor,
        resolution=args.resolution,
        default_prompt=args.default_prompt,
        prompt_dropout_p=args.prompt_dropout_p,
        mask_jitter_px=args.mask_jitter_px,
        training=True,
    )
    validation_dataset = HairInpaintDataset(
        validation_rows,
        tokenizer=components.tokenizer,
        face_processor=components.face_processor,
        resolution=args.resolution,
        default_prompt=args.default_prompt,
        prompt_dropout_p=0.0,
        mask_jitter_px=0,
        training=False,
    )

    sampler = torch.utils.data.WeightedRandomSampler(
        [max(0.05, float(row.get("sample_weight") or 1.0)) for row in train_rows],
        num_samples=len(train_dataset),
        replacement=True,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        sampler=sampler,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=args.dataloader_num_workers > 0,
    )
    validation_dataloader = None
    if validation_rows:
        validation_dataloader = torch.utils.data.DataLoader(
            validation_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=min(2, args.dataloader_num_workers),
            collate_fn=collate_fn,
            pin_memory=True,
        )

    optimizer_params = [parameter for parameter in components.unet.parameters() if parameter.requires_grad]
    if args.use_8bit_adam:
        import bitsandbytes as bnb

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        optimizer_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    def save_model_hook(models, weights, output_dir):
        if not accelerator.is_main_process:
            return
        for model in models:
            save_lora_weights(Path(output_dir), model)
            weights.pop()

    def load_model_hook(models, input_dir):
        while models:
            model = models.pop()
            load_initial_lora(model, Path(input_dir))

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    components.unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        components.unet,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )
    trainable_params = [parameter for parameter in components.unet.parameters() if parameter.requires_grad]
    if validation_dataloader is not None:
        validation_dataloader = accelerator.prepare(validation_dataloader)

    if accelerator.is_main_process:
        accelerator.init_trackers(
            "mirrai-generation-lora",
            config={key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()},
        )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    global_step = 0
    first_epoch = 0
    best_val_loss = float("inf")
    best_step = 0
    patience = 0
    train_loss = 0.0
    stop_training = False

    if args.resume_from_checkpoint:
        checkpoint_path = None
        if args.resume_from_checkpoint == "latest":
            checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
            checkpoints = sorted(checkpoints, key=lambda value: int(value.split("-")[1]))
            checkpoint_path = checkpoints[-1] if checkpoints else None
        else:
            checkpoint_path = os.path.basename(args.resume_from_checkpoint)
        if checkpoint_path:
            accelerator.load_state(os.path.join(args.output_dir, checkpoint_path))
            global_step = int(checkpoint_path.split("-")[1])
            first_epoch = global_step // max(1, num_update_steps_per_epoch)

    progress_bar = tqdm(range(global_step, args.max_train_steps), initial=global_step, disable=not accelerator.is_local_main_process, desc="Steps")
    for epoch in range(first_epoch, math.ceil(args.max_train_steps / max(1, num_update_steps_per_epoch)) + 1):
        components.unet.train()
        for batch in train_dataloader:
            with accelerator.accumulate(components.unet):
                loss = compute_batch_loss(batch, args=args, accelerator=accelerator, components=components, training=True)
                avg_loss = accelerator.gather(loss.detach().unsqueeze(0)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                accelerator.log({"train_loss": train_loss, "lr": lr_scheduler.get_last_lr()[0]}, step=global_step)
                train_loss = 0.0

                if accelerator.is_main_process and global_step % args.checkpointing_steps == 0:
                    checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
                    checkpoints = sorted(checkpoints, key=lambda value: int(value.split("-")[1]))
                    if args.checkpoints_total_limit is not None and len(checkpoints) >= args.checkpoints_total_limit:
                        for checkpoint in checkpoints[: len(checkpoints) - args.checkpoints_total_limit + 1]:
                            shutil.rmtree(os.path.join(args.output_dir, checkpoint), ignore_errors=True)
                    accelerator.save_state(os.path.join(args.output_dir, f"checkpoint-{global_step}"))

                if validation_dataloader is not None and global_step % args.validation_steps == 0:
                    val_loss = evaluate_validation(accelerator=accelerator, args=args, components=components, dataloader=validation_dataloader)
                    accelerator.log({"val_loss": val_loss}, step=global_step)
                    if accelerator.is_main_process:
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_step = global_step
                            patience = 0
                            save_lora_weights(Path(args.output_dir) / "best", unwrap_model(accelerator, components.unet))
                        else:
                            patience += 1
                    if patience >= args.early_stopping_patience:
                        stop_training = True

                if global_step >= args.max_train_steps or stop_training:
                    break
            if global_step >= args.max_train_steps or stop_training:
                break
        if global_step >= args.max_train_steps or stop_training:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_unet = unwrap_model(accelerator, components.unet)
        save_lora_weights(Path(args.output_dir) / "final", final_unet)
        write_summary(
            Path(args.output_dir),
            {
                "manifest": str(args.manifest),
                "validation_manifest": str(args.validation_manifest) if args.validation_manifest else None,
                "output_dir": str(args.output_dir),
                "global_step": global_step,
                "best_step": best_step,
                "best_val_loss": None if best_val_loss == float("inf") else best_val_loss,
                "stopped_early": stop_training,
                "config": {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()},
            },
        )

    accelerator.end_training()


if __name__ == "__main__":
    main()
