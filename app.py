"""Photo intervention with Stable Diffusion 1.5 img2img and one ControlNet at a time.

Designed for a 4 GB GPU: pipelines are offloaded to CPU and the active ControlNet
is replaced when the user changes its type. Models download automatically from
Hugging Face on first use.
"""
from __future__ import annotations

import gc
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import gradio as gr
import numpy as np
import torch
from controlnet_aux import HEDdetector, MidasDetector, OpenposeDetector, PidiNetDetector
from diffusers import (
    ControlNetModel,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionControlNetInpaintPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    UniPCMultistepScheduler,
)
from PIL import Image, ImageFilter, PngImagePlugin
from huggingface_hub import hf_hub_download

# This safetensors file is the exact SD 1.5 checkpoint that Automatic1111
# reports with the short hash 6ce0161689.
CHECKPOINT_REPO = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CHECKPOINT_FILE = "v1-5-pruned-emaonly.safetensors"
CHECKPOINT_SHA256 = "6ce0161689b3853acaa03779ec93eafe75a02f4ced659bee03f50797806fa2fa"
CONTROLNET_MODELS = {
    "Canny": "lllyasviel/control_v11p_sd15_canny",
    "Depth": "lllyasviel/control_v11f1p_sd15_depth",
    "HED": "lllyasviel/control_v11p_sd15_softedge",
    "OpenPose": "lllyasviel/control_v11p_sd15_openpose",
    "Scribble": "lllyasviel/control_v11p_sd15_scribble",
}
# What each ControlNet type actually preserves from the source photo. Shown in
# the UI so a first-time user (e.g. a SAGE reader) doesn't have to guess.
CONTROLNET_DESCRIPTIONS = {
    "None": "Pure img2img, no structural constraint — the model is free to reinterpret the whole photo.",
    "Canny": "Preserves hard edges/outlines. Good for architecture, objects, sharp contours.",
    "Depth": "Preserves depth and volume relationships. Usually the best default for landscapes and scenes.",
    "HED": "Preserves soft, holistic edges — looser than Canny, keeps general shapes without hard lines.",
    "OpenPose": "Preserves a detected human pose/skeleton. Only useful when the source photo already shows a clear human figure.",
    "Scribble": "Preserves simplified, sketch-level strokes — the loosest structural constraint besides None.",
}
# "Reference Only" (Automatic1111's attention-injection mode) has no equivalent
# ControlNet in diffusers. IP-Adapter is the functional substitute: it
# conditions generation on a reference image's style/content via the UNet's
# cross-attention, without deriving a structural map (edges/depth/pose) from
# it, so it can be used alone or layered on top of a regular ControlNet.
IP_ADAPTER_REPO = "h94/IP-Adapter"
IP_ADAPTER_WEIGHT = "ip-adapter_sd15.bin"
# CLIP's text encoder truncates prompts beyond 75 tokens (+ start/end tokens).
CLIP_TOKENIZER_REPO = "openai/clip-vit-large-patch14"
CLIP_MAX_TOKENS = 75
OUTPUT_DIR = Path("outputs")
# `python app.py` in a notebook starts a child process, where importing
# `google.colab` is not sufficient to identify Colab. Its runtime variables
# (and /content on Linux) remain available to that child process.
IN_COLAB = bool(
    os.environ.get("COLAB_GPU")
    or os.environ.get("COLAB_RELEASE_TAG")
    or Path("/content").exists()
)

_clip_tokenizer = None


def _token_count_label(prompt: str) -> str:
    """Live CLIP token count so truncation (Finding: silent prompt cutoff) is visible before generating."""
    global _clip_tokenizer
    prompt = (prompt or "").strip()
    if not prompt:
        return f"0 / {CLIP_MAX_TOKENS} tokens"
    if _clip_tokenizer is None:
        from transformers import CLIPTokenizer

        _clip_tokenizer = CLIPTokenizer.from_pretrained(CLIP_TOKENIZER_REPO)
    # encode() adds the start/end special tokens; subtract them for the count that matters to the 75-token budget.
    count = len(_clip_tokenizer.encode(prompt)) - 2
    if count > CLIP_MAX_TOKENS:
        return f"⚠️ {count} / {CLIP_MAX_TOKENS} tokens — will be truncated, CLIP ignores the rest"
    return f"{count} / {CLIP_MAX_TOKENS} tokens"


def _torch_config() -> tuple[torch.dtype, bool]:
    """Use float16 only on CUDA; CPU float16 is unsupported/slow for this pipeline."""
    return (torch.float16, True) if torch.cuda.is_available() else (torch.float32, False)


def _fit_image(image: Image.Image, width: int, height: int) -> Image.Image:
    image = image.convert("RGB")
    return image.resize((int(width), int(height)), Image.Resampling.LANCZOS)


class Generator:
    def __init__(self) -> None:
        self.pipe = None
        self.active_key: Optional[tuple[str, bool]] = None
        self.detectors: dict[str, object] = {}
        self.checkpoint_path: Optional[str] = None
        self.ip_adapter_loaded = False

    def _clear(self) -> None:
        self.pipe = None
        self.active_key = None
        self.ip_adapter_loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def _configure(self, pipe, uses_cuda: bool):
        if uses_cuda:
            if IN_COLAB:
                # A Colab T4 has enough VRAM for SD 1.5 + one ControlNet. Keeping
                # the models on its GPU is dramatically faster than CPU offloading.
                pipe.to("cuda")
                pipe.enable_vae_slicing()
            else:
                # Sequential offload is slower but is the safest default for 4 GB VRAM.
                pipe.enable_attention_slicing("max")
                pipe.enable_vae_tiling()
                pipe.enable_sequential_cpu_offload()
        return pipe

    def _load_pipe(self, control_type: str, inpaint: bool):
        key = (control_type, inpaint)
        if self.pipe is not None and self.active_key == key:
            return self.pipe
        self._clear()
        dtype, uses_cuda = _torch_config()
        common = {"torch_dtype": dtype, "use_safetensors": True, "safety_checker": None}
        token = os.getenv("HF_TOKEN") or None
        if self.checkpoint_path is None:
            # Downloaded once and then reused from Hugging Face's local cache.
            self.checkpoint_path = hf_hub_download(
                repo_id=CHECKPOINT_REPO, filename=CHECKPOINT_FILE, token=token
            )
        if control_type == "None":
            cls = StableDiffusionInpaintPipeline if inpaint else StableDiffusionImg2ImgPipeline
            self.pipe = cls.from_single_file(
                self.checkpoint_path, config=CHECKPOINT_REPO, token=token, **common
            )
        else:
            controlnet = ControlNetModel.from_pretrained(CONTROLNET_MODELS[control_type], torch_dtype=dtype, token=token)
            cls = StableDiffusionControlNetInpaintPipeline if inpaint else StableDiffusionControlNetImg2ImgPipeline
            self.pipe = cls.from_single_file(
                self.checkpoint_path, config=CHECKPOINT_REPO, controlnet=controlnet, token=token, **common
            )
        self.pipe = self._configure(self.pipe, uses_cuda)
        self.active_key = key
        return self.pipe

    def _ensure_ip_adapter(self, use_reference: bool, uses_cuda: bool) -> None:
        """Attach/detach IP-Adapter (Reference Only) on the current pipe, on demand."""
        if self.pipe is None:
            return
        if use_reference and not self.ip_adapter_loaded:
            self.pipe.load_ip_adapter(IP_ADAPTER_REPO, subfolder="models", weight_name=IP_ADAPTER_WEIGHT)
            self.ip_adapter_loaded = True
            if uses_cuda and self.pipe.image_encoder is not None:
                # The image encoder is registered after `_configure()` already placed
                # the rest of the pipe, so it isn't picked up by CPU-offload hooks set
                # up earlier. Move it explicitly to avoid a CPU/CUDA device mismatch
                # when it encodes the reference image. On a 4 GB GPU this keeps ~1.2 GB
                # resident for as long as Reference Only stays enabled.
                self.pipe.image_encoder.to("cuda")
        elif not use_reference and self.ip_adapter_loaded:
            self.pipe.unload_ip_adapter()
            self.ip_adapter_loaded = False

    def _detector(self, kind: str):
        if kind not in self.detectors:
            if kind == "Depth":
                self.detectors[kind] = MidasDetector.from_pretrained("lllyasviel/Annotators")
            elif kind == "HED":
                self.detectors[kind] = HEDdetector.from_pretrained("lllyasviel/Annotators")
            elif kind == "OpenPose":
                self.detectors[kind] = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
            elif kind == "Scribble":
                self.detectors[kind] = PidiNetDetector.from_pretrained("lllyasviel/Annotators")
        return self.detectors.get(kind)

    @staticmethod
    def _set_scheduler(pipe, sampler: str, schedule: str) -> None:
        """Configure a Diffusers scheduler equivalent to the named UI sampler."""
        config = pipe.scheduler.config
        karras = schedule == "Karras"
        if sampler == "DPM++ 2M":
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                config,
                algorithm_type="dpmsolver++",
                solver_order=2,
                use_karras_sigmas=karras,
            )
        elif sampler == "Euler a":
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
                config, use_karras_sigmas=karras
            )
        elif sampler == "Euler":
            pipe.scheduler = EulerDiscreteScheduler.from_config(config, use_karras_sigmas=karras)
        elif sampler == "DDIM":
            pipe.scheduler = DDIMScheduler.from_config(config)
        elif sampler == "UniPC":
            pipe.scheduler = UniPCMultistepScheduler.from_config(config)

    def control_map(self, image: Image.Image, control_type: str) -> Optional[Image.Image]:
        if control_type == "None":
            return None
        if control_type == "Canny":
            edges = cv2.Canny(np.array(image), 100, 200)
            return Image.fromarray(np.stack([edges] * 3, axis=-1))
        detector = self._detector(control_type)
        if control_type == "Scribble":
            result = detector(image, detect_resolution=512, image_resolution=512, scribble=True)
        else:
            result = detector(image, detect_resolution=512, image_resolution=512)
        # controlnet_aux annotators round their output to the nearest multiple of 64
        # internally, which can land on a different size than the width/height the
        # UI enforces (multiples of 8) — e.g. 760 -> 768. That mismatch is exactly
        # what raises "tensor a must match tensor b" errors deep in ControlNet, since
        # its conditioning map and the image being generated must be the same size.
        # Forcing the map back to `image`'s exact size (already fit to width/height)
        # guarantees they always match.
        return result.resize(image.size, Image.Resampling.LANCZOS)

    @staticmethod
    def _build_mask(editor_value, width: int, height: int, blur: float) -> Optional[Image.Image]:
        """Turn what was painted in the mask editor into a binary-ish mask for inpainting.

        `editor_value` is Gradio's ImageEditor payload: {"background", "layers", "composite"}.
        Painted strokes live in `layers` as RGBA images; anywhere with alpha > 0 is
        "generate freely here", everywhere else stays close to the original photo.
        """
        if not editor_value:
            return None
        layers = editor_value.get("layers") or []
        if not layers:
            return None
        mask = np.zeros(layers[0].size[::-1], dtype=np.uint8)
        for layer in layers:
            alpha = np.array(layer.split()[-1] if layer.mode == "RGBA" else layer.convert("L"))
            mask = np.maximum(mask, alpha)
        if not mask.any():
            return None
        mask_image = Image.fromarray(mask, mode="L").resize((int(width), int(height)), Image.Resampling.LANCZOS)
        if blur:
            # Feathers the edge between the painted region and the rest of the photo
            # so the inpainted patch doesn't read as a hard-cut rectangle/blob.
            mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=float(blur)))
        return mask_image

    def generate(
        self, original, prompt, negative_prompt, control_type, control_source, source_is_map,
        reference_image, ip_scale,
        inpaint_enabled, mask_editor_value, mask_blur,
        strength, control_scale, sampler, schedule, steps, cfg, seed, width, height, count,
        progress=gr.Progress(track_tqdm=True),
    ):
        if original is None:
            raise gr.Error("Upload an original photo before generating.")
        if not (prompt or "").strip():
            raise gr.Error("Write a prompt for the intervention.")
        progress(0.0, desc="Preparing image and mask…")
        # SD 1.5 works on multiples of 8; enforce the UI contract defensively.
        width, height = (int(width) // 8 * 8), (int(height) // 8 * 8)
        original = _fit_image(original, width, height)
        # The original is always img2img's initial image. An optional second image
        # only replaces the source used to make (or supply) ControlNet's map.
        control_source = _fit_image(control_source, width, height) if control_source is not None else original
        control_image = control_source if source_is_map and control_type != "None" else self.control_map(control_source, control_type)
        mask = None
        if inpaint_enabled:
            mask = self._build_mask(mask_editor_value, width, height, mask_blur)
            if mask is None:
                raise gr.Error(
                    "Inpainting mode is on but nothing is painted in the mask editor."
                )
        dtype, uses_cuda = _torch_config()
        # Best-effort heuristic for the progress message: true on the very first call in
        # this process (downloads the ~4GB base checkpoint) or the first time a
        # detector-based ControlNet (Depth/HED/OpenPose/Scribble) is used (downloads its
        # weights + annotator). Doesn't cover every edge case, but catches the slow ones.
        needs_detector = control_type in ("Depth", "HED", "OpenPose", "Scribble") and control_type not in self.detectors
        need_download = self.checkpoint_path is None or needs_detector
        progress(
            0.05,
            desc=(
                "Downloading models (first run after restarting the environment — this can "
                "take several minutes, no need to touch anything)…"
                if need_download
                else "Loading the model…"
            ),
        )
        pipe = self._load_pipe(control_type, inpaint=mask is not None)
        use_reference = reference_image is not None
        self._ensure_ip_adapter(use_reference, uses_cuda)
        if use_reference:
            pipe.set_ip_adapter_scale(float(ip_scale))
        self._set_scheduler(pipe, sampler, schedule)
        progress(0.4, desc="Generating the image…")
        actual_seed = random.randint(0, 2**31 - 1) if int(seed) < 0 else int(seed)
        generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(actual_seed)
        args = dict(
            prompt=prompt, negative_prompt=negative_prompt or None, image=original,
            strength=float(strength), num_inference_steps=int(steps), guidance_scale=float(cfg),
            generator=generator, num_images_per_prompt=int(count),
        )
        if mask is not None:
            # Only the painted region gets denoised freely; the checkpoint is a
            # regular (non-inpainting-finetuned) SD 1.5, so diffusers falls back to
            # its "legacy" latent-blending strategy automatically — this keeps the
            # project on the same pinned/hashed base model instead of swapping in a
            # separately-trained inpainting checkpoint.
            args.update(mask_image=mask)
        if control_image is not None:
            args.update(control_image=control_image, controlnet_conditioning_scale=float(control_scale))
        if use_reference:
            args.update(ip_adapter_image=reference_image.convert("RGB"))
        try:
            images = pipe(**args).images
        except torch.cuda.OutOfMemoryError as exc:
            self._clear()
            raise gr.Error("Insufficient GPU memory. Try 512×512, 1 image, and fewer steps.") from exc
        OUTPUT_DIR.mkdir(exist_ok=True)
        saved = []
        stamp = time.strftime("%Y%m%d-%H%M%S")
        ip_info = f"IP-Adapter (Reference Only): on, scale {ip_scale}" if use_reference else "IP-Adapter (Reference Only): off"
        inpaint_info = f"Inpainting: on, mask blur {mask_blur}px" if mask is not None else "Inpainting: off"
        parameters = (
            f"Prompt: {prompt}\nNegative prompt: {negative_prompt or ''}\n"
            f"Sampler: {sampler}, Schedule type: {schedule}, CFG scale: {cfg}, Seed: {actual_seed}, "
            f"Size: {width}x{height}, Model hash: {CHECKPOINT_SHA256[:10]}, "
            f"Model: {CHECKPOINT_FILE}, Denoising strength: {strength}, "
            f"ControlNet: {control_type}, Control scale: {control_scale}, Steps: {steps}, {ip_info}, {inpaint_info}"
        )
        for index, image in enumerate(images, start=1):
            path = OUTPUT_DIR / f"intervention-{stamp}-seed{actual_seed}-{index}.png"
            metadata = PngImagePlugin.PngInfo()
            metadata.add_text("parameters", parameters)
            image.save(path, pnginfo=metadata)
            saved.append(str(path.resolve()))
        return original, control_image, mask, images, f"{parameters}\nSaved: " + ", ".join(saved)


engine = Generator()

with gr.Blocks(title="Visual Archive Activator · SD 1.5 + ControlNet") as demo:
    gr.Markdown(
        "# Visual Archive Activator\n"
        "The photo is always img2img's initial image; if you pick a ControlNet, its conditioning map is generated "
        "automatically from the same photo. The defaults below reflect the project's recommended configuration "
        "(Depth ControlNet, strength 0.70, control scale 0.45, CFG 8.0) — adjust them freely for your own archive."
    )
    gr.Markdown(
        "**Colab/T4 mode:** models kept on GPU to prioritize speed."
        if IN_COLAB
        else "**4 GB GPU mode:** CPU offload enabled to prioritize stability."
    )
    gr.Markdown("_Developed by Federico Fort._")
    gr.Markdown(
        "**Quick start:** 1) Upload a photo. 2) Write a prompt describing the intervention. "
        "3) Pick a ControlNet (or leave the default) — the description below explains what it preserves. "
        "4) Press **Generate** below. Everything else has sensible defaults; open the accordions only if you "
        "need a custom control image, a reference image, inpainting, or the advanced sampler settings."
    )
    with gr.Row():
        with gr.Column():
            original = gr.Image(label="Original image", type="pil", sources=["upload"])
            prompt = gr.Textbox(label="Prompt", placeholder="e.g.: restore the facade with lush plants growing from the windows", lines=3)
            prompt_tokens = gr.Markdown("0 / 75 tokens")
            negative = gr.Textbox(label="Negative prompt (optional)", lines=2)
            negative_tokens = gr.Markdown("0 / 75 tokens")
            control = gr.Dropdown(
                ["None", "Canny", "Depth", "HED", "OpenPose", "Scribble"], value="Depth", label="ControlNet",
                info="What structure gets preserved from the photo — see the description below.",
            )
            control_description = gr.Markdown(CONTROLNET_DESCRIPTIONS["Depth"])
            with gr.Accordion("Custom control image (optional)", open=False):
                gr.Markdown(
                    "By default the ControlNet map is built automatically from the original photo above. Use this "
                    "only to derive the map from a different image, or to supply a map you already prepared."
                )
                control_source = gr.Image(label="Control image (optional)", type="pil", sources=["upload"])
                source_is_map = gr.Checkbox(label="The control image is already a prepared map", value=False)
            with gr.Accordion("Reference image (optional) — Reference Only / IP-Adapter", open=False):
                gr.Markdown(
                    "Conditions generation on a reference image's style/content via cross-attention, without "
                    "deriving a structural map from it (unlike ControlNet). Can be combined with or without "
                    "ControlNet."
                )
                reference_image = gr.Image(label="Reference image (optional)", type="pil", sources=["upload"])
                ip_scale = gr.Slider(
                    0.0, 1.5, value=0.6, step=0.05, label="IP-Adapter scale",
                    info="How much the reference weighs in. 0.3–0.8 is a good starting point; raise it to lean more into its style.",
                )
            with gr.Accordion("Inpainting (optional) — add something to an area", open=False):
                gr.Markdown(
                    "Paint with the brush where something new should appear — for example a figure in an "
                    "\"inhabited landscape\". Only that area is generated freely; the rest of the photo stays "
                    "close to the original, instead of fighting the structure ControlNet imposes on it."
                )
                inpaint_enabled = gr.Checkbox(label="Enable inpainting mode", value=False)
                mask_editor = gr.ImageEditor(
                    label="Paint the area to generate", type="pil", visible=False,
                    brush=gr.Brush(default_size=40, colors=["#ffffff"], color_mode="fixed"),
                )
                mask_blur = gr.Slider(
                    0, 20, value=4, step=1, label="Mask feather (px)", visible=False,
                    info="Softens the edge between the painted area and the rest so the blend doesn't look cut out.",
                )
        with gr.Column():
            strength = gr.Slider(
                0.05, 0.95, value=0.70, step=0.05, label="Denoising strength (img2img)",
                info="How far it drifts from the original: low keeps more, high allows bigger changes.",
            )
            control_scale = gr.Slider(
                0.0, 2.0, value=0.45, step=0.05, label="ControlNet conditioning scale",
                info="How strictly the ControlNet map (edges/depth/pose) must be respected.",
            )
            cfg = gr.Slider(
                1.0, 15.0, value=8.0, step=0.5, label="CFG scale",
                info="How closely it must follow the prompt literally. Higher = more adherence to text, less variety.",
            )
            with gr.Accordion("Advanced settings", open=False):
                sampler = gr.Dropdown(["DPM++ 2M", "Euler a", "Euler", "DDIM", "UniPC"], value="DPM++ 2M", label="Sampler")
                schedule = gr.Dropdown(["Karras", "Default"], value="Karras", label="Schedule type")
                steps = gr.Slider(10, 60, value=30, step=1, label="Steps")
                seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
                with gr.Row():
                    width = gr.Slider(256, 1024, value=512, step=8, label="Width")
                    height = gr.Slider(256, 1024, value=512, step=8, label="Height")
                count = gr.Slider(1, 2, value=1, step=1, label="Count (1 recommended for 4 GB GPUs)")
    gr.Markdown(
        "_The first generation after restarting the environment (or the first time a new ControlNet/"
        "Reference Only is used) can take several minutes: it's downloading models, not stuck. Watch the "
        "progress bar below, or the console/Colab cell._"
    )
    run = gr.Button("Generate", variant="primary")
    result = gr.Gallery(label="Generated images", columns=2, object_fit="contain")
    with gr.Accordion("Debug previews (resized original, ControlNet map, applied mask)", open=False):
        with gr.Row():
            preview_original = gr.Image(label="Resized original", type="pil")
            preview_control = gr.Image(label="ControlNet map", type="pil")
            preview_mask = gr.Image(label="Applied mask", type="pil")
    status = gr.Textbox(label="Status", interactive=False)
    prompt.change(_token_count_label, prompt, prompt_tokens)
    negative.change(_token_count_label, negative, negative_tokens)
    control.change(lambda c: CONTROLNET_DESCRIPTIONS[c], control, control_description)
    # Paint directly on the same photo that's being generated, instead of a second upload.
    original.change(lambda img: img, original, mask_editor)
    inpaint_enabled.change(
        lambda enabled: (gr.update(visible=enabled), gr.update(visible=enabled)),
        inpaint_enabled, [mask_editor, mask_blur],
    )
    run.click(
        engine.generate,
        [
            original, prompt, negative, control, control_source, source_is_map,
            reference_image, ip_scale,
            inpaint_enabled, mask_editor, mask_blur,
            strength, control_scale, sampler, schedule, steps, cfg, seed, width, height, count,
        ],
        [preview_original, preview_control, preview_mask, result, status],
    )

if __name__ == "__main__":
    # Colab cannot open the notebook VM's localhost from the user's browser.
    # Gradio's temporary share URL is required there; local use remains private.
    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        show_error=True,
        share=IN_COLAB,
    )
