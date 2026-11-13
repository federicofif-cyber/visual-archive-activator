# Visual Archive Activator — SD 1.5 + ControlNet

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/federicofif-cyber/visual-archive-activator/blob/v1.0/colab_intervencion_sd15_controlnet.ipynb)

Developed by Federico Fort.

A small Gradio app for intervening in archival photographs with generative AI. It uses the same photo in two ways: as **img2img**'s initial image and, if a ControlNet is chosen, as the source for automatically building the conditioning map. Every generated image is saved with its prompt and settings embedded as PNG metadata, so each run leaves a reproducible record.

This is the companion tool to the Sage Research Methods dataset *Generative AI as Archival Intervention: Reimagining Bolivian Photographic Archives (1925–1935) With Stable Diffusion 1.5 and the Visual Archive Activator*. Version `v1.0` is the version used to produce that dataset.

It runs on Google Colab (free T4 GPU) or locally on a GPU with as little as 4 GB of VRAM.

## Quick start (Google Colab)

1. Click the **Open in Colab** badge above.
2. Choose **Runtime → Change runtime type → T4 GPU**.
3. Run the cells top to bottom. The last cell downloads the app from this repository (version `v1.0`) and prints a public Gradio link; open it to use the interface.

The notebook:

1. shows the available GPU;
2. installs the dependencies quietly (Colab's preinstalled CUDA PyTorch is reused; one is installed only if missing). pip may note version conflicts with packages Colab ships for other purposes (e.g. `google-genai`, `google-adk`, `hf-gradio`); the app doesn't use them, so the notebook reports this in one line and keeps the full log in `pip_install.log`;
3. clones this repository at the pinned version and launches `app.py`;
4. downloads model weights automatically on the first generation (about 4 GB; it takes several minutes, and the interface shows progress messages).

Colab deletes its files when the session ends, so download each image you want to keep as soon as it is generated.

For a gated base model, create a Colab secret named `HF_TOKEN` or uncomment the `login()` lines in the last cell.

## Usage

1. Upload a photo.
2. Write the prompt for the intervention. The negative prompt is optional. A live counter under each prompt shows how many of CLIP's 75 tokens you are using.
3. Choose `None` for pure img2img or a ControlNet type — the description under the dropdown explains what each one preserves.
4. Set width and height (multiples of 8). Start small on a 4 GB GPU (512×512, 20–30 steps, 1 image).
5. Press **Generate**. The gallery shows the result; the Status field shows the seed and the saved path. Diagnostic previews (resized original, ControlNet map, mask) are in a collapsed section below.

### img2img and ControlNet are different things

- **Initial image (img2img):** the photograph the model transforms. `Denoising strength` controls how far it drifts from the original: low keeps more, high allows bigger changes.
- **ControlNet:** preserves an additional structure derived automatically from that same photograph. `Canny` preserves outlines; `Depth`, volumes/depth; `HED`, soft edges; `OpenPose`, the pose; and `Scribble`, simplified strokes. `ControlNet conditioning scale` controls how strictly that map must be respected.

No need to prepare maps manually: the app builds them from the original image.

### Reproducibility

The seed appears after each generation and is also part of the PNG's filename. To repeat an image, use the same photo, prompt, negative prompt, ControlNet selection and every value, including seed, resolution, steps, CFG, denoising and scale. The resolution matters too: the same seed produces different starting noise at a different size. A seed of `-1` requests a random one. Identical settings can still give slightly different images on different hardware or library versions.

## Models

| Use | Hugging Face repository |
| --- | --- |
| Exact SD 1.5 base | `stable-diffusion-v1-5/stable-diffusion-v1-5` — `v1-5-pruned-emaonly.safetensors` (SHA256 `6ce0161689…`) |
| Canny | `lllyasviel/control_v11p_sd15_canny` |
| Depth | `lllyasviel/control_v11f1p_sd15_depth` |
| HED | `lllyasviel/control_v11p_sd15_softedge` |
| OpenPose | `lllyasviel/control_v11p_sd15_openpose` |
| Scribble | `lllyasviel/control_v11p_sd15_scribble` |
| Reference Only (IP-Adapter) | `h94/IP-Adapter` — `ip-adapter_sd15.bin` |

On first use, the base model, the chosen ControlNet and — for Depth/HED/OpenPose/Scribble — its detector are downloaded. The base model may require accepting its license on Hugging Face.

## Important parameters

The interface's defaults are the configuration recommended for the Bolivian state photographic archive (1925–1935):

| Parameter | Default value |
| --- | --- |
| Width / Height | 512 × 512 to start; multiples of 8 |
| Count | 1 |
| Steps | 30 |
| ControlNet | Depth |
| Denoising strength | 0.70 |
| ControlNet scale | 0.45 |
| CFG | 8.0 |

They are a reasonable starting point for any archive, not fixed values — adjust them for your own corpus. If a memory error appears, close other GPU-using applications, keep 512×512 and one image, and reduce steps.

## Reference Only (IP-Adapter)

Automatic1111 has a ControlNet mode called "Reference Only" that doesn't exist in `diffusers`: instead of deriving a structural map (edges, depth, pose) from an image, it injects its style/content directly into the model's cross-attention. This app solves the same problem with **IP-Adapter**:

- Upload a reference image distinct from the original and from the control image — it's a third, independent input, in its own collapsible section.
- `IP-Adapter scale` controls how much that reference weighs; 0.3–0.8 is a good starting range.
- It can be used alone (ControlNet set to `None`) or alongside any active ControlNet.
- On a 4 GB GPU, enabling it adds ~1.2 GB of resident VRAM while in use — if a memory error appears, disable it (leave the reference image empty) or lower resolution/steps.

For portraits, Reference Only can be more useful than Canny/Depth when the goal is for the model to reinterpret the composition rather than trace it. The friction between what the reference suggests and what the model produces is itself material for analysis.

## Inpainting (adding something to an area — e.g. an "inhabited landscape")

ControlNet has a limitation when the goal is to *add* something the original photo doesn't have — typically, inserting a human figure into a documentary landscape from which Indigenous presence was erased. Depth/Canny force the model to respect the photo's exact geometry, so the higher the `control scale`, the less room the model has to generate something that map doesn't account for. Here the technical architecture itself reproduces the archive's absence, even when explicitly asked to do the opposite.

Inpainting mode addresses this by generating *only* the area you mark freely, while the rest of the photo stays close to the original:

1. Upload the photo in "Original image" — it syncs automatically into the mask editor inside the Inpainting section.
2. Open the "Inpainting" section and check "Enable inpainting mode".
3. Paint with the white brush over the area where the new content should appear.
4. Adjust "Mask feather" if the edge between the painted area and the rest looks too cut-out (4px by default is usually enough).
5. `Denoising strength` still controls how much freedom that painted area gets; for inpainting, a high value works best (0.75–0.95).
6. It can be combined with ControlNet (with a low `control scale`) and with Reference Only.

Technical note: the checkpoint is a regular SD 1.5, not a specialised inpainting version. Diffusers detects this automatically and falls back to its "legacy" latent-blending strategy — it works well for this use case, but edges may look somewhat less clean than with a specialised checkpoint. This is intentional, to stay on the pinned, hashed checkpoint that underpins the project's reproducibility.

### Recommended preset: inhabited landscape (adding a figure)

| Parameter | Recommended value | Why |
| --- | --- | --- |
| ControlNet | `None` | The empty landscape's Depth map would push against adding something exactly there. If the rest of the composition drifts too much, try Depth with control scale 0.15–0.20. |
| Denoising strength | 0.85–0.95 | The mask already protects the rest; the painted area needs near-total freedom. |
| CFG scale | 8.0–9.0 | Without ControlNet, the text prompt becomes the only structural guide. |
| Steps | 30–40 | More room to resolve detail in a small figure. |
| Mask feather | 4–8px | Raise it if the blend's edge looks too sharp. |
| Negative prompt | add `distorted figure, extra limbs, floating body, disfigured, blurry face` | SD 1.5 is prone to anatomy artifacts in small figures within wide scenes. |
| Reference Only | optional, scale 0.5–0.6 | A reference photo can inform the painted area without competing structurally with the inpainting. |
| Width / Height | 512×512 to start; 640×640 or 768×768 on Colab | On a 4 GB GPU, don't raise resolution until the basic case works. |

## Local installation (Windows, 4 GB GPU)

On a local GPU the app uses `float16`, attention slicing, VAE tiling and sequential CPU offload, and loads only the selected ControlNet. It prioritises being able to generate at all over speed. No CUDA Toolkit is needed: the PyTorch wheel bundles the CUDA runtime.

Install **64-bit Python 3.11** from [python.org](https://www.python.org/downloads/), then, from this folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

Check that PyTorch sees the GPU:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

If the base model requires a Hugging Face token, set it before launching:

```powershell
$env:HF_TOKEN = "your_read_only_token"
```

Then launch the app and open the local URL Gradio prints:

```powershell
python app.py
```

Images are saved automatically under `outputs/`, which Git ignores.

## Pinned versions

`requirements.txt` pins Diffusers 0.35.1, Transformers 4.54.1, Accelerate 1.10.1, Gradio 5.44.1 and their helpers. PyTorch is installed separately: the CUDA 12.8 wheel on Windows, and on Colab the CUDA build it already provides.

## Security, scope and ethics

Locally, the app binds only to `127.0.0.1` and exposes no public URL; on Colab it uses Gradio's temporary share link, because Colab cannot open the VM's localhost from your browser. Diffusers' safety checker is disabled to avoid inconsistent results on artistic or archival photographs. Use the app responsibly, respecting the rights, privacy and consent of the people depicted, and label generated images as AI-generated and speculative when you publish them.

## Citation

If you use this tool, please cite it. Citation metadata is in [`CITATION.cff`](CITATION.cff); GitHub's "Cite this repository" button formats it for you. Each release is archived on Zenodo with its own DOI.

## License

The code in this repository is released under the [MIT License](LICENSE). The models it downloads are distributed under their own licenses (Stable Diffusion 1.5 under the CreativeML OpenRAIL-M license); check them on Hugging Face before use.
