# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "python-dotenv",
#     "requests",
# ]
# ///
"""Generate watercolor images for quote pages using a local ComfyUI + Flux setup."""

import json
import os
import random
import sys
import time
import urllib.parse
from pathlib import Path

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUOTES_DIR = PROJECT_ROOT / "content" / "quotes"
IMAGES_DIR = PROJECT_ROOT / "static" / "images" / "quotes"

STYLE_CONSTRAINTS = (
    "Watercolor painting on cold-pressed paper with wet-on-wet brushstrokes and "
    "visible pigment bleeds. Muted natural palette: soft blues, warm ochres, gentle "
    "greens, quiet grays. Abstract or semi-abstract composition evoking emotional "
    "essence. No text, words, letters, or numbers anywhere in the image. "
    "No recognizable human faces."
)

# ComfyUI / Flux defaults (override via .env)
DEFAULT_COMFYUI_URL = "http://stormtrooper.accapehart.com:8188"
DEFAULT_FLUX_UNET = "flux1-dev.safetensors"
DEFAULT_FLUX_CLIP1 = "clip_l.safetensors"
DEFAULT_FLUX_CLIP2 = "t5xxl_fp16.safetensors"
DEFAULT_FLUX_VAE = "ae.safetensors"

POLL_INTERVAL = 2  # seconds between status checks


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Parse TOML frontmatter delimited by +++ lines.

    Returns (frontmatter_dict, body_text).
    """
    lines = text.split("\n")
    if not lines or lines[0].strip() != "+++":
        return {}, text

    end = None
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "+++":
            end = i
            break
    if end is None:
        return {}, text

    fm = {}
    for line in lines[1:end]:
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        # Strip surrounding quotes
        for q in ("'", '"'):
            if value.startswith(q) and value.endswith(q):
                value = value[1:-1]
                break
        fm[key] = value

    body = "\n".join(lines[end + 1 :])
    return fm, body


def write_image_field(filepath: Path, image_path: str) -> None:
    """Insert or update the image field in TOML frontmatter."""
    text = filepath.read_text(encoding="utf-8")
    lines = text.split("\n")

    if not lines or lines[0].strip() != "+++":
        return

    end = None
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "+++":
            end = i
            break
    if end is None:
        return

    # Check if image field already exists
    image_line = f"image = '{image_path}'"
    for i in range(1, end):
        if lines[i].strip().startswith("image"):
            lines[i] = image_line
            filepath.write_text("\n".join(lines), encoding="utf-8")
            return

    # Insert before closing +++
    lines.insert(end, image_line)
    filepath.write_text("\n".join(lines), encoding="utf-8")


def truncate_at_word(text: str, max_chars: int = 300) -> str:
    """Truncate text at a word boundary."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > 0:
        truncated = truncated[:last_space]
    return truncated + "..."


def build_prompt(quote: str, attribution: str, body: str) -> str:
    """Build the image generation prompt from quote metadata.

    Deliberately excludes the raw quote text to prevent Flux from rendering
    it as calligraphy/typography in the image. Uses only the reflection body
    for thematic context.
    """
    body_excerpt = truncate_at_word(body.strip(), max_chars=200)
    return (
        f"Abstract watercolor landscape painting. "
        f"The scene evokes the emotional mood of a philosophical reflection by {attribution}. "
        f"Thematic context: {body_excerpt}\n\n"
        f"{STYLE_CONSTRAINTS}"
    )


def build_workflow(prompt_text: str, cfg: dict) -> dict:
    """Build a ComfyUI workflow for Flux dev image generation."""
    seed = random.randint(0, 2**32 - 1)
    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": cfg["flux_unet"],
                "weight_dtype": "default",
            },
        },
        "2": {
            "class_type": "DualCLIPLoader",
            "inputs": {
                "clip_name1": cfg["flux_clip1"],
                "clip_name2": cfg["flux_clip2"],
                "type": "flux",
            },
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": cfg["flux_vae"],
            },
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": prompt_text,
                "clip": ["2", 0],
            },
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": "text, words, letters, numbers, script, quote, citation, writing, calligraphy, typography, caption, label, signature, watermark",
                "clip": ["2", 0],
            },
        },
        "6": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {
                "width": 1344,
                "height": 896,
                "batch_size": 1,
            },
        },
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["4", 0],
                "negative": ["5", 0],
                "latent_image": ["6", 0],
                "seed": seed,
                "steps": 20,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["7", 0],
                "vae": ["3", 0],
            },
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["8", 0],
                "filename_prefix": "quote_gen",
            },
        },
    }


def generate_image(comfyui_url: str, prompt_text: str, cfg: dict) -> bytes | None:
    """Submit a workflow to ComfyUI, wait for completion, return PNG bytes."""
    workflow = build_workflow(prompt_text, cfg)
    payload = {"prompt": workflow}

    # Submit the prompt
    try:
        resp = requests.post(f"{comfyui_url}/prompt", json=payload, timeout=10)
        resp.raise_for_status()
    except requests.ConnectionError:
        print(f"  WARNING: Cannot connect to ComfyUI at {comfyui_url}", file=sys.stderr)
        return None
    except requests.RequestException as e:
        print(f"  WARNING: ComfyUI submit error: {e}", file=sys.stderr)
        if hasattr(e, "response") and e.response is not None:
            print(f"  Response: {e.response.text[:500]}", file=sys.stderr)
        return None

    prompt_id = resp.json().get("prompt_id")
    if not prompt_id:
        print(f"  WARNING: No prompt_id in ComfyUI response", file=sys.stderr)
        return None

    print(f"  Queued (prompt_id: {prompt_id}), waiting for completion...")

    # Poll for completion
    for _ in range(300):  # up to 10 minutes
        time.sleep(POLL_INTERVAL)
        try:
            hist_resp = requests.get(f"{comfyui_url}/history/{prompt_id}", timeout=10)
            hist_resp.raise_for_status()
            history = hist_resp.json()
        except requests.RequestException:
            continue

        if prompt_id not in history:
            continue

        outputs = history[prompt_id].get("outputs", {})
        # Find the SaveImage node output (node "9")
        for node_id, node_output in outputs.items():
            images = node_output.get("images", [])
            if images:
                img_info = images[0]
                filename = img_info["filename"]
                subfolder = img_info.get("subfolder", "")
                img_type = img_info.get("type", "output")
                params = urllib.parse.urlencode(
                    {"filename": filename, "subfolder": subfolder, "type": img_type}
                )
                img_resp = requests.get(f"{comfyui_url}/view?{params}", timeout=30)
                img_resp.raise_for_status()
                return img_resp.content

        # Prompt finished but no images found
        print(f"  WARNING: ComfyUI completed but no images in output", file=sys.stderr)
        return None

    print(f"  WARNING: Timed out waiting for ComfyUI", file=sys.stderr)
    return None


def process_file(comfyui_url: str, cfg: dict, filepath: Path) -> bool:
    """Process a single quote markdown file. Returns True if generation was attempted."""
    print(f"Processing: {filepath.name}")

    text = filepath.read_text(encoding="utf-8")
    fm, body = parse_frontmatter(text)

    if not fm.get("quote"):
        print(f"  Skipping (no quote field)")
        return False

    # Check if image already exists
    existing_image = fm.get("image", "").strip("'\"")
    if existing_image:
        disk_path = PROJECT_ROOT / "static" / existing_image.lstrip("/")
        if disk_path.exists():
            print(f"  Skipping (image already exists: {existing_image})")
            return False

    slug = filepath.stem
    quote = fm["quote"]
    attribution = fm.get("attribution", "Unknown")

    prompt_text = build_prompt(quote, attribution, body)
    print(f"  Generating image for '{slug}'...")

    image_data = generate_image(comfyui_url, prompt_text, cfg)
    if image_data is None:
        print(f"  WARNING: Failed to generate image, skipping.")
        return True

    # Save image
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    image_file = IMAGES_DIR / f"{slug}.png"
    image_file.write_bytes(image_data)
    print(f"  Saved: {image_file.relative_to(PROJECT_ROOT)}")

    # Update frontmatter
    image_path = f"/images/quotes/{slug}.png"
    write_image_field(filepath, image_path)
    print(f"  Updated frontmatter with image = '{image_path}'")
    return True


def main() -> None:
    # Load .env from project root
    load_dotenv(PROJECT_ROOT / ".env")

    comfyui_url = os.environ.get("COMFYUI_URL", DEFAULT_COMFYUI_URL).rstrip("/")

    # Check that ComfyUI is reachable
    try:
        resp = requests.get(f"{comfyui_url}/system_stats", timeout=5)
        resp.raise_for_status()
        print(f"Connected to ComfyUI at {comfyui_url}")
    except requests.RequestException:
        print(f"Cannot reach ComfyUI at {comfyui_url}. Skipping image generation.")
        sys.exit(0)

    cfg = {
        "flux_unet": os.environ.get("FLUX_UNET", DEFAULT_FLUX_UNET),
        "flux_clip1": os.environ.get("FLUX_CLIP1", DEFAULT_FLUX_CLIP1),
        "flux_clip2": os.environ.get("FLUX_CLIP2", DEFAULT_FLUX_CLIP2),
        "flux_vae": os.environ.get("FLUX_VAE", DEFAULT_FLUX_VAE),
    }

    # Determine which files to process
    if len(sys.argv) > 1:
        files = [Path(f).resolve() for f in sys.argv[1:]]
    else:
        files = sorted(QUOTES_DIR.glob("*.md"))

    # Filter out _index.md
    files = [f for f in files if f.name != "_index.md"]

    if not files:
        print("No quote files to process.")
        return

    for filepath in files:
        if not filepath.exists():
            print(f"WARNING: File not found: {filepath}", file=sys.stderr)
            continue
        process_file(comfyui_url, cfg, filepath)

    print("Done.")


if __name__ == "__main__":
    main()
