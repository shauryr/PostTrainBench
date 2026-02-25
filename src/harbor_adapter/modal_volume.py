"""Populate a Modal volume with pre-cached HuggingFace models and datasets.

Downloads all models and datasets listed in containers/download_hf_cache/resources.json
into a Modal volume so that trial sandboxes can mount it as a shared, read-only cache
(via fuse-overlayfs, see hooks.py).

Standalone usage (recommended for the first-time population):
    modal run src/harbor_adapter/modal_volume.py
    modal run src/harbor_adapter/modal_volume.py --models-only
    modal run src/harbor_adapter/modal_volume.py --datasets-only

Programmatic usage (called automatically by run_job.py):
    from modal_volume import ensure_hf_cache
    ensure_hf_cache()   # idempotent, skips already-cached resources
"""

import json
import os
from pathlib import Path

import modal

DEFAULT_VOLUME_NAME = "posttrainbench-hf-cache"
HF_HOME_MOUNT = "/hf-home"

RESOURCES_PATH = (
    Path(__file__).parent.parent.parent
    / "containers"
    / "download_hf_cache"
    / "resources.json"
)

app = modal.App("posttrainbench-cache")
volume = modal.Volume.from_name(DEFAULT_VOLUME_NAME, create_if_missing=True)

_download_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "huggingface_hub[hf_transfer]",
        "datasets",
    )
)

def _resolve_hf_token() -> str | None:
    """Read HF_TOKEN from the environment or the HuggingFace token file."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    # Fall back to the stored token (written by `huggingface-cli login`).
    hf_home = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    token_file = hf_home / "token"
    if token_file.is_file():
        return token_file.read_text().strip()
    return None


# Forward HF_TOKEN into the Modal function so gated model/dataset
# downloads work.
_secrets: list[modal.Secret] = []
_hf_token = _resolve_hf_token()
if _hf_token:
    _secrets.append(modal.Secret.from_dict({"HF_TOKEN": _hf_token}))


@app.function(
    image=_download_image,
    volumes={HF_HOME_MOUNT: volume},
    secrets=_secrets,
    timeout=43200,  # 12 hours (first run downloads ~500GB)
)
def download_resources(
    models: list[str],
    datasets_list: list[dict],
) -> dict[str, int]:
    """Download models and datasets into the mounted volume.

    Runs remotely on Modal.  Skips resources whose cache directories already
    exist, so repeated calls are cheap.
    """
    os.environ["HF_HOME"] = HF_HOME_MOUNT
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    # Keep temp files on the volume so large downloads don't exhaust the
    # container's ephemeral disk.
    os.environ["TMPDIR"] = f"{HF_HOME_MOUNT}/tmp"
    os.makedirs(os.environ["TMPDIR"], exist_ok=True)

    from huggingface_hub import snapshot_download
    from datasets import load_dataset as hf_load_dataset

    hub_dir = Path(HF_HOME_MOUNT) / "hub"
    downloaded = 0
    skipped = 0
    failed = 0

    # ---- Models ----
    for i, model_name in enumerate(models, 1):
        cache_dir = hub_dir / f"models--{model_name.replace('/', '--')}"
        if cache_dir.exists() and any(cache_dir.iterdir()):
            print(f"[Model {i}/{len(models)}] Cached: {model_name}")
            skipped += 1
            continue
        try:
            print(f"[Model {i}/{len(models)}] Downloading: {model_name}...")
            snapshot_download(model_name)
            downloaded += 1
            volume.commit()
            print(f"[Model {i}/{len(models)}] Done: {model_name}")
        except Exception as e:
            print(f"[Model {i}/{len(models)}] FAILED: {model_name}: {e}")
            failed += 1

    # ---- Datasets ----
    for i, entry in enumerate(datasets_list, 1):
        name = entry["dataset"]
        cache_dir = hub_dir / f"datasets--{name.replace('/', '--')}"
        if cache_dir.exists() and any(cache_dir.iterdir()):
            print(f"[Dataset {i}/{len(datasets_list)}] Cached: {name}")
            skipped += 1
            continue

        configs = entry.get("configs", [entry.get("config", "default")])
        splits = entry.get("splits", [])

        try:
            print(f"[Dataset {i}/{len(datasets_list)}] Downloading: {name}...")
            for config in configs:
                kwargs: dict[str, str] = {}
                if config and config != "default":
                    kwargs["name"] = config
                if splits:
                    for split in splits:
                        hf_load_dataset(name, split=split, **kwargs)
                else:
                    hf_load_dataset(name, **kwargs)
            downloaded += 1
            if downloaded % 20 == 0:
                volume.commit()
            print(f"[Dataset {i}/{len(datasets_list)}] Done: {name}")
        except Exception as e:
            print(f"[Dataset {i}/{len(datasets_list)}] FAILED: {name}: {e}")
            failed += 1

    volume.commit()
    return {"downloaded": downloaded, "skipped": skipped, "failed": failed}


@app.local_entrypoint()
def main(
    resources: str = str(RESOURCES_PATH),
    models_only: bool = False,
    datasets_only: bool = False,
):
    """Download all resources from resources.json into the Modal volume."""
    data = json.loads(Path(resources).read_text())
    models = [] if datasets_only else data.get("models", [])
    datasets = [] if models_only else data.get("datasets", [])

    print(f"Volume: {DEFAULT_VOLUME_NAME}")
    print(f"Models: {len(models)}")
    print(f"Datasets: {len(datasets)}")
    print()

    result = download_resources.remote(models, datasets)
    print(
        f"\nDownloaded: {result['downloaded']}, "
        f"Skipped: {result['skipped']}, "
        f"Failed: {result['failed']}"
    )


def ensure_hf_cache(
    resources_path: Path = RESOURCES_PATH,
) -> str:
    """Ensure the HF cache volume exists and is populated.

    Idempotent: skips resources that are already cached, so repeated calls
    are fast.  Called automatically from run_job.py before launching trials.

    Returns the volume name.
    """
    data = json.loads(resources_path.read_text())
    models = data.get("models", [])
    datasets_list = data.get("datasets", [])

    print(f"Ensuring HF cache volume '{DEFAULT_VOLUME_NAME}' is populated...")
    print(f"  Models: {len(models)}, Datasets: {len(datasets_list)}")

    with modal.enable_output():
        with app.run():
            result = download_resources.remote(models, datasets_list)

    print(
        f"  Downloaded: {result['downloaded']}, "
        f"Skipped: {result['skipped']}, "
        f"Failed: {result['failed']}"
    )
    return DEFAULT_VOLUME_NAME
