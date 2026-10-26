"""Download the Demucs model into build/models/<name> for the installer to bundle."""

import shutil
import sys
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download

from songpolish.separation import MODEL_NAME

REPO = "adefossez/HTDemucs"


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "build/models") / MODEL_NAME
    out.mkdir(parents=True, exist_ok=True)
    bag = hf_hub_download(REPO, f"{MODEL_NAME}.yaml")
    shutil.copy(bag, out / f"{MODEL_NAME}.yaml")
    for sig in yaml.safe_load(Path(bag).read_text())["models"]:
        shutil.copy(hf_hub_download(REPO, f"{sig}.safetensors"), out / f"{sig}.safetensors")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
