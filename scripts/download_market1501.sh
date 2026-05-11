#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${1:-"$ROOT_DIR/dataset/market1501"}"
METHOD="${2:-auto}" # auto | gdown | kaggle
KEEP_ZIPS="${KEEP_ZIPS:-0}"
GDRIVE_ID="${MARKET1501_GDRIVE_ID:-0B8-rUzbwVRk0c054eEozWG9COHM}"
KAGGLE_DATASET="${MARKET1501_KAGGLE_DATASET:-pengcw1/market-1501}"
DATASET_DIR="$TARGET_DIR/Market-1501-v15.09.15"

mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

if [[ -d "$DATASET_DIR/bounding_box_train" && -d "$DATASET_DIR/query" && -d "$DATASET_DIR/bounding_box_test" ]]; then
  echo "[SKIP] Market-1501 already exists at $DATASET_DIR"
  exit 0
fi

use_gdown() {
  if command -v gdown >/dev/null 2>&1; then
    gdown --id "$GDRIVE_ID" -O Market-1501.zip
  else
    python -m gdown --id "$GDRIVE_ID" -O Market-1501.zip
  fi
}

use_kaggle() {
  if command -v kaggle >/dev/null 2>&1; then
    kaggle datasets download -d "$KAGGLE_DATASET" -p "$TARGET_DIR"
  else
    python -m kaggle datasets download -d "$KAGGLE_DATASET" -p "$TARGET_DIR"
  fi
  if [[ -f market-1501.zip && ! -f Market-1501.zip ]]; then
    mv market-1501.zip Market-1501.zip
  fi
}

echo "[INFO] Downloading Market-1501 into $TARGET_DIR by method=$METHOD"
case "$METHOD" in
  gdown)
    use_gdown
    ;;
  kaggle)
    use_kaggle
    ;;
  auto)
    if python -c "import gdown" >/dev/null 2>&1 || command -v gdown >/dev/null 2>&1; then
      use_gdown
    elif command -v kaggle >/dev/null 2>&1 || python -c "import kaggle" >/dev/null 2>&1; then
      use_kaggle
    else
      echo "[ERROR] Neither gdown nor kaggle is available."
      echo "        Install one of them, or run:"
      echo "        python -m pip install gdown"
      echo "        python -m pip install kaggle"
      exit 2
    fi
    ;;
  *)
    echo "[ERROR] Unknown method: $METHOD"
    echo "        Valid: auto, gdown, kaggle"
    exit 2
    ;;
esac

if [[ ! -f Market-1501.zip ]]; then
  echo "[ERROR] Market-1501.zip was not downloaded."
  exit 1
fi

if command -v unzip >/dev/null 2>&1; then
  unzip -q Market-1501.zip
else
  python - Market-1501.zip <<'PY'
from pathlib import Path
from zipfile import ZipFile
import sys

with ZipFile(Path(sys.argv[1])) as archive:
    archive.extractall(".")
PY
fi

if [[ "$KEEP_ZIPS" != "1" ]]; then
  rm -f Market-1501.zip market-1501.zip
fi

python - "$TARGET_DIR" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1]) / "Market-1501-v15.09.15"
required = ["bounding_box_train", "bounding_box_test", "query", "gt_bbox"]
missing = [name for name in required if not (root / name).exists()]
if missing:
    raise SystemExit("[ERROR] Missing Market-1501 folders: " + ", ".join(missing))

counts = {name: len(list((root / name).glob("*.jpg"))) for name in required}
print("[OK] Market-1501 ready:", counts)
PY

echo "[DONE] Market-1501 ready at $DATASET_DIR"
