#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${1:-"$ROOT_DIR/dataset/coco"}"
KEEP_ZIPS="${KEEP_ZIPS:-0}"

VAL_URL="http://images.cocodataset.org/zips/val2017.zip"
ANN_URL="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

download() {
  local url="$1"
  local output="$2"
  if [[ -f "$output" ]]; then
    echo "[SKIP] $output already exists"
    return
  fi
  if command -v wget >/dev/null 2>&1; then
    wget -c "$url" -O "$output"
  elif command -v curl >/dev/null 2>&1; then
    curl -L --fail --continue-at - "$url" -o "$output"
  else
    python - "$url" "$output" <<'PY'
from pathlib import Path
from urllib.request import urlretrieve
import sys

url, output = sys.argv[1], Path(sys.argv[2])
urlretrieve(url, output)
PY
  fi
}

extract_zip() {
  local zip_file="$1"
  local marker="$2"
  if [[ -e "$marker" ]]; then
    echo "[SKIP] $marker already exists"
    return
  fi
  if command -v unzip >/dev/null 2>&1; then
    unzip -q "$zip_file"
  else
    python - "$zip_file" <<'PY'
from pathlib import Path
from zipfile import ZipFile
import sys

with ZipFile(Path(sys.argv[1])) as archive:
    archive.extractall(".")
PY
  fi
}

echo "[INFO] Downloading COCO val2017 into $TARGET_DIR"
download "$VAL_URL" "val2017.zip"
download "$ANN_URL" "annotations_trainval2017.zip"

extract_zip "val2017.zip" "val2017"
extract_zip "annotations_trainval2017.zip" "annotations/person_keypoints_val2017.json"

if [[ "$KEEP_ZIPS" != "1" ]]; then
  rm -f val2017.zip annotations_trainval2017.zip
fi

python - "$TARGET_DIR" <<'PY'
from pathlib import Path
import json
import sys

root = Path(sys.argv[1])
required = [
    root / "val2017",
    root / "annotations" / "instances_val2017.json",
    root / "annotations" / "person_keypoints_val2017.json",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit("[ERROR] Missing COCO files:\n" + "\n".join(missing))

try:
    from pycocotools.coco import COCO
except Exception:
    keypoints = json.loads((root / "annotations" / "person_keypoints_val2017.json").read_text(encoding="utf-8"))
    print(f"[OK] COCO keypoints JSON loaded, images: {len(keypoints.get('images', []))}")
else:
    coco = COCO(str(root / "annotations" / "person_keypoints_val2017.json"))
    print(f"[OK] COCO keypoints OK, images: {len(coco.imgs)}")
PY

echo "[DONE] COCO val2017 ready at $TARGET_DIR"
