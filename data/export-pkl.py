"""
Export one person's embedding folder into a single pickle file.

Default input:
  D:/Capstone_Project/data/embeddings/APhu

Default output:
  D:/Capstone_Project/data/embeddings/APhu/APhu_embeddings.pkl
"""

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np

DEFAULT_PERSON_ID = "Huy"
DEFAULT_EMBEDDING_DIR = "D:/Capstone_Project/data/embeddings/Huy"


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    vector = vector.astype(np.float32).reshape(-1)
    return vector / (np.linalg.norm(vector) + 1e-12)


def load_vectors(paths: list[Path]) -> tuple[np.ndarray, list[str]]:
    vectors: list[np.ndarray] = []
    names: list[str] = []

    for path in paths:
        try:
            vector = np.load(str(path)).astype(np.float32).reshape(-1)
        except Exception as exc:
            print(f"[WARN] Skip unreadable embedding: {path.name}: {exc}")
            continue

        vectors.append(l2_normalize(vector))
        names.append(path.name)

    if not vectors:
        return np.empty((0, 0), dtype=np.float32), names

    dims = {vec.shape[0] for vec in vectors}
    if len(dims) != 1:
        raise ValueError(f"Mixed embedding dimensions found: {sorted(dims)}")

    return np.stack(vectors, axis=0).astype(np.float32), names


def make_prototype(vectors: np.ndarray) -> np.ndarray | None:
    if vectors.size == 0:
        return None
    prototype = np.mean(vectors, axis=0)
    return l2_normalize(prototype)


def read_meta(embedding_dir: Path) -> dict:
    meta_path = embedding_dir / "meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] Cannot read meta.json: {exc}")
        return {}


def export_embeddings(person_id: str, embedding_dir: Path, output_path: Path) -> Path:
    body_paths = sorted(embedding_dir.glob("body_*.npy"))
    face_paths = sorted(embedding_dir.glob("face_*.npy"))

    body_vectors, body_files = load_vectors(body_paths)
    face_vectors, face_files = load_vectors(face_paths)

    data = {
        "format": "cpose_embedding_gallery_v1",
        "person_id": person_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_dir": str(embedding_dir),
        "meta": read_meta(embedding_dir),
        "body": {
            "model": "osnet_x0_25",
            "files": body_files,
            "embeddings": body_vectors,
            "prototype": make_prototype(body_vectors),
            "count": int(body_vectors.shape[0]),
            "dim": int(body_vectors.shape[1]) if body_vectors.ndim == 2 and body_vectors.size else 0,
        },
        "face": {
            "files": face_files,
            "embeddings": face_vectors,
            "prototype": make_prototype(face_vectors),
            "count": int(face_vectors.shape[0]),
            "dim": int(face_vectors.shape[1]) if face_vectors.ndim == 2 and face_vectors.size else 0,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--person_id", default=DEFAULT_PERSON_ID)
    parser.add_argument("--embedding_dir", default=DEFAULT_EMBEDDING_DIR)
    parser.add_argument(
        "--output",
        default=None,
        help="Output pkl path. Default: <embedding_dir>/<person_id>_embeddings.pkl",
    )
    args = parser.parse_args()

    embedding_dir = Path(args.embedding_dir)
    if not embedding_dir.exists():
        raise FileNotFoundError(f"Embedding folder does not exist: {embedding_dir}")

    output_path = (
        Path(args.output)
        if args.output
        else embedding_dir / f"{args.person_id}_embeddings.pkl"
    )

    saved_path = export_embeddings(args.person_id, embedding_dir, output_path)

    with saved_path.open("rb") as file:
        exported = pickle.load(file)

    print(f"Saved: {saved_path}")
    print(f"Person ID: {exported['person_id']}")
    print(f"Body embeddings: {exported['body']['count']} dim={exported['body']['dim']}")
    print(f"Face embeddings: {exported['face']['count']} dim={exported['face']['dim']}")


if __name__ == "__main__":
    main()
