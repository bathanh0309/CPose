"""
CPose — app/utils/file_handler.py
Storage management and filesystem utilities.
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("[Storage]")


class FileHandler:
    """Filesystem utility: list files, enforce storage limits, parse configs."""

    # ──────────────────────────────────────────────────────────────────────
    def parse_resources(self, resources_file: Path) -> list[dict]:
        """
        Parse resources.txt → list of camera dicts.
        Format: one RTSP URL per line; blank lines and # comments are ignored.

        Returns:
            [{"cam_id": "01", "url": "rtsp://..."}, ...]
        """
        cameras = []
        idx = 1
        with open(resources_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                cameras.append({
                    "cam_id": str(idx).zfill(2),
                    "url": line,
                })
                idx += 1
        logger.info("Parsed %d cameras from %s", len(cameras), resources_file.name)
        return cameras

    # ──────────────────────────────────────────────────────────────────────
    def list_videos(self, raw_videos_dir: Path) -> list[dict]:
        """
        List all .mp4 clips in raw_videos directory.
        Returns list sorted by modification time (newest first).
        """
        files = sorted(
            raw_videos_dir.glob("*.mp4"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        result = []
        for f in files:
            stat = f.stat()
            result.append({
                "filename": f.name,
                "size_mb": round(stat.st_size / 1e6, 2),
                "mtime": stat.st_mtime,
            })
        return result

    # ──────────────────────────────────────────────────────────────────────
    def list_results(self, output_dir: Path) -> list[dict]:
        """
        List Phase 2 output folders.
        Returns each clip's result folder with PNG count and label line count.
        """
        results = []
        for folder in sorted(output_dir.iterdir()):
            if not folder.is_dir():
                continue
            pngs  = list(folder.glob("*.png"))
            labels = list(folder.glob("*_labels.txt"))
            label_lines = 0
            if labels:
                with open(labels[0], encoding="utf-8") as f:
                    label_lines = sum(1 for ln in f if ln.strip() and not ln.startswith("#"))
            results.append({
                "clip_stem": folder.name,
                "frames": len(pngs),
                "label_count": label_lines,
                "label_file": labels[0].name if labels else None,
            })
        return results

    # ──────────────────────────────────────────────────────────────────────
    def storage_info(self, raw_videos_dir: Path) -> dict:
        """Return total size and file count of raw_videos directory."""
        files = list(raw_videos_dir.glob("*.mp4"))
        total_bytes = sum(f.stat().st_size for f in files)
        return {
            "used_gb": round(total_bytes / 1e9, 3),
            "used_mb": round(total_bytes / 1e6, 1),
            "file_count": len(files),
        }

    # ──────────────────────────────────────────────────────────────────────
    def enforce_storage_limit(self, raw_videos_dir: Path, limit_gb: float):
        """
        If total size of raw_videos exceeds limit_gb, delete oldest clips
        until usage is back under the limit.
        Oldest = lowest mtime (recorded earliest).
        """
        limit_bytes = limit_gb * 1e9
        files = sorted(
            raw_videos_dir.glob("*.mp4"),
            key=lambda p: p.stat().st_mtime,   # ascending → oldest first
        )
        total = sum(f.stat().st_size for f in files)

        if total <= limit_bytes:
            return

        logger.warning(
            "Storage %.2f GB exceeds limit %.2f GB — pruning oldest clips",
            total / 1e9, limit_gb,
        )

        for f in files:
            if total <= limit_bytes:
                break
            size = f.stat().st_size
            f.unlink()
            total -= size
            logger.info("Deleted oldest clip: %s (%.1f MB)", f.name, size / 1e6)

        from app import socketio
        socketio.emit("storage_warning", {
            "used_gb": round(total / 1e9, 3),
            "limit_gb": limit_gb,
            "pct": round(total / limit_bytes * 100),
        })
