from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.common.json_io import save_json
from src.config import LIVENESS_MODEL, load_model_registry
from src.paths import resolve_path
from src.schemas import FaceEvent, ModuleMetrics, to_dict
from src.video_io import camera_id_from_video_name, create_video_writer, show_frame_preview


def _source_label(source: str) -> str:
    if source.isdigit():
        return f"webcam_{source}"
    if "://" in source:
        return "stream"
    return Path(source).stem or "source"


def _open_capture(source: str) -> cv2.VideoCapture:
    capture_source: int | str
    if source.isdigit():
        capture_source = int(source)
    elif "://" in source:
        capture_source = source
    else:
        capture_source = str(resolve_path(source))

    capture = cv2.VideoCapture(capture_source)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open face demo source: {source}")
    return capture


def _to_json_embedding(embedding: Any) -> list[float] | None:
    if embedding is None:
        return None
    array = np.asarray(embedding, dtype=np.float32).reshape(-1)
    return [round(float(value), 6) for value in array.tolist()]


def _safe_liveness(recognizer: Any, frame: np.ndarray, bbox: list[int]) -> dict[str, Any]:
    try:
        from src.core.face_recognizer import crop

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_crop = crop(rgb, tuple(bbox), 1.5)
        face_crop = cv2.resize(face_crop, (128, 128))
        return recognizer.is_real_face(face_crop)
    except Exception as exc:
        return {
            "is_real": None,
            "status": "unknown",
            "prob_real": None,
            "confidence": None,
            "reason": str(exc),
        }


def _draw_face(
    frame: np.ndarray,
    bbox: list[int],
    identity: str,
    similarity: float | None,
    face_score: float | None,
    spoof_status: str,
    spoof_conf: float | None,
) -> None:
    known = identity != "unknown"
    color = (0, 220, 0) if known else (0, 190, 255)
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    parts = [identity]
    if similarity is not None:
        parts.append(f"sim={similarity:.2f}")
    if face_score is not None:
        parts.append(f"det={face_score:.2f}")
    if spoof_status:
        if spoof_conf is None:
            parts.append(spoof_status)
        else:
            parts.append(f"{spoof_status}={spoof_conf:.2f}")

    label = " | ".join(parts)
    y_text = max(20, y1 - 8)
    cv2.putText(frame, label, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def run_face_demo(
    source: str,
    output_dir: str | Path,
    config: str | Path | None = None,
    gallery: str | Path = "data/face",
    liveness_model: str | Path | None = LIVENESS_MODEL,
    camera_id: str | None = None,
    run_every_n_frames: int | None = None,
    max_frames: int = 0,
    anti_spoof: bool = False,
    preview: bool = True,
    save_video: bool = True,
) -> Path:
    registry = load_model_registry(config)
    face_cfg = registry.get("face", {}) if isinstance(registry.get("face"), dict) else {}
    threshold = float(face_cfg.get("cosine_threshold", face_cfg.get("threshold", 0.45)))
    spoof_threshold = float(face_cfg.get("anti_spoof_score", face_cfg.get("spoof_threshold", 0.50)))
    frame_step = max(1, int(run_every_n_frames or face_cfg.get("run_every_n_frames", 1)))

    liveness_path = Path(liveness_model) if liveness_model else None
    use_liveness = bool(anti_spoof and liveness_path and resolve_path(liveness_path).exists())

    from src.core.face_recognizer import FaceRecognizer

    recognizer = FaceRecognizer(
        gallery_dir=gallery,
        liveness_model=resolve_path(liveness_path) if use_liveness and liveness_path else None,
        threshold=threshold,
        spoof_threshold=spoof_threshold,
    )

    capture = _open_capture(source)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    source_name = _source_label(source)
    camera = camera_id or (source_name if source.isdigit() else camera_id_from_video_name(source_name))
    output_base = resolve_path(output_dir) / source_name
    output_base.mkdir(parents=True, exist_ok=True)

    writer = None
    if save_video and width > 0 and height > 0:
        writer = create_video_writer(output_base / "face_overlay.mp4", fps or 25.0, width, height)

    started = time.perf_counter()
    frame_id = 0
    total_frames = 0
    processed_frames = 0
    events: list[dict[str, Any]] = []
    last_faces: list[dict[str, Any]] = []

    try:
        while True:
            if max_frames > 0 and frame_id >= max_frames:
                break
            ok, frame = capture.read()
            if not ok:
                break
            total_frames += 1

            timestamp_sec = frame_id / fps if fps > 0 else time.perf_counter() - started

            if frame_id % frame_step == 0:
                last_faces = recognizer.detect(frame)
                processed_frames += 1

                for track_id, face in enumerate(last_faces, start=1):
                    bbox = [int(value) for value in face.get("bbox", [])]
                    embedding = face.get("embedding")
                    if embedding is not None:
                        person_id, similarity = recognizer.match(embedding)
                        similarity = float(similarity) if similarity is not None else None
                    else:
                        person_id, similarity = None, None
                    identity = str(person_id) if person_id and similarity is not None and similarity >= threshold else "unknown"

                    liveness = _safe_liveness(recognizer, frame, bbox) if use_liveness else {
                        "status": "not_run",
                        "prob_real": None,
                        "confidence": None,
                    }
                    spoof_status = str(liveness.get("status") or "unknown")
                    spoof_conf = liveness.get("prob_real")
                    spoof_conf = float(spoof_conf) if spoof_conf is not None else None
                    face_score = face.get("score")
                    face_score = float(face_score) if face_score is not None else None

                    event = FaceEvent(
                        frame_id=frame_id,
                        timestamp_sec=round(float(timestamp_sec), 4),
                        camera_id=str(camera),
                        track_id=track_id,
                        face_detected=True,
                        face_bbox=[float(value) for value in bbox],
                        embedding_dim=len(np.asarray(embedding).reshape(-1)) if embedding is not None else None,
                        embedding=_to_json_embedding(embedding),
                        face_quality=face_score,
                        spoof_status=spoof_status,
                        failure_reason="OK" if identity != "unknown" else "NO_FACE_MATCH",
                    )
                    row = to_dict(event)
                    row["identity"] = identity
                    row["similarity"] = round(similarity, 4) if similarity is not None else None
                    row["spoof_prob_real"] = round(spoof_conf, 4) if spoof_conf is not None else None
                    events.append(row)

                    _draw_face(frame, bbox, identity, similarity, face_score, spoof_status, spoof_conf)

            else:
                for face in last_faces:
                    bbox = [int(value) for value in face.get("bbox", [])]
                    _draw_face(frame, bbox, "face", None, face.get("score"), "cached", None)

            cv2.putText(
                frame,
                f"CPose Face | frame={frame_id} | faces={len(last_faces)} | q/esc to stop",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if writer is not None:
                writer.write(frame)
            elif save_video:
                frame_h, frame_w = frame.shape[:2]
                writer = create_video_writer(output_base / "face_overlay.mp4", fps or 25.0, frame_w, frame_h)
                writer.write(frame)

            if preview and show_frame_preview("CPose Face Module", frame, fps or 25.0):
                break

            frame_id += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if preview:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    elapsed = time.perf_counter() - started
    save_json(output_base / "face_events.json", events)
    metrics = ModuleMetrics(
        metric_type="proxy",
        total_frames=total_frames,
        processed_frames=processed_frames,
        elapsed_sec=round(elapsed, 4),
        fps_processing=round(processed_frames / elapsed, 4) if elapsed > 0 else None,
        avg_latency_ms_per_frame=round((elapsed / processed_frames) * 1000.0, 4) if processed_frames else None,
        model_info={
            "face_detector": "insightface/buffalo_s",
            "gallery": str(gallery),
            "cosine_threshold": threshold,
            "anti_spoofing_enabled": use_liveness,
            "liveness_model": str(liveness_path) if liveness_path else None,
        },
        input_video=source,
        camera_id=str(camera),
        start_time=None,
        output_paths={
            "events": output_base / "face_events.json",
            "overlay": output_base / "face_overlay.mp4" if save_video else None,
        },
        failure_reason="OK",
    )
    save_json(output_base / "face_metrics.json", to_dict(metrics))
    print(f"[INFO] Face events: {output_base / 'face_events.json'}")
    print(f"[INFO] Face metrics: {output_base / 'face_metrics.json'}")
    if save_video:
        print(f"[INFO] Face overlay: {output_base / 'face_overlay.mp4'}")
    return output_base


def main() -> None:
    parser = argparse.ArgumentParser(description="Show CPose face detection, recognition, and anti-spoofing")
    parser.add_argument("--source", default="0", help="Webcam index, video path, or RTSP URL")
    parser.add_argument("--output", default="dataset/outputs/0_face")
    parser.add_argument("--config", default="configs/profiles/dev.yaml")
    parser.add_argument("--gallery", default="data/face")
    parser.add_argument("--liveness-model", default=str(LIVENESS_MODEL))
    parser.add_argument("--camera-id", default=None)
    parser.add_argument("--run-every-n-frames", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means run until source ends or user stops preview")
    parser.add_argument("--anti-spoof", action="store_true", help="Run ONNX anti-spoofing when the liveness model exists")
    parser.add_argument("--save-video", action="store_true")
    preview_group = parser.add_mutually_exclusive_group()
    preview_group.add_argument("--preview", dest="preview", action="store_true", default=True)
    preview_group.add_argument("--no-preview", dest="preview", action="store_false")
    args = parser.parse_args()

    run_face_demo(
        source=args.source,
        output_dir=args.output,
        config=args.config,
        gallery=args.gallery,
        liveness_model=args.liveness_model,
        camera_id=args.camera_id,
        run_every_n_frames=args.run_every_n_frames,
        max_frames=args.max_frames,
        anti_spoof=args.anti_spoof,
        preview=args.preview,
        save_video=args.save_video,
    )


if __name__ == "__main__":
    main()
