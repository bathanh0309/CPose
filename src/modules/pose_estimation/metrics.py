from __future__ import annotations


def build_pose_metrics(records: list[dict], elapsed_sec: float, output_video_path: str, output_json_path: str) -> dict:
    total_instances = 0
    keypoint_count = 0
    visible_count = 0
    confidence_sum = 0.0
    failed_frames = 0
    for record in records:
        persons = record["persons"]
        if not persons:
            failed_frames += 1
        total_instances += len(persons)
        for person in persons:
            if person.get("visible_keypoint_count") is not None:
                visible_count += int(person.get("visible_keypoint_count", 0))
                keypoint_count += len(person.get("keypoints", []))
                confidence_sum += sum(float(k.get("confidence", 0.0)) for k in person.get("keypoints", []))
                continue
            for keypoint in person["keypoints"]:
                keypoint_count += 1
                confidence = float(keypoint.get("confidence", 0.0))
                confidence_sum += confidence
                if confidence >= 0.25:
                    visible_count += 1
    total_frames = len(records)
    return {
        "total_frames": total_frames,
        "total_pose_instances": total_instances,
        "mean_keypoint_confidence": confidence_sum / keypoint_count if keypoint_count else 0.0,
        "visible_keypoint_ratio": visible_count / keypoint_count if keypoint_count else 0.0,
        "missing_keypoint_rate": 1.0 - (visible_count / keypoint_count) if keypoint_count else 0.0,
        "pose_failure_rate": failed_frames / total_frames if total_frames else 0.0,
        "fps_processing": total_frames / elapsed_sec if elapsed_sec > 0 else 0.0,
        "avg_latency_ms_per_frame": (elapsed_sec / total_frames * 1000.0) if total_frames else 0.0,
        "elapsed_sec": elapsed_sec,
        "output_video_path": output_video_path,
        "output_json_path": output_json_path,
    }
