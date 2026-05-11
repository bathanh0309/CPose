"""Global ReID metric helpers."""
from __future__ import annotations


def build_reid_summary(results: list[dict], counts: dict) -> dict:
    score_values = [row.get("avg_score_total") for row in results if row.get("avg_score_total") is not None]
    return {
        "metric_type": "proxy",
        "global_id_count": int(counts.get("global_id_count", counts.get("total_global_ids", counts.get("new_id_count", 0)))),
        "pending_count": int(counts.get("pending_count", 0)),
        "conflict_count": int(counts.get("multi_candidate_conflict_count", counts.get("conflict_count", 0))),
        "topology_conflict_count": int(counts.get("topology_conflict_count", 0)),
        "unknown_match_count": int(counts.get("unknown_match_count", 0)),
        "avg_score_total": sum(score_values) / len(score_values) if score_values else None,
    }


__all__ = ["build_reid_summary"]
