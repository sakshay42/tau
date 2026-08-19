import math
from datetime import datetime, timezone


def recency_weight(published_at, tau_hours, now=None):
    """exp(-age_hours / tau_hours); 1.0 when published_at is unknown."""
    if published_at is None:
        return 1.0

    if now is None:
        now = datetime.now(timezone.utc)

    age_hours = (now - published_at).total_seconds() / 3600

    age_hours = max(age_hours, 0)

    return math.exp(-age_hours / tau_hours)


def apply_temporal_decay(results, tau_hours, now=None):
    """Multiply each rerank_score by its recency_weight and sort by the resulting final_score, descending."""
    scored = []

    for result in results:
        weight = recency_weight(
            result["published_at"],
            tau_hours,
            now=now,
        )

        final_score = result["rerank_score"] * weight

        scored.append({
            **result,
            "recency_weight": weight,
            "final_score": final_score,
        })

    return sorted(
        scored,
        key=lambda x: x["final_score"],
        reverse=True,
    )