"""Post-deployment monitoring contract and runtime observability.

Defines signal categories to monitor after serving begins. At this stage,
the contract is declarative — full metric computation requires live traffic
or delivered outcome labels that arrive after bookings complete.

What IS observable today (without live traffic):
    - Snapshot freshness and staleness warning
    - Model artifact identity and version
    - Feature schema and cardinality
    - Training-data distribution summary (from artifact)
    - Release gate status
    - Test suite status

What requires live traffic (future):
    - Feature drift vs training distribution
    - Unknown category rate
    - Prediction distribution shift
    - Evidence-tier distribution
    - Abstention/refusal rate

What requires outcome labels (future):
    - Realised MAE / median AE
    - Segment MAE (room type, neighbourhood, price band)
    - Conformal coverage
    - Conditional coverage
    - New-host vs seen-host performance
    - Premium-segment bias
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any


@dataclass
class MonitoringSignal:
    """A single monitoring signal with current state."""

    name: str
    category: str  # input, prediction, quality
    status: str  # healthy, warning, critical, not_available
    observed: Any
    threshold: Any | None = None
    description: str = ""


@dataclass
class MonitoringReport:
    """Snapshot of current monitoring state."""

    generated_at: str
    artifact_snapshot: str
    artifact_age_days: int
    deployment_authority: str
    signals: list[MonitoringSignal] = field(default_factory=list)


def compute_monitoring(artifact: dict[str, Any]) -> MonitoringReport:
    """Produce the current monitoring report from runtime-observable signals."""
    snapshot_label = artifact.get("snapshot_label", "unknown")
    try:
        training_date = date.fromisoformat(
            artifact.get("latest_training_as_of_date", "2000-01-01")
        )
    except ValueError:
        training_date = date.today() - timedelta(days=999)
    age_days = (date.today() - training_date).days

    authority = artifact.get("deployment_authority", "research_only")

    signals = [
        MonitoringSignal(
            name="snapshot_freshness",
            category="input",
            status="warning" if age_days > 90 else "healthy",
            observed=age_days,
            threshold="<= 90 days",
            description="Days since the training snapshot's as-of date",
        ),
        MonitoringSignal(
            name="snapshot_staleness",
            category="input",
            status="critical" if age_days > 120 else ("warning" if age_days > 90 else "healthy"),
            observed=age_days,
            threshold="<= 120 days (refuse) / <= 90 days (warn)",
            description="Staleness gate: predictions refused at >120 days",
        ),
        MonitoringSignal(
            name="deployment_authority",
            category="input",
            status="warning" if authority != "temporally_validated" else "healthy",
            observed=authority,
            threshold="temporally_validated",
            description="Model authority level",
        ),
        MonitoringSignal(
            name="temporal_validation",
            category="input",
            status=(
                "healthy"
                if artifact.get("temporal_validation_status") == "TEMPORAL_PRICE_VALIDATION_READY"
                else "critical"
            ),
            observed=artifact.get("temporal_validation_status", "NOT_ASSESSED"),
            threshold="TEMPORAL_PRICE_VALIDATION_READY",
            description="Whether forward-time validation evidence exists",
        ),
        MonitoringSignal(
            name="feature_schema_version",
            category="input",
            status="healthy",
            observed={
                "numeric": len(artifact.get("numeric_features", [])),
                "categorical": len(artifact.get("categorical_features", [])),
            },
            description="Feature schema stability (detect drift on new snapshots)",
        ),
        MonitoringSignal(
            name="prediction_price_range",
            category="prediction",
            status="healthy",
            observed=artifact.get("supported_price_range", [0, 0]),
            description="Supported prediction range [p01, p99] from training data",
        ),
        MonitoringSignal(
            name="conformal_coverage_target",
            category="prediction",
            status="healthy",
            observed="90% nominal",
            threshold=">= 85% marginal",
            description="Split conformal interval target coverage rate",
        ),
        MonitoringSignal(
            name="refusal_gate_thresholds",
            category="prediction",
            status="healthy",
            observed=artifact.get("gate_thresholds", {}),
            description="Predeclared evidence gate thresholds in effect",
        ),
        MonitoringSignal(
            name="feature_drift_detection",
            category="input",
            status="not_available",
            observed=None,
            threshold="requires live traffic or new snapshot",
            description="Feature distribution drift vs training — not yet observable",
        ),
        MonitoringSignal(
            name="unknown_category_rate",
            category="input",
            status="not_available",
            observed=None,
            threshold="requires live traffic",
            description="Rate of unseen categorical values in production requests",
        ),
        MonitoringSignal(
            name="evidence_tier_distribution",
            category="prediction",
            status="not_available",
            observed=None,
            threshold="requires live traffic",
            description="HIGH/MEDIUM/LOW/REFUSE distribution in production",
        ),
        MonitoringSignal(
            name="realised_mae",
            category="quality",
            status="not_available",
            observed=None,
            threshold="requires outcome labels (post-booking prices)",
            description="Realised prediction error when outcome labels arrive",
        ),
        MonitoringSignal(
            name="segment_performance",
            category="quality",
            status="not_available",
            observed=None,
            threshold="requires outcome labels + sufficient segment volumes",
            description="Per-segment MAE/coverage (room type, neighbourhood, price band)",
        ),
    ]

    return MonitoringReport(
        generated_at=date.today().isoformat(),
        artifact_snapshot=snapshot_label,
        artifact_age_days=age_days,
        deployment_authority=authority,
        signals=signals,
    )


def monitoring_to_dict(report: MonitoringReport) -> dict[str, Any]:
    """Serialize monitoring report to JSON-compatible dict."""
    return {
        "report_version": 1,
        "generated_at": report.generated_at,
        "artifact_snapshot": report.artifact_snapshot,
        "artifact_age_days": report.artifact_age_days,
        "deployment_authority": report.deployment_authority,
        "summary": {
            "healthy": sum(1 for s in report.signals if s.status == "healthy"),
            "warning": sum(1 for s in report.signals if s.status == "warning"),
            "critical": sum(1 for s in report.signals if s.status == "critical"),
            "not_available": sum(1 for s in report.signals if s.status == "not_available"),
        },
        "signals": [
            {
                "name": s.name,
                "category": s.category,
                "status": s.status,
                "observed": s.observed,
                "threshold": s.threshold,
                "description": s.description,
            }
            for s in report.signals
        ],
    }
