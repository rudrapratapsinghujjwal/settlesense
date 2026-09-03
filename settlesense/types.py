"""
SettleSense — Core Type Definitions
All domain types used across the pipeline.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ExceptionCategory(str, Enum):
    """The four exception categories SettleSense classifies."""
    SPLIT_SETTLEMENT = "split_settlement"
    REFUND_MISATTRIBUTION = "refund_misattribution"
    FEE_TIER = "fee_tier"
    NEAR_DUPLICATE = "near_duplicate"
    UNRESOLVED = "unresolved"
    CLEAN = "clean"  # Not an exception — matched successfully


class DecisionStatus(str, Enum):
    """Status of a pipeline decision."""
    AUTO_RESOLVED = "auto_resolved"
    HUMAN_REVIEW = "human_review"
    HUMAN_APPROVED = "human_approved"
    HUMAN_REJECTED = "human_rejected"
    HUMAN_OVERRIDDEN = "human_overridden"
    ESCALATED = "escalated"


class DataSource(str, Enum):
    """Which of the four data sources a record originates from."""
    RAZORPAY_PAYMENT = "razorpay_payment"
    RAZORPAY_RECON = "razorpay_recon"
    MERCHANT_LEDGER = "merchant_ledger"
    BANK_STATEMENT = "bank_statement"


class Currency(str, Enum):
    INR = "INR"
    USD = "USD"
    EUR = "EUR"


# ---------------------------------------------------------------------------
# Normalized Transaction (internal model)
# ---------------------------------------------------------------------------

@dataclass
class NormalizedTransaction:
    """
    Internal canonical representation after normalization.
    Preserves traceability to source.
    """
    txn_id: str                        # Unique internal ID
    source: DataSource
    source_record_id: str              # Original ID in source system
    amount: Decimal                    # Always in base currency unit (paise for INR)
    currency: Currency
    transaction_date: datetime
    description: str
    reference_ids: list[str]           # Payment IDs, order IDs, UTR refs etc.
    raw_fields: dict[str, Any]         # Preserved original fields
    is_synthetic: bool = True          # Will be False for real Razorpay data

    # Optional enriched fields
    fee: Optional[Decimal] = None
    tax: Optional[Decimal] = None
    settlement_id: Optional[str] = None
    order_id: Optional[str] = None
    payment_id: Optional[str] = None
    method: Optional[str] = None
    status: Optional[str] = None


# ---------------------------------------------------------------------------
# Evidence Item
# ---------------------------------------------------------------------------

@dataclass
class EvidenceItem:
    """
    One piece of evidence assembled for the LLM.
    Never contains untrusted text in instruction-like positions.
    """
    field: str
    value: str
    relevance: str
    source: DataSource


# ---------------------------------------------------------------------------
# Candidate Match
# ---------------------------------------------------------------------------

@dataclass
class CandidateMatch:
    """A candidate record proposed for matching against an exception record."""
    candidate_id: str
    match_score: float           # Composite similarity 0.0–1.0
    amount_proximity: float      # Normalized amount distance
    date_proximity_days: float
    description_similarity: float
    shared_reference_ids: list[str]
    candidate_record: NormalizedTransaction


# ---------------------------------------------------------------------------
# LLM Output (exactly the required schema)
# ---------------------------------------------------------------------------

@dataclass
class LLMOutput:
    """
    Validated structured output from the LLM classification call.
    Malformed outputs are rejected — never silently recovered.
    """
    record_id: str
    candidate_category: ExceptionCategory
    proposed_linked_ids: list[str]
    evidence_used: list[EvidenceItem]
    raw_model_signal: float          # 0.0–1.0 model's own confidence signal
    recommended_action: str
    reasoning_summary: str

    # Validation metadata
    is_valid: bool = True
    validation_error: Optional[str] = None
    hallucinated_evidence_detected: bool = False


# ---------------------------------------------------------------------------
# Confidence Signals
# ---------------------------------------------------------------------------

@dataclass
class ConfidenceSignals:
    """
    Independent signals used for confidence calibration.
    Never derived from LLM's self-reported confidence alone.
    """
    candidate_margin: float          # (best_score - second_best_score) / best_score
    rule_agreement: float            # 1.0 if deterministic rule agrees, 0.0 otherwise
    evidence_completeness: float     # Fraction of required fields present (0.0–1.0)
    raw_model_signal: float          # From LLM output
    calibrated_confidence: float = 0.0   # Set after calibration


# ---------------------------------------------------------------------------
# Pipeline Decision
# ---------------------------------------------------------------------------

@dataclass
class PipelineDecision:
    """
    Complete record of what the pipeline decided for one exception record.
    Stored in SQLite for audit trail.
    """
    decision_id: str
    record_id: str
    category: ExceptionCategory
    status: DecisionStatus
    confidence_signals: ConfidenceSignals
    llm_output: Optional[LLMOutput]
    threshold_used: float
    timestamp: datetime
    pipeline_stage: str              # Which stage produced this decision

    # Human review fields
    human_action: Optional[str] = None
    human_override_category: Optional[ExceptionCategory] = None
    human_notes: Optional[str] = None
    human_timestamp: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Evaluation Results
# ---------------------------------------------------------------------------

@dataclass
class CategoryMetrics:
    category: ExceptionCategory
    precision: float
    recall: float
    f1: float
    support: int
    true_positives: int
    false_positives: int
    false_negatives: int


@dataclass
class EvaluationResult:
    """
    Immutable evaluation artifact — saved after holdout run.
    Never recomputed during dashboard load.
    """
    run_id: str
    dataset_split: str               # "tune" | "validation" | "holdout"
    model_name: str
    prompt_version: str
    threshold: float
    random_seed: int
    timestamp: datetime

    total_records: int
    clean_records: int
    exception_records: int
    auto_resolved: int
    human_review: int
    correctly_classified: int
    false_auto_resolved: int

    # Derived metrics
    automation_rate: float
    escalation_rate: float
    false_auto_resolve_rate: float
    overall_accuracy: float

    per_category: list[CategoryMetrics]
    confusion_matrix: list[list[int]]
    confusion_labels: list[str]

    # Performance
    throughput_records_per_second: float
    avg_latency_ms: float

    notes: str = ""
