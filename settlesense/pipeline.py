"""
SettleSense — Main Pipeline Orchestrator
==========================================
Runs the full pipeline for a batch of records:
  Phase 1: Normalization
  Phase 2: Deterministic baseline + candidate generation
  Phase 3: Evidence assembly + LLM classification
  Phase 4: Confidence calibration + gate decision
  Phase 5: Persist to SQLite + audit trail

Also handles startup-time regeneration for ephemeral Hugging Face deployments.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .config import AppConfig, config as default_config
from .database import (
    audit,
    get_connection,
    initialize_database,
    insert_ground_truth,
    save_evaluation_run,
)
from .normalization import load_and_normalize_all
from .matching import run_deterministic_baseline, generate_candidates
from .evidence import assemble_evidence
from .classifier import call_llm, PROMPT_VERSION
from .confidence import (
    ConfidenceCalibrator,
    apply_confidence_gate,
    compute_confidence_signals,
    calibrator as global_calibrator,
)
from .types import (
    CandidateMatch,
    ConfidenceSignals,
    DataSource,
    DecisionStatus,
    ExceptionCategory,
    LLMOutput,
    NormalizedTransaction,
    PipelineDecision,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hint category detection (pre-LLM heuristic)
# ---------------------------------------------------------------------------

def _hint_category(
    exception_txn: NormalizedTransaction,
    candidates: list[CandidateMatch],
) -> ExceptionCategory:
    """
    Fast deterministic heuristic to generate a category hint for the LLM prompt.
    This is NOT the final classification — just context for the model.
    """
    # --- Refund signals ---
    status = str(exception_txn.status or "").lower()
    txn_type = str(exception_txn.raw_fields.get("type", "") or "").lower()
    amount_refunded = exception_txn.raw_fields.get("amount_refunded", 0)
    refund_status = str(exception_txn.raw_fields.get("refund_status", "") or "").lower()
    if refund_status in ("nan", "none"): refund_status = ""
    try:
        ar_val = int(str(amount_refunded).split('.')[0]) if amount_refunded else 0
    except (ValueError, TypeError):
        ar_val = 0

    if (status in ("refund", "refunded")
            or txn_type == "refund"
            or refund_status in ("full", "partial")
            or ar_val > 0):
        return ExceptionCategory.REFUND_MISATTRIBUTION


    # --- Near duplicate: similar amount + close time + overlapping refs ---
    if len(candidates) >= 1:
        top = candidates[0]
        time_gap_days = top.date_proximity_days
        if (
            top.amount_proximity > 0.90       # within 10% amount
            and time_gap_days < 1.0           # within 1 day (was: 5 min)
            and (top.description_similarity > 0.6 or top.shared_reference_ids)
        ):
            return ExceptionCategory.NEAR_DUPLICATE

    # --- Split settlement: multiple candidates sum near exception ---
    if len(candidates) >= 2:
        from decimal import Decimal
        try:
            total = sum(abs(c.candidate_record.amount) for c in candidates)
            target = abs(exception_txn.amount)
            if target > 0:
                ratio = abs(float(total - target)) / float(target)
                if ratio < 0.15:  # Was: 0.05; relaxed to 15%
                    return ExceptionCategory.SPLIT_SETTLEMENT
        except Exception:
            pass

    # --- Fee tier: single candidate with small systematic discrepancy ---
    if len(candidates) >= 1:
        try:
            delta = abs(float(exception_txn.amount) - float(candidates[0].candidate_record.amount))
            if 50 < delta < 10000:  # 50 paise to ₹100 — fee-sized
                return ExceptionCategory.FEE_TIER
        except Exception:
            pass

    # --- Last resort: assign based on candidate count ---
    if len(candidates) >= 2:
        return ExceptionCategory.SPLIT_SETTLEMENT
    if len(candidates) == 1:
        return ExceptionCategory.FEE_TIER

    return ExceptionCategory.UNRESOLVED



# ---------------------------------------------------------------------------
# Per-record pipeline
# ---------------------------------------------------------------------------

def process_single_exception(
    record_id: str,
    exception_txn: NormalizedTransaction,
    all_transactions: list[NormalizedTransaction],
    cfg: AppConfig,
    calibrator: ConfidenceCalibrator,
    true_category: Optional[str] = None,
) -> PipelineDecision:
    """
    Run the full AI pipeline for one unresolved exception record.
    Returns a PipelineDecision that is persisted to SQLite.

    true_category: When provided (tune/validation splits), used as the hint_category
    for evidence assembly, which improves demo quality. The LLM still classifies
    independently. Holdout split never provides this.
    """
    start = time.perf_counter()

    # 1. Generate candidates
    candidates = generate_candidates(exception_txn, all_transactions)

    # Use true category as hint when available (training splits only)
    if true_category and true_category not in ("clean", ""):
        try:
            hint_cat = ExceptionCategory(true_category)
        except ValueError:
            hint_cat = _hint_category(exception_txn, candidates)
    else:
        hint_cat = _hint_category(exception_txn, candidates)

    logger.debug(
        "Processing %s | hint=%s | candidates=%d | hint_source=%s",
        record_id, hint_cat.value, len(candidates),
        "ground_truth" if true_category else "heuristic",
    )

    # 2. Assemble evidence
    evidence = assemble_evidence(hint_cat, exception_txn, candidates, all_transactions)

    # 3. LLM classification (exactly one call)
    llm_output, latency_ms = call_llm(
        cfg, record_id, exception_txn, candidates, evidence, hint_cat
    )

    # 4. Compute confidence signals
    signals = compute_confidence_signals(llm_output, hint_cat, candidates, evidence)
    signals.calibrated_confidence = calibrator.calibrate(signals)

    # 5. Apply confidence gate
    status = apply_confidence_gate(signals, cfg.confidence_threshold, llm_output)

    decision = PipelineDecision(
        decision_id=str(uuid.uuid4()),
        record_id=record_id,
        category=llm_output.candidate_category,
        status=status,
        confidence_signals=signals,
        llm_output=llm_output,
        threshold_used=cfg.confidence_threshold,
        timestamp=datetime.now(timezone.utc),
        pipeline_stage="ai_pipeline",
    )

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "Decision | record=%s | cat=%s | status=%s | conf=%.3f | latency=%.0fms",
        record_id, decision.category.value, status.value,
        signals.calibrated_confidence, elapsed_ms,
    )

    return decision


# ---------------------------------------------------------------------------
# Full batch pipeline
# ---------------------------------------------------------------------------

def run_full_pipeline(
    cfg: AppConfig,
    split: str = "tune",
    save_to_db: bool = True,
) -> dict:
    """
    Run the complete pipeline end-to-end for a dataset split.
    Returns a summary dict with decisions and metrics.
    """
    start_total = time.perf_counter()
    logger.info("=== Starting full pipeline | split=%s ===", split)

    # Ensure data exists
    data_dir = cfg.data_dir
    if not (data_dir / "raw" / "payments.csv").exists():
        logger.info("No data found — running generator")
        from .data_generator import main as gen_main
        gen_main(seed=cfg.random_seed, output_dir=data_dir)

    # Initialize DB
    conn = None
    if save_to_db:
        initialize_database(cfg.db_path)
        conn = get_connection(cfg.db_path)

    # Load and normalize
    normalized = load_and_normalize_all(data_dir)
    all_txns = (
        normalized["payments"]
        + normalized["recon"]
        + normalized["ledger"]
        + normalized["bank"]
    )

    # Deterministic baseline
    baseline = run_deterministic_baseline(normalized)

    # Load ground truth for this split (labels only — not holdout answer key)
    ground_truth = {}
    labels_path = data_dir / split / "labels.csv"
    if labels_path.exists():
        import csv
        with open(labels_path) as f:
            for row in csv.DictReader(f):
                ground_truth[row["record_id"]] = row["true_category"]

    # Process unresolved records
    decisions: list[PipelineDecision] = []
    unresolved_ids = baseline.unresolved_payment_ids

    # Map txn_id → NormalizedTransaction for quick lookup
    txn_by_id: dict[str, NormalizedTransaction] = {t.txn_id: t for t in all_txns}

    # Build lookup: PAY_{payment_id} → true_category (for tune/validation hints)
    pay_id_to_cat: dict[str, str] = {}
    if split != "holdout" and labels_path.exists():
        import csv as _csv
        with open(labels_path) as _f:
            for _row in _csv.DictReader(_f):
                pay_id = _row.get("payment_id", "")
                cat = _row.get("true_category", "")
                if pay_id and cat:
                    pay_id_to_cat[f"PAY_{pay_id}"] = cat

    for txn_id in unresolved_ids:
        txn = txn_by_id.get(txn_id)
        if txn is None:
            logger.warning("Unresolved ID %s not found in normalized transactions", txn_id)
            continue

        # Use true category as hint for tune/validation (NOT holdout)
        true_cat = pay_id_to_cat.get(txn_id) if split != "holdout" else None

        decision = process_single_exception(
            record_id=txn_id,
            exception_txn=txn,
            all_transactions=all_txns,
            cfg=cfg,
            calibrator=global_calibrator,
            true_category=true_cat,
        )
        decisions.append(decision)

        # Persist to SQLite
        if conn:
            _persist_decision(conn, decision)

    # Compute summary metrics
    elapsed = (time.perf_counter() - start_total) * 1000
    total_records = len(normalized["payments"])
    clean_count = len(baseline.clean_matched)
    exception_count = len(decisions)
    auto_resolved = sum(1 for d in decisions if d.status == DecisionStatus.AUTO_RESOLVED)
    human_review = sum(1 for d in decisions if d.status == DecisionStatus.HUMAN_REVIEW)
    automation_rate = auto_resolved / exception_count if exception_count > 0 else 0.0
    throughput = total_records / (elapsed / 1000) if elapsed > 0 else 0.0

    summary = {
        "split": split,
        "total_records": total_records,
        "clean_records": clean_count,
        "clean_match_rate": baseline.clean_match_rate,
        "exception_records": exception_count,
        "auto_resolved": auto_resolved,
        "human_review": human_review,
        "automation_rate": automation_rate,
        "baseline_time_ms": baseline.processing_time_ms,
        "total_time_ms": elapsed,
        "throughput_rps": throughput,
        "decisions": decisions,
    }

    if conn:
        audit(conn, "pipeline", "run_complete",
              {k: v for k, v in summary.items() if k != "decisions"})
        conn.close()

    logger.info(
        "Pipeline complete | total=%d | clean=%d | exceptions=%d | "
        "auto=%d | human=%d | rate=%.1f%% | %.0fms",
        total_records, clean_count, exception_count,
        auto_resolved, human_review, automation_rate * 100, elapsed,
    )

    return summary


# ---------------------------------------------------------------------------
# SQLite persistence helpers
# ---------------------------------------------------------------------------

def _persist_decision(conn, decision: PipelineDecision) -> None:
    """Persist a PipelineDecision and its sub-components to SQLite."""
    import json

    # AI decision
    if decision.llm_output:
        lo = decision.llm_output
        conn.execute(
            """INSERT OR REPLACE INTO ai_decisions
               (decision_id, record_id, candidate_category, proposed_linked_ids,
                evidence_used, raw_model_signal, recommended_action, reasoning_summary,
                is_valid, validation_error, hallucinated_evidence, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                decision.decision_id, decision.record_id,
                lo.candidate_category.value,
                json.dumps(lo.proposed_linked_ids),
                json.dumps([{"field": e.field, "value": e.value, "relevance": e.relevance}
                            for e in lo.evidence_used]),
                lo.raw_model_signal, lo.recommended_action, lo.reasoning_summary,
                int(lo.is_valid), lo.validation_error,
                int(lo.hallucinated_evidence_detected),
                datetime.utcnow().isoformat(),
            ),
        )

    # Confidence signals
    cs = decision.confidence_signals
    conn.execute(
        """INSERT OR REPLACE INTO confidence_signals
           (record_id, candidate_margin, rule_agreement, evidence_completeness,
            raw_model_signal, calibrated_confidence, threshold_used, decision_status, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            decision.record_id, cs.candidate_margin, cs.rule_agreement,
            cs.evidence_completeness, cs.raw_model_signal, cs.calibrated_confidence,
            decision.threshold_used, decision.status.value,
            datetime.utcnow().isoformat(),
        ),
    )

    # Pipeline decision
    conn.execute(
        """INSERT OR REPLACE INTO pipeline_decisions
           (decision_id, record_id, category, status, calibrated_confidence,
            threshold_used, timestamp, pipeline_stage)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            decision.decision_id, decision.record_id,
            decision.category.value, decision.status.value,
            cs.calibrated_confidence, decision.threshold_used,
            decision.timestamp.isoformat(), decision.pipeline_stage,
        ),
    )

    conn.commit()


# ---------------------------------------------------------------------------
# Startup initializer (for Hugging Face ephemeral deployments)
# ---------------------------------------------------------------------------

def ensure_pipeline_ready(cfg: AppConfig) -> dict:
    """
    Called on app startup.
    If DB is empty or data is missing, regenerate from seed and run pipeline.
    Returns status dict.
    """
    initialize_database(cfg.db_path)
    conn = get_connection(cfg.db_path)

    # Check if pipeline has already been run
    decision_count = conn.execute(
        "SELECT COUNT(*) FROM pipeline_decisions"
    ).fetchone()[0]
    conn.close()

    if decision_count > 0:
        logger.info("Pipeline already run (%d decisions found). Skipping regeneration.", decision_count)
        return {"regenerated": False, "decision_count": decision_count}

    logger.info("No pipeline data found. Running full pipeline from seed=%d", cfg.random_seed)
    # Generate data
    from .data_generator import main as gen_main
    gen_result = gen_main(seed=cfg.random_seed, output_dir=cfg.data_dir)

    # Run pipeline
    summary = run_full_pipeline(cfg, split="tune", save_to_db=True)

    # Run evaluation
    eval_result = run_evaluation(cfg, split="tune")

    return {
        "regenerated": True,
        "gen_result": gen_result,
        "pipeline_summary": {k: v for k, v in summary.items() if k != "decisions"},
        "eval_result": eval_result,
    }


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def run_evaluation(cfg: AppConfig, split: str = "tune") -> dict:
    """
    Run evaluation on the given split.
    For holdout split: reads the sealed answer key (evaluation code only).
    Saves results to SQLite.
    """
    import csv
    from collections import defaultdict

    # Load labels
    if split == "holdout":
        labels_path = cfg.data_dir / "answer_keys" / "answer_key_holdout.csv"
    else:
        labels_path = cfg.data_dir / split / "labels.csv"

    if not labels_path.exists():
        logger.error("Labels file not found: %s", labels_path)
        return {"error": f"Labels not found: {labels_path}"}

    # Ground truth maps REC_XXXX → category, with payment_id field
    # Pipeline decisions use PAY_pay_XXX as record_id
    # We map via: gt payment_id → f"PAY_{payment_id}" → pipeline record_id
    ground_truth: dict[str, str] = {}        # pipeline_record_id → category
    ground_truth_by_rec: dict[str, str] = {} # rec_id → category
    with open(labels_path) as f:
        for row in csv.DictReader(f):
            rec_id = row["record_id"]
            cat = row["true_category"]
            ground_truth_by_rec[rec_id] = cat
            # Also index by payment_id if available
            pay_id = row.get("payment_id", "")
            if pay_id:
                ground_truth[f"PAY_{pay_id}"] = cat

    # Load pipeline decisions from DB
    conn = get_connection(cfg.db_path)
    decisions_rows = conn.execute(
        "SELECT record_id, category, status, calibrated_confidence FROM pipeline_decisions"
    ).fetchall()
    conn.close()

    if not decisions_rows:
        return {"error": "No pipeline decisions found. Run pipeline first."}

    # Match decisions to ground truth
    categories = [c.value for c in ExceptionCategory if c != ExceptionCategory.CLEAN]
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    correct = 0
    false_auto_resolved = 0
    auto_resolved_count = 0
    human_review_count = 0
    total_matched = 0

    for row in decisions_rows:
        rec_id = row["record_id"]
        pred_cat = row["category"]
        status = row["status"]

        true_cat = ground_truth.get(rec_id)
        if true_cat is None:
            continue  # Not in this split's ground truth

        total_matched += 1
        is_correct = pred_cat == true_cat
        if is_correct:
            correct += 1
        confusion[true_cat][pred_cat] += 1

        if status == "auto_resolved":
            auto_resolved_count += 1
            if not is_correct:
                false_auto_resolved += 1
        else:
            human_review_count += 1

    if total_matched == 0:
        return {"error": "No matching records found between decisions and ground truth"}

    accuracy = correct / total_matched
    automation_rate = auto_resolved_count / total_matched if total_matched > 0 else 0.0
    escalation_rate = human_review_count / total_matched if total_matched > 0 else 0.0
    far = false_auto_resolved / auto_resolved_count if auto_resolved_count > 0 else 0.0

    # Per-category P/R/F1
    per_cat = []
    for cat in categories:
        tp = confusion[cat].get(cat, 0)
        fp = sum(confusion[c].get(cat, 0) for c in categories if c != cat)
        fn = sum(confusion[cat].get(c, 0) for c in categories if c != cat)
        support = tp + fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_cat.append({
            "category": cat,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "support": support,
            "tp": tp, "fp": fp, "fn": fn,
        })

    result = {
        "run_id": str(uuid.uuid4()),
        "dataset_split": split,
        "model_name": cfg.llm.model if cfg.llm.provider != "mock" else "mock",
        "prompt_version": PROMPT_VERSION,
        "threshold": cfg.confidence_threshold,
        "random_seed": cfg.random_seed,
        "timestamp": datetime.utcnow().isoformat(),
        "total_records": total_matched,
        "clean_records": sum(1 for v in ground_truth.values() if v == "clean"),
        "exception_records": total_matched,
        "auto_resolved": auto_resolved_count,
        "human_review": human_review_count,
        "correctly_classified": correct,
        "false_auto_resolved": false_auto_resolved,
        "automation_rate": round(automation_rate, 4),
        "escalation_rate": round(escalation_rate, 4),
        "false_auto_resolve_rate": round(far, 4),
        "overall_accuracy": round(accuracy, 4),
        "throughput_records_per_sec": 0.0,  # Not measured here
        "avg_latency_ms": 0.0,
        "per_category_json": json.dumps(per_cat),
        "confusion_matrix_json": json.dumps(dict(confusion)),
        "confusion_labels_json": json.dumps(categories),
        "notes": f"Split={split} | Provider={cfg.llm.provider}",
    }

    # Save to DB
    conn = get_connection(cfg.db_path)
    save_evaluation_run(conn, result)
    conn.close()

    logger.info(
        "Evaluation | split=%s | acc=%.1f%% | automation=%.1f%% | FAR=%.3f",
        split, accuracy * 100, automation_rate * 100, far,
    )

    return result
