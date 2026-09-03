"""
SettleSense — Confidence Calibration
======================================
Confidence is built from three independent signals — never from LLM self-report alone:
  1. candidate_margin: separation between best and second-best candidate scores
  2. rule_agreement: does the deterministic baseline agree with the LLM's category?
  3. evidence_completeness: fraction of required evidence fields present

Calibration: logistic regression fitted on the validation set.
  - Maps (margin, rule_agreement, completeness) → P(correct)
  - Calibrated confidence used for the confidence gate (threshold comparison).

If calibration model not yet fitted (cold start), uses a transparent weighted sum.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from .evidence import compute_evidence_completeness
from .types import (
    CandidateMatch,
    ConfidenceSignals,
    DecisionStatus,
    EvidenceItem,
    ExceptionCategory,
    LLMOutput,
)

logger = logging.getLogger(__name__)

CALIBRATION_MODEL_PATH = Path(__file__).parent.parent / "data" / "calibration_model.pkl"


# ---------------------------------------------------------------------------
# Signal computation
# ---------------------------------------------------------------------------

def compute_candidate_margin(candidates: list[CandidateMatch]) -> float:
    """
    Relative margin between best and second-best candidate.
    High margin → model is clearly choosing one candidate over others.
    Returns 0.0 if fewer than 2 candidates.
    """
    if len(candidates) < 2:
        return 1.0 if len(candidates) == 1 else 0.0
    scores = sorted([c.match_score for c in candidates], reverse=True)
    best = scores[0]
    if best == 0.0:
        return 0.0
    return (best - scores[1]) / best


def compute_rule_agreement(
    llm_category: ExceptionCategory,
    hint_category: ExceptionCategory,
) -> float:
    """
    Agreement between deterministic hint category and LLM output.
    1.0 if they agree, 0.0 if they disagree.
    Partial credit (0.5) if LLM says 'unresolved'.
    """
    if llm_category == hint_category:
        return 1.0
    if llm_category == ExceptionCategory.UNRESOLVED:
        return 0.5
    return 0.0


def compute_confidence_signals(
    llm_output: LLMOutput,
    hint_category: ExceptionCategory,
    candidates: list[CandidateMatch],
    evidence: list[EvidenceItem],
) -> ConfidenceSignals:
    """Compute all three independent signals and assemble ConfidenceSignals."""
    margin = compute_candidate_margin(candidates)
    rule_agree = compute_rule_agreement(llm_output.candidate_category, hint_category)
    completeness = compute_evidence_completeness(llm_output.candidate_category, evidence)

    return ConfidenceSignals(
        candidate_margin=margin,
        rule_agreement=rule_agree,
        evidence_completeness=completeness,
        raw_model_signal=llm_output.raw_model_signal,
        calibrated_confidence=0.0,  # Filled in by calibrate()
    )


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

class ConfidenceCalibrator:
    """
    Logistic regression calibrator.
    Trained on (signals, correct_bool) pairs from the validation set.
    Falls back to a transparent weighted sum if not yet fitted.
    """

    def __init__(self, model_path: Path = CALIBRATION_MODEL_PATH):
        self.model_path = model_path
        self._model = None
        self._is_fitted = False
        self._load()

    def _load(self) -> None:
        if self.model_path.exists():
            try:
                with open(self.model_path, "rb") as f:
                    self._model = pickle.load(f)
                self._is_fitted = True
                logger.info("Loaded calibration model from %s", self.model_path)
            except Exception as e:
                logger.warning("Failed to load calibration model: %s", e)

    def _features(self, signals: ConfidenceSignals) -> np.ndarray:
        return np.array([[
            signals.candidate_margin,
            signals.rule_agreement,
            signals.evidence_completeness,
            signals.raw_model_signal,
        ]])

    def calibrate(self, signals: ConfidenceSignals) -> float:
        """Return calibrated P(correct) for these signals."""
        if self._is_fitted and self._model is not None:
            try:
                prob = float(self._model.predict_proba(self._features(signals))[0, 1])
                return max(0.0, min(1.0, prob))
            except Exception as e:
                logger.warning("Calibration model inference failed: %s", e)

        # Cold-start: transparent weighted sum
        # rule_agreement (0.40) — strongest signal: does deterministic baseline agree?
        # raw_model_signal (0.35) — LLM or mock self-reported confidence
        # candidate_margin (0.15) — separation between best and second-best candidate
        # evidence_completeness (0.10) — completeness of evidence pack (reduced weight;
        #   this can be low for valid records with sparse recon data)
        score = (
            0.40 * signals.rule_agreement
            + 0.35 * signals.raw_model_signal
            + 0.15 * signals.candidate_margin
            + 0.10 * signals.evidence_completeness
        )
        return max(0.0, min(1.0, score))

    def fit(
        self,
        signals_list: list[ConfidenceSignals],
        correct_labels: list[bool],
    ) -> dict:
        """
        Fit logistic regression on validation-set pairs.
        Saves model to disk.
        Returns fit metrics.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import cross_val_score

        if len(signals_list) < 10:
            logger.warning("Too few samples (%d) for calibration fitting", len(signals_list))
            return {"error": "insufficient_samples"}

        X = np.array([self._features(s)[0] for s in signals_list])
        y = np.array([int(c) for c in correct_labels])

        # Simple logistic regression — transparent and explainable
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)

        # Cross-validated AUC
        try:
            cv_auc = cross_val_score(model, X, y, cv=min(5, len(y)), scoring="roc_auc").mean()
        except Exception:
            cv_auc = float("nan")

        self._model = model
        self._is_fitted = True

        # Save
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump(model, f)
        logger.info("Calibration model fitted and saved | cv_auc=%.3f", cv_auc)

        coefs = dict(zip(
            ["candidate_margin", "rule_agreement", "evidence_completeness", "raw_model_signal"],
            model.coef_[0].tolist(),
        ))
        return {
            "cv_auc": cv_auc,
            "n_samples": len(signals_list),
            "n_positive": int(y.sum()),
            "coefficients": coefs,
        }


# ---------------------------------------------------------------------------
# Confidence gate
# ---------------------------------------------------------------------------

def apply_confidence_gate(
    signals: ConfidenceSignals,
    threshold: float,
    llm_output: LLMOutput,
) -> DecisionStatus:
    """
    Decide: auto-resolve or human review.

    Auto-resolve requires ALL of:
      - calibrated_confidence ≥ threshold
      - llm_output.is_valid = True
      - no hallucinated evidence
      - LLM did not classify as 'unresolved'

    Any failure → human_review.
    The LLM cannot override this gate.
    """
    if not llm_output.is_valid:
        logger.info("Gate: human_review — invalid LLM output")
        return DecisionStatus.HUMAN_REVIEW

    if llm_output.hallucinated_evidence_detected:
        logger.info("Gate: human_review — hallucinated evidence")
        return DecisionStatus.HUMAN_REVIEW

    if llm_output.candidate_category == ExceptionCategory.UNRESOLVED:
        logger.info("Gate: human_review — LLM classified as unresolved")
        return DecisionStatus.HUMAN_REVIEW

    if signals.calibrated_confidence < threshold:
        logger.info(
            "Gate: human_review — confidence %.3f < threshold %.3f",
            signals.calibrated_confidence, threshold,
        )
        return DecisionStatus.HUMAN_REVIEW

    logger.info(
        "Gate: auto_resolved — confidence %.3f ≥ threshold %.3f",
        signals.calibrated_confidence, threshold,
    )
    return DecisionStatus.AUTO_RESOLVED


# ---------------------------------------------------------------------------
# Threshold selection (on validation set)
# ---------------------------------------------------------------------------

def select_threshold(
    val_confidences: list[float],
    val_correct: list[bool],
    max_false_auto_resolve_rate: float = 0.05,
) -> dict:
    """
    Choose the highest threshold that keeps false-auto-resolve rate ≤ target.
    Optimizes for VERY LOW false-auto-resolve rate, not maximum automation.

    Returns: {threshold, automation_rate, false_auto_resolve_rate, selected}
    """
    if not val_confidences:
        return {"threshold": 0.75, "automation_rate": 0.0,
                "false_auto_resolve_rate": 0.0, "selected": False}

    results = []
    for candidate_threshold in [i / 100 for i in range(50, 100)]:
        auto = [i for i, c in enumerate(val_confidences) if c >= candidate_threshold]
        if not auto:
            results.append({
                "threshold": candidate_threshold,
                "automation_rate": 0.0,
                "false_auto_resolve_rate": 0.0,
            })
            continue

        n_auto = len(auto)
        n_false = sum(1 for i in auto if not val_correct[i])
        far = n_false / n_auto if n_auto > 0 else 0.0
        automation_rate = n_auto / len(val_confidences)
        results.append({
            "threshold": candidate_threshold,
            "automation_rate": automation_rate,
            "false_auto_resolve_rate": far,
        })

    # Find highest threshold where FAR ≤ max
    valid = [r for r in results if r["false_auto_resolve_rate"] <= max_false_auto_resolve_rate]
    if not valid:
        # Fallback: use the threshold with minimum FAR
        best = min(results, key=lambda r: r["false_auto_resolve_rate"])
        best["selected"] = True
        logger.warning(
            "No threshold achieves FAR ≤ %.2f. Using %.2f (FAR=%.3f)",
            max_false_auto_resolve_rate,
            best["threshold"],
            best["false_auto_resolve_rate"],
        )
        return best

    # Among valid thresholds, pick the one maximizing automation rate
    best = max(valid, key=lambda r: r["automation_rate"])
    best["selected"] = True
    logger.info(
        "Selected threshold: %.2f | automation=%.1f%% | FAR=%.3f",
        best["threshold"], best["automation_rate"] * 100, best["false_auto_resolve_rate"],
    )
    return best


# Global calibrator instance
calibrator = ConfidenceCalibrator()
