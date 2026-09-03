"""
SettleSense — Deterministic Baseline
=====================================
Runs BEFORE the LLM. Handles cases the LLM should never see:
  - Exact amount + identifier matches
  - Valid date-window matches
  - Known deterministic relationships

Reports: clean match rate, unresolved remainder, processing time.
LLM only receives what the deterministic layer cannot resolve.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from .types import (
    CandidateMatch,
    DataSource,
    ExceptionCategory,
    NormalizedTransaction,
)

logger = logging.getLogger(__name__)

# Tolerances
AMOUNT_TOLERANCE_PAISE = Decimal("5")   # ₹0.05 rounding tolerance
DATE_WINDOW_DAYS = 7                     # Settlement should arrive within 7 days


@dataclass
class BaselineResult:
    """Result of the deterministic baseline pass."""
    clean_matched: list[tuple[str, str]]              # (payment_txn_id, ledger_txn_id)
    unresolved_payment_ids: list[str]                 # IDs that need AI analysis
    unresolved_ledger_ids: list[str]
    clean_match_rate: float
    processing_time_ms: float
    match_details: dict[str, dict]                    # txn_id → match detail dict


def _amounts_match(a: Decimal, b: Decimal) -> bool:
    """Check if two paise amounts match within tolerance."""
    return abs(a - b) <= AMOUNT_TOLERANCE_PAISE


def _ids_overlap(ids_a: list[str], ids_b: list[str]) -> bool:
    """Check if two reference ID lists share any common non-empty element."""
    set_a = {x for x in ids_a if x}
    set_b = {x for x in ids_b if x}
    return bool(set_a & set_b)


def _date_within_window(
    payment_date,
    ledger_date,
    window_days: int = DATE_WINDOW_DAYS,
) -> bool:
    """Check if ledger date is within settlement window of payment date."""
    if payment_date is None or ledger_date is None:
        return True  # Give benefit of doubt on missing dates
    delta = abs((ledger_date - payment_date).total_seconds()) / 86400
    return delta <= window_days


def run_deterministic_baseline(
    normalized: dict[str, list[NormalizedTransaction]],
) -> BaselineResult:
    """
    Phase 1: Exact and rule-based matching.

    Matching strategy:
    1. Match payment ↔ ledger via shared reference IDs (order_id, payment_id)
    2. Validate: amounts match + date within window
    3. Unmatched payments → candidates for AI analysis

    Returns BaselineResult with clean matches and unresolved remainder.
    """
    start = time.perf_counter()

    payments = normalized.get("payments", [])
    ledger = normalized.get("ledger", [])
    recon = normalized.get("recon", [])
    bank = normalized.get("bank", [])

    logger.info(
        "Deterministic baseline | payments=%d | ledger=%d | recon=%d | bank=%d",
        len(payments), len(ledger), len(recon), len(bank),
    )

    # Build lookup indexes
    # Recon indexed by payment_id and settlement_id
    recon_by_payment: dict[str, list[NormalizedTransaction]] = defaultdict(list)
    recon_by_settlement: dict[str, list[NormalizedTransaction]] = defaultdict(list)
    for r in recon:
        if r.payment_id:
            recon_by_payment[r.payment_id].append(r)
        if r.settlement_id:
            recon_by_settlement[r.settlement_id].append(r)

    # Ledger indexed by invoice_ref
    ledger_by_invoice: dict[str, NormalizedTransaction] = {}
    for l in ledger:
        for ref in l.reference_ids:
            if ref != l.source_record_id:  # Skip own ID, use invoice_ref
                ledger_by_invoice[ref] = l

    # Bank indexed by UTR
    bank_by_utr: dict[str, NormalizedTransaction] = {}
    for b in bank:
        for ref in b.reference_ids:
            if ref.startswith("UTR"):
                bank_by_utr[ref] = b

    matched_payment_ids: set[str] = set()
    matched_ledger_ids: set[str] = set()
    clean_matched: list[tuple[str, str]] = []
    match_details: dict[str, dict] = {}

    for pay in payments:
        payment_id = pay.payment_id or pay.source_record_id
        if not payment_id:
            continue

        # Strategy 1: Match via recon → UTR → bank → ledger chain
        recon_records = recon_by_payment.get(payment_id, [])
        if not recon_records and pay.settlement_id:
            recon_records = recon_by_settlement.get(pay.settlement_id, [])

        matched = False
        for recon_rec in recon_records:
            # Find matching bank record via UTR
            utr = None
            for ref in recon_rec.reference_ids:
                if ref.startswith("UTR"):
                    utr = ref
                    break

            bank_rec = bank_by_utr.get(utr) if utr else None

            # Find matching ledger via order_receipt or invoice_ref
            ledger_rec = None
            for ref in recon_rec.reference_ids:
                if ref in ledger_by_invoice:
                    ledger_rec = ledger_by_invoice[ref]
                    break

            if ledger_rec is None:
                # Fallback: search by amount + date proximity
                for l in ledger:
                    if (
                        l.source_record_id not in matched_ledger_ids
                        and _amounts_match(abs(pay.amount), abs(l.amount))
                        and _date_within_window(pay.transaction_date, l.transaction_date)
                    ):
                        ledger_rec = l
                        break

            if ledger_rec is None:
                continue

            # Validate match quality
            amount_ok = _amounts_match(abs(pay.amount), abs(ledger_rec.amount))
            date_ok = _date_within_window(pay.transaction_date, ledger_rec.transaction_date)

            if amount_ok and date_ok and ledger_rec.source_record_id not in matched_ledger_ids:
                matched_payment_ids.add(pay.txn_id)
                matched_ledger_ids.add(ledger_rec.txn_id)
                clean_matched.append((pay.txn_id, ledger_rec.txn_id))
                match_details[pay.txn_id] = {
                    "ledger_id": ledger_rec.txn_id,
                    "bank_id": bank_rec.txn_id if bank_rec else None,
                    "amount_match": True,
                    "date_match": True,
                    "match_method": "recon_utr_chain",
                }
                matched = True
                break

        # Strategy 2: Direct amount + date match if recon chain didn't work
        if not matched:
            for l in ledger:
                if l.txn_id in matched_ledger_ids:
                    continue
                if (
                    _amounts_match(abs(pay.amount), abs(l.amount))
                    and _date_within_window(pay.transaction_date, l.transaction_date)
                    and _ids_overlap(pay.reference_ids, l.reference_ids)
                ):
                    matched_payment_ids.add(pay.txn_id)
                    matched_ledger_ids.add(l.txn_id)
                    clean_matched.append((pay.txn_id, l.txn_id))
                    match_details[pay.txn_id] = {
                        "ledger_id": l.txn_id,
                        "bank_id": None,
                        "amount_match": True,
                        "date_match": True,
                        "match_method": "direct_amount_date_id",
                    }
                    matched = True
                    break

    # Unresolved = payments not cleanly matched
    unresolved_payment_ids = [
        p.txn_id for p in payments if p.txn_id not in matched_payment_ids
    ]
    unresolved_ledger_ids = [
        l.txn_id for l in ledger if l.txn_id not in matched_ledger_ids
    ]

    total = len(payments)
    clean_count = len(clean_matched)
    clean_match_rate = clean_count / total if total > 0 else 0.0

    elapsed_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "Baseline complete | clean=%d | unresolved_payments=%d | unresolved_ledger=%d | "
        "match_rate=%.1f%% | time=%.1fms",
        clean_count,
        len(unresolved_payment_ids),
        len(unresolved_ledger_ids),
        clean_match_rate * 100,
        elapsed_ms,
    )

    return BaselineResult(
        clean_matched=clean_matched,
        unresolved_payment_ids=unresolved_payment_ids,
        unresolved_ledger_ids=unresolved_ledger_ids,
        clean_match_rate=clean_match_rate,
        processing_time_ms=elapsed_ms,
        match_details=match_details,
    )


# ---------------------------------------------------------------------------
# Candidate Generation
# ---------------------------------------------------------------------------

def generate_candidates(
    exception_txn: NormalizedTransaction,
    all_transactions: list[NormalizedTransaction],
    max_candidates: int = 5,
) -> list[CandidateMatch]:
    """
    For one unresolved transaction, generate a small candidate set.
    Candidates are ranked by composite similarity score.
    Does NOT call the LLM — pure deterministic scoring.
    """
    from rapidfuzz import fuzz

    # Don't match against self
    others = [t for t in all_transactions if t.txn_id != exception_txn.txn_id]

    candidates: list[CandidateMatch] = []

    for candidate in others:
        # 1. Amount proximity (normalized to 0-1)
        amount_diff = abs(float(exception_txn.amount) - float(candidate.amount))
        max_amount = max(float(exception_txn.amount), float(candidate.amount), 1.0)
        amount_proximity = max(0.0, 1.0 - (amount_diff / max_amount))

        # Only consider candidates with some amount proximity
        if amount_proximity < 0.5:
            continue

        # 2. Date proximity
        if exception_txn.transaction_date and candidate.transaction_date:
            date_diff_days = abs(
                (exception_txn.transaction_date - candidate.transaction_date).total_seconds()
            ) / 86400
        else:
            date_diff_days = 999.0
        date_proximity = max(0.0, 1.0 - (date_diff_days / 30.0))

        # 3. Description similarity (rapidfuzz)
        desc_sim = fuzz.token_sort_ratio(
            exception_txn.description, candidate.description
        ) / 100.0

        # 4. Shared reference IDs
        shared_ids = list(
            set(exception_txn.reference_ids) & set(candidate.reference_ids)
        )
        shared_id_bonus = min(0.3, len(shared_ids) * 0.15)

        # Composite score
        match_score = (
            0.40 * amount_proximity
            + 0.20 * date_proximity
            + 0.20 * desc_sim
            + shared_id_bonus
        )

        if match_score >= 0.3:  # Minimum threshold to be a candidate
            candidates.append(CandidateMatch(
                candidate_id=candidate.txn_id,
                match_score=match_score,
                amount_proximity=amount_proximity,
                date_proximity_days=date_diff_days,
                description_similarity=desc_sim,
                shared_reference_ids=shared_ids,
                candidate_record=candidate,
            ))

    # Sort by score descending, return top N
    candidates.sort(key=lambda c: c.match_score, reverse=True)
    return candidates[:max_candidates]
