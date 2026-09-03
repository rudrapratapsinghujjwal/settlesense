"""
SettleSense — Normalization Layer
==================================
Converts all four raw data sources into NormalizedTransaction objects.
Preserves full traceability to source records.
Handles: inconsistent naming, missing values, type coercion,
         datetime normalization, amount normalization (paise).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .types import (
    Currency,
    DataSource,
    NormalizedTransaction,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Map of known column name variants → canonical name
_PAYMENT_COL_MAP = {
    "id": "payment_id",
    "pay_id": "payment_id",
    "created_at": "created_at",
    "amount": "amount",
    "currency": "currency",
    "order_id": "order_id",
    "description": "description",
    "notes": "notes",
    "status": "status",
    "fee": "fee",
    "tax": "tax",
    "settlement_id": "settlement_id",
    "method": "method",
    "amount_refunded": "amount_refunded",
    "refund_status": "refund_status",
    "captured": "captured",
}

_RECON_COL_MAP = {
    "entity_id": "entity_id",
    "type": "type",
    "payment_id": "payment_id",
    "order_id": "order_id",
    "settlement_id": "settlement_id",
    "settlement_utr": "settlement_utr",
    "amount": "amount",
    "credit": "credit",
    "debit": "debit",
    "fee": "fee",
    "tax": "tax",
    "currency": "currency",
    "description": "description",
    "notes": "notes",
    "created_at": "created_at",
    "settled_at": "settled_at",
    "method": "method",
    "order_receipt": "order_receipt",
    "on_hold": "on_hold",
    "settled": "settled",
    "dispute_id": "dispute_id",
}

_LEDGER_COL_MAP = {
    "ledger_entry_id": "ledger_entry_id",
    "invoice_ref": "invoice_ref",
    "expected_amount": "expected_amount",
    "expected_date": "expected_date",
    "narrative_text": "narrative_text",
}

_BANK_COL_MAP = {
    "bank_txn_id": "bank_txn_id",
    "narration_text": "narration_text",
    "value_date": "value_date",
    "amount": "amount",
    "utr_reference": "utr_reference",
}


# ---------------------------------------------------------------------------
# Utility parsers
# ---------------------------------------------------------------------------

def _parse_amount(raw: Any) -> Optional[Decimal]:
    """
    Normalize an amount value to Decimal paise.
    Handles int, float, string, and None.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    try:
        return Decimal(str(raw)).quantize(Decimal("1"))
    except (InvalidOperation, ValueError, TypeError):
        return None


def _parse_datetime(raw: Any) -> Optional[datetime]:
    """
    Normalize a datetime value to UTC datetime.
    Handles: Unix timestamp (int/float), ISO 8601 string, YYYY-MM-DD string.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    if isinstance(raw, datetime):
        return raw.replace(tzinfo=timezone.utc) if raw.tzinfo is None else raw
    if isinstance(raw, (int, float)):
        try:
            return datetime.fromtimestamp(float(raw), tz=timezone.utc)
        except (OSError, ValueError, OverflowError):
            return None
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return None
        # Try ISO 8601 first
        for fmt in (
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%d/%m/%Y",
            "%d-%m-%Y",
        ):
            try:
                dt = datetime.strptime(raw, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                continue
        # Try numeric string
        try:
            return datetime.fromtimestamp(float(raw), tz=timezone.utc)
        except (ValueError, OSError, OverflowError):
            pass
    return None


def _parse_currency(raw: Any) -> Currency:
    """Normalize currency string to Currency enum. Defaults to INR."""
    if raw is None:
        return Currency.INR
    try:
        return Currency(str(raw).strip().upper())
    except ValueError:
        logger.warning("Unknown currency '%s', defaulting to INR", raw)
        return Currency.INR


def _parse_json_field(raw: Any) -> dict:
    """Safely parse a JSON string field. Returns empty dict on failure."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return {}


def _safe_str(raw: Any) -> str:
    """Convert to string, strip whitespace, handle None/NaN."""
    if raw is None:
        return ""
    if isinstance(raw, float) and pd.isna(raw):
        return ""
    return str(raw).strip()


def _rename_columns(df: pd.DataFrame, col_map: dict[str, str]) -> pd.DataFrame:
    """Rename columns using mapping. Unknown columns are preserved."""
    rename_dict = {k: v for k, v in col_map.items() if k in df.columns}
    return df.rename(columns=rename_dict)


# ---------------------------------------------------------------------------
# Per-source normalizers
# ---------------------------------------------------------------------------

def normalize_payments(df: pd.DataFrame, is_synthetic: bool = True) -> list[NormalizedTransaction]:
    """Normalize raw payment records (Source 1)."""
    df = _rename_columns(df.copy(), _PAYMENT_COL_MAP)
    results: list[NormalizedTransaction] = []

    for i, row in df.iterrows():
        payment_id = _safe_str(row.get("payment_id", ""))
        if not payment_id:
            logger.warning("Payment row %d missing payment_id, skipping", i)
            continue

        amount = _parse_amount(row.get("amount"))
        if amount is None:
            logger.warning("Payment %s missing amount, skipping", payment_id)
            continue

        txn_date = _parse_datetime(row.get("created_at"))
        if txn_date is None:
            logger.warning("Payment %s missing created_at, using epoch", payment_id)
            txn_date = datetime(1970, 1, 1, tzinfo=timezone.utc)

        notes_dict = _parse_json_field(row.get("notes", "{}"))
        description = _safe_str(row.get("description", ""))
        if not description:
            description = f"Payment {payment_id}"

        reference_ids = [payment_id]
        for ref_field in ("order_id", "settlement_id"):
            val = _safe_str(row.get(ref_field, ""))
            if val:
                reference_ids.append(val)

        txn = NormalizedTransaction(
            txn_id=f"PAY_{payment_id}",
            source=DataSource.RAZORPAY_PAYMENT,
            source_record_id=payment_id,
            amount=amount,
            currency=_parse_currency(row.get("currency")),
            transaction_date=txn_date,
            description=description,
            reference_ids=reference_ids,
            raw_fields=row.to_dict(),
            is_synthetic=is_synthetic,
            fee=_parse_amount(row.get("fee")),
            tax=_parse_amount(row.get("tax")),
            settlement_id=_safe_str(row.get("settlement_id")) or None,
            order_id=_safe_str(row.get("order_id")) or None,
            payment_id=payment_id,
            method=_safe_str(row.get("method")) or None,
            status=_safe_str(row.get("status")) or None,
        )
        results.append(txn)

    logger.info("Normalized %d/%d payment records", len(results), len(df))
    return results


def normalize_recon(df: pd.DataFrame, is_synthetic: bool = True) -> list[NormalizedTransaction]:
    """Normalize raw recon records (Source 2)."""
    df = _rename_columns(df.copy(), _RECON_COL_MAP)
    results: list[NormalizedTransaction] = []

    for i, row in df.iterrows():
        entity_id = _safe_str(row.get("entity_id", ""))
        if not entity_id:
            logger.warning("Recon row %d missing entity_id, skipping", i)
            continue

        # Prefer 'amount' column; fall back to credit - debit
        amount_raw = row.get("amount")
        if amount_raw is None or (isinstance(amount_raw, float) and pd.isna(amount_raw)):
            credit = _parse_amount(row.get("credit")) or Decimal(0)
            debit = _parse_amount(row.get("debit")) or Decimal(0)
            amount = credit - debit
        else:
            amount = _parse_amount(amount_raw) or Decimal(0)

        txn_date = _parse_datetime(row.get("created_at"))
        if txn_date is None:
            txn_date = _parse_datetime(row.get("settled_at"))
        if txn_date is None:
            txn_date = datetime(1970, 1, 1, tzinfo=timezone.utc)

        payment_id = _safe_str(row.get("payment_id", ""))
        order_id = _safe_str(row.get("order_id", ""))
        settlement_id = _safe_str(row.get("settlement_id", ""))
        utr = _safe_str(row.get("settlement_utr", ""))

        reference_ids = [entity_id]
        for ref in (payment_id, order_id, settlement_id, utr):
            if ref:
                reference_ids.append(ref)

        description = _safe_str(row.get("description", ""))
        if not description:
            description = f"Recon {entity_id}"

        txn = NormalizedTransaction(
            txn_id=f"RECON_{entity_id}",
            source=DataSource.RAZORPAY_RECON,
            source_record_id=entity_id,
            amount=amount,
            currency=_parse_currency(row.get("currency")),
            transaction_date=txn_date,
            description=description,
            reference_ids=reference_ids,
            raw_fields=row.to_dict(),
            is_synthetic=is_synthetic,
            fee=_parse_amount(row.get("fee")),
            tax=_parse_amount(row.get("tax")),
            settlement_id=settlement_id or None,
            order_id=order_id or None,
            payment_id=payment_id or None,
            method=_safe_str(row.get("method")) or None,
            status=_safe_str(row.get("type")) or None,
        )
        results.append(txn)

    logger.info("Normalized %d/%d recon records", len(results), len(df))
    return results


def normalize_ledger(df: pd.DataFrame, is_synthetic: bool = True) -> list[NormalizedTransaction]:
    """Normalize merchant ledger records (Source 3)."""
    df = _rename_columns(df.copy(), _LEDGER_COL_MAP)
    results: list[NormalizedTransaction] = []

    for i, row in df.iterrows():
        ledger_id = _safe_str(row.get("ledger_entry_id", ""))
        if not ledger_id:
            logger.warning("Ledger row %d missing ledger_entry_id, skipping", i)
            continue

        amount = _parse_amount(row.get("expected_amount"))
        if amount is None:
            logger.warning("Ledger %s missing expected_amount, skipping", ledger_id)
            continue

        txn_date = _parse_datetime(row.get("expected_date"))
        if txn_date is None:
            txn_date = datetime(1970, 1, 1, tzinfo=timezone.utc)

        invoice_ref = _safe_str(row.get("invoice_ref", ""))
        reference_ids = [ledger_id]
        if invoice_ref:
            reference_ids.append(invoice_ref)

        description = _safe_str(row.get("narrative_text", ""))
        if not description:
            description = f"Ledger {ledger_id}"

        txn = NormalizedTransaction(
            txn_id=f"LDG_{ledger_id}",
            source=DataSource.MERCHANT_LEDGER,
            source_record_id=ledger_id,
            amount=amount,
            currency=Currency.INR,
            transaction_date=txn_date,
            description=description,
            reference_ids=reference_ids,
            raw_fields=row.to_dict(),
            is_synthetic=is_synthetic,
        )
        results.append(txn)

    logger.info("Normalized %d/%d ledger records", len(results), len(df))
    return results


def normalize_bank(df: pd.DataFrame, is_synthetic: bool = True) -> list[NormalizedTransaction]:
    """Normalize bank statement records (Source 4)."""
    df = _rename_columns(df.copy(), _BANK_COL_MAP)
    results: list[NormalizedTransaction] = []

    for i, row in df.iterrows():
        bank_id = _safe_str(row.get("bank_txn_id", ""))
        if not bank_id:
            logger.warning("Bank row %d missing bank_txn_id, skipping", i)
            continue

        amount = _parse_amount(row.get("amount"))
        if amount is None:
            logger.warning("Bank %s missing amount, skipping", bank_id)
            continue

        txn_date = _parse_datetime(row.get("value_date"))
        if txn_date is None:
            txn_date = datetime(1970, 1, 1, tzinfo=timezone.utc)

        utr = _safe_str(row.get("utr_reference", ""))
        reference_ids = [bank_id]
        if utr:
            reference_ids.append(utr)

        description = _safe_str(row.get("narration_text", ""))
        if not description:
            description = f"Bank {bank_id}"

        txn = NormalizedTransaction(
            txn_id=f"BNK_{bank_id}",
            source=DataSource.BANK_STATEMENT,
            source_record_id=bank_id,
            amount=amount,
            currency=Currency.INR,
            transaction_date=txn_date,
            description=description,
            reference_ids=reference_ids,
            raw_fields=row.to_dict(),
            is_synthetic=is_synthetic,
        )
        results.append(txn)

    logger.info("Normalized %d/%d bank records", len(results), len(df))
    return results


# ---------------------------------------------------------------------------
# Load from CSVs
# ---------------------------------------------------------------------------

def load_and_normalize_all(
    data_dir: Path,
    is_synthetic: bool = True,
) -> dict[str, list[NormalizedTransaction]]:
    """
    Load raw CSVs from data_dir/raw/ and normalize all four sources.
    Returns dict with keys: payments, recon, ledger, bank.
    """
    raw_dir = data_dir / "raw"

    def _load(name: str) -> pd.DataFrame:
        path = raw_dir / f"{name}.csv"
        if not path.exists():
            logger.warning("Raw data file not found: %s", path)
            return pd.DataFrame()
        df = pd.read_csv(path, dtype=str)  # Load all as strings first
        logger.info("Loaded %d rows from %s", len(df), path)
        return df

    return {
        "payments": normalize_payments(_load("payments"), is_synthetic),
        "recon": normalize_recon(_load("recon"), is_synthetic),
        "ledger": normalize_ledger(_load("ledger"), is_synthetic),
        "bank": normalize_bank(_load("bank"), is_synthetic),
    }
