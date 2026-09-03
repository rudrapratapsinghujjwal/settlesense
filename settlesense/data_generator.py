"""
SettleSense — Synthetic Data Generator
======================================
Generates 200 labeled records:
  - 120 clean (exact matches)
  -  20 split_settlement
  -  20 refund_misattribution
  -  20 fee_tier
  -  20 near_duplicate

Design constraints:
  - Deterministic given RANDOM_SEED
  - Labels NEVER leaked into narrative text
  - Ambiguity emerges from relational/numerical structure, not keywords
  - Four separate data-source CSVs produced

SYNTHETIC DATA — NOT real merchant data.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import random
import string
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants / realistic pools
# ---------------------------------------------------------------------------

RANDOM_SEED = 42          # Overrideable at call time

# Realistic narration prefixes — no label keywords
NARRATION_PREFIXES = [
    "NEFT", "IMPS", "RTGS", "UPI", "NACH",
]
NARRATION_BANKS = [
    "HDFC", "ICICI", "SBI", "AXIS", "KOTAK", "YES", "INDUSIND", "FEDERAL",
]
SETTLEMENT_PREFIXES = ["RZPY", "RAZPAY", "RPAY", "RPY"]
PRODUCT_NAMES = [
    "Premium Plan", "Basic Subscription", "Annual License", "Enterprise Package",
    "Top-up Credit", "Consultation Fee", "Platform Access", "Monthly Pass",
    "Service Charge", "Renewal", "Add-on Module", "Standard Tier",
]
UTR_PREFIX = "UTR"

# Razorpay-style fee tiers (SYNTHETIC ASSUMPTION — documented as such)
# Tier: (lower_amount_paise, upper_amount_paise, rate_bps, flat_fee_paise)
# ASSUMPTION: These fee rules are illustrative synthetic assumptions.
# Real Razorpay fee rules may differ. See README.
SYNTHETIC_FEE_TIERS = [
    (0,        50_000,   200, 0),      # 0–₹500: 2.00%
    (50_000,   200_000,  180, 0),      # ₹500–₹2000: 1.80%
    (200_000,  500_000,  160, 0),      # ₹2000–₹5000: 1.60%
    (500_000,  2000_000, 140, 0),      # ₹5000–₹20000: 1.40%
    (2000_000, None,     120, 0),      # ₹20000+: 1.20%
]
GST_RATE = Decimal("0.18")


def compute_synthetic_fee(amount_paise: int) -> tuple[int, int, int]:
    """
    Returns (fee_paise, tax_paise, rate_bps) for a given amount.
    SYNTHETIC ASSUMPTION — not real Razorpay pricing.
    """
    for lower, upper, rate_bps, flat in SYNTHETIC_FEE_TIERS:
        if upper is None or amount_paise < upper:
            fee_paise = int(amount_paise * rate_bps / 10_000) + flat
            tax_paise = int(fee_paise * float(GST_RATE))
            return fee_paise, tax_paise, rate_bps
    # Fallback to lowest rate
    fee_paise = int(amount_paise * 120 / 10_000)
    tax_paise = int(fee_paise * float(GST_RATE))
    return fee_paise, tax_paise, 120


# ---------------------------------------------------------------------------
# ID generators
# ---------------------------------------------------------------------------

_id_counter: dict[str, int] = {}


def _make_id(prefix: str, rng: random.Random) -> str:
    suffix = "".join(rng.choices(string.ascii_uppercase + string.digits, k=14))
    return f"{prefix}{suffix}"


def _utr(rng: random.Random) -> str:
    return f"{UTR_PREFIX}{rng.randint(10_000_000_000, 99_999_999_999)}"


# ---------------------------------------------------------------------------
# Row dataclasses (matching actual Razorpay schema where applicable)
# ---------------------------------------------------------------------------

@dataclass
class PaymentRow:
    """SOURCE 1 — Razorpay Payment record schema."""
    id: str                   # pay_XXXX
    order_id: str             # order_XXXX
    amount: int               # paise
    currency: str
    status: str               # captured | refunded | partially_refunded
    method: str               # upi | netbanking | card | wallet
    captured: bool
    amount_refunded: int      # paise
    refund_status: Optional[str]
    fee: int                  # paise
    tax: int                  # paise
    description: str
    notes: str                # JSON string — untrusted field
    created_at: int           # Unix timestamp
    settlement_id: str


@dataclass
class ReconRow:
    """SOURCE 2 — Razorpay Recon record schema."""
    entity_id: str
    type: str                 # payment | refund | adjustment | transfer
    debit: int
    credit: int
    amount: int
    currency: str
    fee: int
    tax: int
    on_hold: bool
    settled: bool
    created_at: int
    settled_at: Optional[int]
    settlement_id: str
    description: str
    notes: str
    payment_id: str
    settlement_utr: str
    order_id: str
    order_receipt: str
    method: str
    card_network: Optional[str]
    card_issuer: Optional[str]
    card_type: Optional[str]
    dispute_id: Optional[str]


@dataclass
class LedgerRow:
    """SOURCE 3 — Merchant internal ledger."""
    ledger_entry_id: str
    invoice_ref: str
    expected_amount: int      # paise
    expected_date: str        # YYYY-MM-DD
    narrative_text: str


@dataclass
class BankRow:
    """SOURCE 4 — Bank statement."""
    bank_txn_id: str
    narration_text: str
    value_date: str           # YYYY-MM-DD
    amount: int               # paise (credit from merchant's perspective)
    utr_reference: str


@dataclass
class GroundTruthRow:
    """Sealed answer key — NEVER read by inference pipeline."""
    record_id: str
    true_category: str
    payment_id: str
    ledger_id: str
    linked_ids: str           # JSON list


# ---------------------------------------------------------------------------
# Generator helpers
# ---------------------------------------------------------------------------

def _realistic_narration(rng: random.Random, settlement_utr: str, bank: Optional[str] = None) -> str:
    """Build a realistic bank narration. Never embeds the exception class."""
    prefix = rng.choice(NARRATION_PREFIXES)
    bk = bank or rng.choice(NARRATION_BANKS)
    parts = [prefix, settlement_utr, f"RAZORPAY/{bk}"]
    # Random noise: sometimes add trailing digits
    if rng.random() < 0.3:
        parts.append(str(rng.randint(1000, 9999)))
    return "/".join(parts)


def _realistic_description(rng: random.Random, product: Optional[str] = None) -> str:
    prod = product or rng.choice(PRODUCT_NAMES)
    return f"Payment for {prod}"


def _ts(base_date: datetime, jitter_seconds: int = 0) -> int:
    return int((base_date + timedelta(seconds=jitter_seconds)).timestamp())


def _iso(base_date: datetime, jitter_days: int = 0) -> str:
    return (base_date + timedelta(days=jitter_days)).strftime("%Y-%m-%d")


def _method(rng: random.Random) -> str:
    return rng.choice(["upi", "netbanking", "card", "wallet"])


def _card_fields(rng: random.Random, method: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    if method == "card":
        network = rng.choice(["Visa", "Mastercard", "RuPay", "Amex"])
        issuer = rng.choice(["HDFC", "ICICI", "SBI", "AXIS"])
        card_type = rng.choice(["credit", "debit"])
        return network, issuer, card_type
    return None, None, None


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

class SyntheticDataGenerator:
    """
    Generates all four data sources with sealed labels.

    Usage:
        gen = SyntheticDataGenerator(seed=42)
        datasets = gen.generate()
        gen.save(datasets, output_dir)
    """

    def __init__(self, seed: int = RANDOM_SEED):
        self.seed = seed
        self.rng = random.Random(seed)

    def _base_date(self) -> datetime:
        return datetime(2025, 1, 1, 9, 0, 0)

    def _gen_payment_id(self) -> str:
        return _make_id("pay_", self.rng)

    def _gen_order_id(self) -> str:
        return _make_id("order_", self.rng)

    def _gen_settlement_id(self) -> str:
        return _make_id("setl_", self.rng)

    def _gen_recon_id(self) -> str:
        return _make_id("recon_", self.rng)

    def _gen_ledger_id(self) -> str:
        return _make_id("LDG_", self.rng)

    def _gen_invoice_ref(self) -> str:
        return f"INV-{self.rng.randint(10000, 99999)}"

    def _gen_bank_id(self) -> str:
        return _make_id("BNK_", self.rng)

    def generate(self) -> dict[str, list]:
        """Generate all records. Returns dict of lists of row dataclasses."""
        payments: list[PaymentRow] = []
        recons: list[ReconRow] = []
        ledger: list[LedgerRow] = []
        bank: list[BankRow] = []
        ground_truth: list[GroundTruthRow] = []

        base = self._base_date()
        record_counter = [0]

        def next_record_id(category: str) -> str:
            record_counter[0] += 1
            return f"REC_{record_counter[0]:04d}"

        # --- 120 CLEAN records ---
        for i in range(120):
            rec_id = next_record_id("clean")
            amount = self.rng.randint(5_000, 500_000)  # ₹50 – ₹5000
            fee, tax, _ = compute_synthetic_fee(amount)
            method = _method(self.rng)
            cn, ci, ct = _card_fields(self.rng, method)
            product = self.rng.choice(PRODUCT_NAMES)
            pay_id = self._gen_payment_id()
            order_id = self._gen_order_id()
            setl_id = self._gen_settlement_id()
            utr = _utr(self.rng)
            day_offset = i % 60  # Spread over 60 days
            txn_date = base + timedelta(days=day_offset, hours=self.rng.randint(8, 20))
            settlement_date = txn_date + timedelta(days=self.rng.randint(1, 3))
            note_json = json.dumps({"order": order_id})  # No label

            pay = PaymentRow(
                id=pay_id,
                order_id=order_id,
                amount=amount,
                currency="INR",
                status="captured",
                method=method,
                captured=True,
                amount_refunded=0,
                refund_status=None,
                fee=fee,
                tax=tax,
                description=_realistic_description(self.rng, product),
                notes=note_json,
                created_at=_ts(txn_date),
                settlement_id=setl_id,
            )
            recon = ReconRow(
                entity_id=self._gen_recon_id(),
                type="payment",
                debit=0,
                credit=amount - fee - tax,
                amount=amount,
                currency="INR",
                fee=fee,
                tax=tax,
                on_hold=False,
                settled=True,
                created_at=_ts(txn_date),
                settled_at=_ts(settlement_date),
                settlement_id=setl_id,
                description=_realistic_description(self.rng, product),
                notes=note_json,
                payment_id=pay_id,
                settlement_utr=utr,
                order_id=order_id,
                order_receipt=self._gen_invoice_ref(),
                method=method,
                card_network=cn,
                card_issuer=ci,
                card_type=ct,
                dispute_id=None,
            )
            lgd_id = self._gen_ledger_id()
            ledger_row = LedgerRow(
                ledger_entry_id=lgd_id,
                invoice_ref=recon.order_receipt,
                expected_amount=amount,
                expected_date=_iso(txn_date),
                narrative_text=f"Settlement for {product}",
            )
            bank_row = BankRow(
                bank_txn_id=self._gen_bank_id(),
                narration_text=_realistic_narration(self.rng, utr),
                value_date=_iso(settlement_date),
                amount=amount - fee - tax,
                utr_reference=utr,
            )

            payments.append(pay)
            recons.append(recon)
            ledger.append(ledger_row)
            bank.append(bank_row)
            ground_truth.append(GroundTruthRow(
                record_id=rec_id,
                true_category="clean",
                payment_id=pay_id,
                ledger_id=lgd_id,
                linked_ids=json.dumps([pay_id]),
            ))

        # --- 20 SPLIT SETTLEMENT ---
        # One ledger entry expects amount X.
        # But it was settled in two separate batches (settlement IDs differ).
        # Neither batch alone matches X, but they sum to X - fees.
        # The correct grouping is not obvious from amount alone.
        for i in range(20):
            rec_id = next_record_id("split_settlement")
            total_amount = self.rng.randint(50_000, 300_000)
            # Split into two arbitrary parts (not equal halves)
            split_ratio = self.rng.uniform(0.3, 0.7)
            amount_a = int(total_amount * split_ratio)
            amount_b = total_amount - amount_a
            fee_a, tax_a, _ = compute_synthetic_fee(amount_a)
            fee_b, tax_b, _ = compute_synthetic_fee(amount_b)

            method = _method(self.rng)
            product = self.rng.choice(PRODUCT_NAMES)
            pay_id_a = self._gen_payment_id()
            pay_id_b = self._gen_payment_id()
            order_id = self._gen_order_id()
            setl_id_a = self._gen_settlement_id()
            setl_id_b = self._gen_settlement_id()
            utr_a = _utr(self.rng)
            utr_b = _utr(self.rng)
            day_offset = 60 + i
            txn_date = base + timedelta(days=day_offset, hours=self.rng.randint(8, 20))
            # Second settlement 1–2 days later (timing creates ambiguity)
            settlement_date_a = txn_date + timedelta(days=1)
            settlement_date_b = txn_date + timedelta(days=2)

            cn, ci, ct = _card_fields(self.rng, method)
            note_json = json.dumps({"order": order_id})

            pay_a = PaymentRow(
                id=pay_id_a, order_id=order_id, amount=amount_a, currency="INR",
                status="captured", method=method, captured=True, amount_refunded=0,
                refund_status=None, fee=fee_a, tax=tax_a,
                description=_realistic_description(self.rng, product),
                notes=note_json, created_at=_ts(txn_date), settlement_id=setl_id_a,
            )
            pay_b = PaymentRow(
                id=pay_id_b, order_id=order_id, amount=amount_b, currency="INR",
                status="captured", method=method, captured=True, amount_refunded=0,
                refund_status=None, fee=fee_b, tax=tax_b,
                description=_realistic_description(self.rng, product),
                notes=note_json, created_at=_ts(txn_date, jitter_seconds=self.rng.randint(60, 3600)),
                settlement_id=setl_id_b,
            )
            recon_a = ReconRow(
                entity_id=self._gen_recon_id(), type="payment",
                debit=0, credit=amount_a - fee_a - tax_a, amount=amount_a,
                currency="INR", fee=fee_a, tax=tax_a, on_hold=False, settled=True,
                created_at=_ts(txn_date), settled_at=_ts(settlement_date_a),
                settlement_id=setl_id_a,
                description=_realistic_description(self.rng, product),
                notes=note_json, payment_id=pay_id_a, settlement_utr=utr_a,
                order_id=order_id, order_receipt=self._gen_invoice_ref(),
                method=method, card_network=cn, card_issuer=ci, card_type=ct,
                dispute_id=None,
            )
            recon_b = ReconRow(
                entity_id=self._gen_recon_id(), type="payment",
                debit=0, credit=amount_b - fee_b - tax_b, amount=amount_b,
                currency="INR", fee=fee_b, tax=tax_b, on_hold=False, settled=True,
                created_at=_ts(txn_date, jitter_seconds=self.rng.randint(60, 3600)),
                settled_at=_ts(settlement_date_b),
                settlement_id=setl_id_b,
                description=_realistic_description(self.rng, product),
                notes=note_json, payment_id=pay_id_b, settlement_utr=utr_b,
                order_id=order_id, order_receipt=recon_a.order_receipt,
                method=method, card_network=cn, card_issuer=ci, card_type=ct,
                dispute_id=None,
            )
            lgd_id = self._gen_ledger_id()
            ledger_row = LedgerRow(
                ledger_entry_id=lgd_id,
                invoice_ref=recon_a.order_receipt,
                expected_amount=total_amount,
                expected_date=_iso(txn_date),
                narrative_text=f"Settlement for {product}",
            )
            # Only one bank credit visible per settlement (two bank credits)
            bank_row_a = BankRow(
                bank_txn_id=self._gen_bank_id(),
                narration_text=_realistic_narration(self.rng, utr_a),
                value_date=_iso(settlement_date_a),
                amount=amount_a - fee_a - tax_a,
                utr_reference=utr_a,
            )
            bank_row_b = BankRow(
                bank_txn_id=self._gen_bank_id(),
                narration_text=_realistic_narration(self.rng, utr_b),
                value_date=_iso(settlement_date_b),
                amount=amount_b - fee_b - tax_b,
                utr_reference=utr_b,
            )

            payments.extend([pay_a, pay_b])
            recons.extend([recon_a, recon_b])
            ledger.append(ledger_row)
            bank.extend([bank_row_a, bank_row_b])
            ground_truth.append(GroundTruthRow(
                record_id=rec_id,
                true_category="split_settlement",
                payment_id=pay_id_a,
                ledger_id=lgd_id,
                linked_ids=json.dumps([pay_id_a, pay_id_b, setl_id_a, setl_id_b]),
            ))

        # --- 20 REFUND MISATTRIBUTION ---
        # A refund event is ambiguous between two original payments:
        # same approximate amount, close timestamps, overlapping context.
        # Correct origin determined by relational structure.
        for i in range(20):
            rec_id = next_record_id("refund_misattribution")
            amount = self.rng.randint(10_000, 100_000)
            # Add small noise so amounts are close but not identical
            decoy_amount = amount + self.rng.randint(-500, 500)
            fee, tax, _ = compute_synthetic_fee(amount)
            fee_d, tax_d, _ = compute_synthetic_fee(decoy_amount)

            method = _method(self.rng)
            product = self.rng.choice(PRODUCT_NAMES)
            pay_id_true = self._gen_payment_id()    # The actual refunded payment
            pay_id_decoy = self._gen_payment_id()   # Red herring payment
            order_id_true = self._gen_order_id()
            order_id_decoy = self._gen_order_id()
            setl_id = self._gen_settlement_id()
            utr = _utr(self.rng)
            day_offset = 80 + i
            txn_date = base + timedelta(days=day_offset, hours=self.rng.randint(8, 20))
            refund_date = txn_date + timedelta(days=self.rng.randint(1, 5))
            # Decoy was created at a very similar time (creates ambiguity)
            decoy_date = txn_date + timedelta(hours=self.rng.randint(1, 6))

            cn, ci, ct = _card_fields(self.rng, method)

            pay_true = PaymentRow(
                id=pay_id_true, order_id=order_id_true, amount=amount, currency="INR",
                status="refunded", method=method, captured=True, amount_refunded=amount,
                refund_status="full",
                fee=fee, tax=tax,
                description=_realistic_description(self.rng, product),
                notes=json.dumps({"order": order_id_true}),
                created_at=_ts(txn_date), settlement_id=setl_id,
            )
            pay_decoy = PaymentRow(
                id=pay_id_decoy, order_id=order_id_decoy, amount=decoy_amount, currency="INR",
                status="captured", method=method, captured=True, amount_refunded=0,
                refund_status=None,
                fee=fee_d, tax=tax_d,
                description=_realistic_description(self.rng, product),
                notes=json.dumps({"order": order_id_decoy}),
                created_at=_ts(decoy_date), settlement_id=self._gen_settlement_id(),
            )
            # Refund recon entry
            recon_refund = ReconRow(
                entity_id=self._gen_recon_id(), type="refund",
                debit=amount, credit=0, amount=amount,
                currency="INR", fee=0, tax=0, on_hold=False, settled=True,
                created_at=_ts(refund_date), settled_at=_ts(refund_date + timedelta(days=1)),
                settlement_id=setl_id,
                description=f"Refund - {product}",
                notes=json.dumps({"order": order_id_true}),
                payment_id=pay_id_true, settlement_utr=utr,
                order_id=order_id_true, order_receipt=self._gen_invoice_ref(),
                method=method, card_network=cn, card_issuer=ci, card_type=ct,
                dispute_id=None,
            )
            lgd_id = self._gen_ledger_id()
            ledger_row = LedgerRow(
                ledger_entry_id=lgd_id,
                invoice_ref=recon_refund.order_receipt,
                expected_amount=-amount,   # negative = expected credit back
                expected_date=_iso(refund_date),
                narrative_text=f"Refund credit expected",
            )
            bank_row = BankRow(
                bank_txn_id=self._gen_bank_id(),
                narration_text=_realistic_narration(self.rng, utr),
                value_date=_iso(refund_date + timedelta(days=1)),
                amount=-amount,   # debit from merchant POV
                utr_reference=utr,
            )

            payments.extend([pay_true, pay_decoy])
            recons.append(recon_refund)
            ledger.append(ledger_row)
            bank.append(bank_row)
            ground_truth.append(GroundTruthRow(
                record_id=rec_id,
                true_category="refund_misattribution",
                payment_id=pay_id_true,
                ledger_id=lgd_id,
                linked_ids=json.dumps([pay_id_true, pay_id_decoy]),
            ))

        # --- 20 FEE TIER ---
        # Observed fee differs from what a naive single-rate calculation gives.
        # Correct fee requires identifying the applicable tier based on amount.
        # The ambiguity: ledger recorded incorrect expected fee for a borderline amount.
        for i in range(20):
            rec_id = next_record_id("fee_tier")
            # Choose borderline amounts (near tier boundaries) to create ambiguity
            tier_boundaries = [50_000, 200_000, 500_000, 2_000_000]
            boundary = self.rng.choice(tier_boundaries)
            # Amount within ±10% of a tier boundary
            amount = boundary + self.rng.randint(
                -int(boundary * 0.05), int(boundary * 0.05)
            )
            amount = max(1_000, amount)  # at least ₹10

            fee_actual, tax_actual, tier_rate = compute_synthetic_fee(amount)
            # Ledger used a naive wrong rate (one tier below actual)
            naive_rate = 200  # always assumes 2% regardless of tier
            fee_expected_naive = int(amount * naive_rate / 10_000)
            tax_expected_naive = int(fee_expected_naive * float(GST_RATE))

            method = _method(self.rng)
            product = self.rng.choice(PRODUCT_NAMES)
            pay_id = self._gen_payment_id()
            order_id = self._gen_order_id()
            setl_id = self._gen_settlement_id()
            utr = _utr(self.rng)
            day_offset = 100 + i
            txn_date = base + timedelta(days=day_offset, hours=self.rng.randint(8, 20))
            settlement_date = txn_date + timedelta(days=self.rng.randint(1, 3))
            cn, ci, ct = _card_fields(self.rng, method)

            pay = PaymentRow(
                id=pay_id, order_id=order_id, amount=amount, currency="INR",
                status="captured", method=method, captured=True, amount_refunded=0,
                refund_status=None, fee=fee_actual, tax=tax_actual,
                description=_realistic_description(self.rng, product),
                notes=json.dumps({"order": order_id}),
                created_at=_ts(txn_date), settlement_id=setl_id,
            )
            recon = ReconRow(
                entity_id=self._gen_recon_id(), type="payment",
                debit=0, credit=amount - fee_actual - tax_actual, amount=amount,
                currency="INR", fee=fee_actual, tax=tax_actual,
                on_hold=False, settled=True,
                created_at=_ts(txn_date), settled_at=_ts(settlement_date),
                settlement_id=setl_id,
                description=_realistic_description(self.rng, product),
                notes=json.dumps({"order": order_id}),
                payment_id=pay_id, settlement_utr=utr,
                order_id=order_id, order_receipt=self._gen_invoice_ref(),
                method=method, card_network=cn, card_issuer=ci, card_type=ct,
                dispute_id=None,
            )
            lgd_id = self._gen_ledger_id()
            # Ledger records expected_amount as (amount - naive_fee - naive_tax) — wrong
            ledger_row = LedgerRow(
                ledger_entry_id=lgd_id,
                invoice_ref=recon.order_receipt,
                expected_amount=amount - fee_expected_naive - tax_expected_naive,
                expected_date=_iso(txn_date),
                narrative_text=f"Settlement expected for {product}",
            )
            bank_row = BankRow(
                bank_txn_id=self._gen_bank_id(),
                narration_text=_realistic_narration(self.rng, utr),
                value_date=_iso(settlement_date),
                amount=amount - fee_actual - tax_actual,  # Actual bank credit
                utr_reference=utr,
            )

            payments.append(pay)
            recons.append(recon)
            ledger.append(ledger_row)
            bank.append(bank_row)
            ground_truth.append(GroundTruthRow(
                record_id=rec_id,
                true_category="fee_tier",
                payment_id=pay_id,
                ledger_id=lgd_id,
                linked_ids=json.dumps([pay_id, setl_id]),
            ))

        # --- 20 NEAR DUPLICATE ---
        # Two highly similar transactions — one is a duplicate, one is legitimate.
        # Must weigh timing, description, amount, order IDs — not one threshold.
        for i in range(20):
            rec_id = next_record_id("near_duplicate")
            amount = self.rng.randint(5_000, 100_000)
            # Duplicate has exactly same amount, legitimate has minor variation
            is_true_duplicate = self.rng.random() > 0.5
            if is_true_duplicate:
                amount_b = amount  # Exact duplicate
                time_gap_seconds = self.rng.randint(30, 300)  # Very close in time
            else:
                amount_b = amount + self.rng.randint(-200, 200)  # Slightly different
                time_gap_seconds = self.rng.randint(3600, 86400)  # Hours apart

            fee, tax, _ = compute_synthetic_fee(amount)
            fee_b, tax_b, _ = compute_synthetic_fee(amount_b)
            method = _method(self.rng)
            product = self.rng.choice(PRODUCT_NAMES)

            # True duplicate: same order_id (that's the key signal)
            # Legitimate: different order_id
            order_id_a = self._gen_order_id()
            order_id_b = order_id_a if is_true_duplicate else self._gen_order_id()

            pay_id_a = self._gen_payment_id()
            pay_id_b = self._gen_payment_id()
            setl_id_a = self._gen_settlement_id()
            setl_id_b = self._gen_settlement_id()
            utr_a = _utr(self.rng)
            utr_b = _utr(self.rng)
            day_offset = 120 + i
            txn_date = base + timedelta(days=day_offset, hours=self.rng.randint(8, 20))
            txn_date_b = txn_date + timedelta(seconds=time_gap_seconds)
            settlement_date = txn_date + timedelta(days=1)
            cn, ci, ct = _card_fields(self.rng, method)

            pay_a = PaymentRow(
                id=pay_id_a, order_id=order_id_a, amount=amount, currency="INR",
                status="captured", method=method, captured=True, amount_refunded=0,
                refund_status=None, fee=fee, tax=tax,
                description=_realistic_description(self.rng, product),
                notes=json.dumps({"order": order_id_a}),
                created_at=_ts(txn_date), settlement_id=setl_id_a,
            )
            pay_b = PaymentRow(
                id=pay_id_b, order_id=order_id_b, amount=amount_b, currency="INR",
                status="captured", method=method, captured=True, amount_refunded=0,
                refund_status=None, fee=fee_b, tax=tax_b,
                description=_realistic_description(self.rng, product),
                notes=json.dumps({"order": order_id_b}),
                created_at=_ts(txn_date_b), settlement_id=setl_id_b,
            )
            recon_a = ReconRow(
                entity_id=self._gen_recon_id(), type="payment",
                debit=0, credit=amount - fee - tax, amount=amount,
                currency="INR", fee=fee, tax=tax, on_hold=False, settled=True,
                created_at=_ts(txn_date), settled_at=_ts(settlement_date),
                settlement_id=setl_id_a,
                description=_realistic_description(self.rng, product),
                notes=json.dumps({"order": order_id_a}),
                payment_id=pay_id_a, settlement_utr=utr_a,
                order_id=order_id_a, order_receipt=self._gen_invoice_ref(),
                method=method, card_network=cn, card_issuer=ci, card_type=ct,
                dispute_id=None,
            )
            recon_b = ReconRow(
                entity_id=self._gen_recon_id(), type="payment",
                debit=0, credit=amount_b - fee_b - tax_b, amount=amount_b,
                currency="INR", fee=fee_b, tax=tax_b, on_hold=False, settled=True,
                created_at=_ts(txn_date_b), settled_at=_ts(settlement_date + timedelta(days=1)),
                settlement_id=setl_id_b,
                description=_realistic_description(self.rng, product),
                notes=json.dumps({"order": order_id_b}),
                payment_id=pay_id_b, settlement_utr=utr_b,
                order_id=order_id_b, order_receipt=self._gen_invoice_ref(),
                method=method, card_network=cn, card_issuer=ci, card_type=ct,
                dispute_id=None,
            )
            lgd_id_a = self._gen_ledger_id()
            ledger_row_a = LedgerRow(
                ledger_entry_id=lgd_id_a,
                invoice_ref=recon_a.order_receipt,
                expected_amount=amount,
                expected_date=_iso(txn_date),
                narrative_text=f"Settlement for {product}",
            )
            bank_row_a = BankRow(
                bank_txn_id=self._gen_bank_id(),
                narration_text=_realistic_narration(self.rng, utr_a),
                value_date=_iso(settlement_date),
                amount=amount - fee - tax,
                utr_reference=utr_a,
            )
            bank_row_b = BankRow(
                bank_txn_id=self._gen_bank_id(),
                narration_text=_realistic_narration(self.rng, utr_b),
                value_date=_iso(settlement_date + timedelta(days=1)),
                amount=amount_b - fee_b - tax_b,
                utr_reference=utr_b,
            )

            payments.extend([pay_a, pay_b])
            recons.extend([recon_a, recon_b])
            ledger.append(ledger_row_a)
            bank.extend([bank_row_a, bank_row_b])
            ground_truth.append(GroundTruthRow(
                record_id=rec_id,
                true_category="near_duplicate",
                payment_id=pay_id_a,
                ledger_id=lgd_id_a,
                linked_ids=json.dumps([pay_id_a, pay_id_b]),
            ))

        return {
            "payments": payments,
            "recons": recons,
            "ledger": ledger,
            "bank": bank,
            "ground_truth": ground_truth,
        }

    def save(self, datasets: dict, output_dir: Path) -> dict[str, Path]:
        """
        Save all datasets to CSV files.
        Ground truth is saved to the answer_keys directory (sealed).
        Returns dict of {name: path}.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        answer_key_dir = output_dir / "answer_keys"
        answer_key_dir.mkdir(parents=True, exist_ok=True)

        paths = {}

        # Save payment, recon, ledger, bank to raw/
        raw_dir = output_dir / "raw"
        raw_dir.mkdir(exist_ok=True)

        def _write_csv(rows: list, path: Path) -> None:
            if not rows:
                return
            import dataclasses
            fieldnames = [f.name for f in dataclasses.fields(rows[0])]
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow(dataclasses.asdict(row))

        _write_csv(datasets["payments"], raw_dir / "payments.csv")
        _write_csv(datasets["recons"], raw_dir / "recon.csv")
        _write_csv(datasets["ledger"], raw_dir / "ledger.csv")
        _write_csv(datasets["bank"], raw_dir / "bank.csv")
        _write_csv(datasets["ground_truth"], answer_key_dir / "answer_key_all.csv")

        paths["payments"] = raw_dir / "payments.csv"
        paths["recon"] = raw_dir / "recon.csv"
        paths["ledger"] = raw_dir / "ledger.csv"
        paths["bank"] = raw_dir / "bank.csv"
        paths["ground_truth_all"] = answer_key_dir / "answer_key_all.csv"

        logger.info(
            "Synthetic data saved | payments=%d | recons=%d | ledger=%d | bank=%d | gt=%d",
            len(datasets["payments"]), len(datasets["recons"]),
            len(datasets["ledger"]), len(datasets["bank"]),
            len(datasets["ground_truth"]),
        )
        return paths

    def split_ground_truth(
        self,
        ground_truth: list[GroundTruthRow],
        output_dir: Path,
        seed: int,
    ) -> dict[str, Path]:
        """
        Split ground truth into tune/validation/holdout (60/20/20).
        Stratified by category.
        Holdout answer key is saved to a separate sealed directory.
        Returns paths.
        """
        from collections import defaultdict

        # Group by category
        by_cat: dict[str, list] = defaultdict(list)
        for row in ground_truth:
            by_cat[row.true_category].append(row)

        rng = random.Random(seed)
        tune, validation, holdout = [], [], []

        for cat, rows in by_cat.items():
            rng.shuffle(rows)
            n = len(rows)
            n_tune = int(n * 0.6)
            n_val = int(n * 0.2)
            tune.extend(rows[:n_tune])
            validation.extend(rows[n_tune:n_tune + n_val])
            holdout.extend(rows[n_tune + n_val:])

        # Add split field
        def _with_split(rows: list, split: str) -> list[dict]:
            import dataclasses
            result = []
            for r in rows:
                d = dataclasses.asdict(r)
                d["split"] = split
                result.append(d)
            return result

        tune_dicts = _with_split(tune, "tune")
        val_dicts = _with_split(validation, "validation")
        holdout_dicts = _with_split(holdout, "holdout")

        answer_key_dir = output_dir / "answer_keys"
        answer_key_dir.mkdir(parents=True, exist_ok=True)
        tune_dir = output_dir / "tune"
        val_dir = output_dir / "validation"
        holdout_dir = output_dir / "holdout"
        for d in [tune_dir, val_dir, holdout_dir]:
            d.mkdir(exist_ok=True)

        def _write_split(dicts: list[dict], path: Path) -> None:
            if not dicts:
                return
            fieldnames = list(dicts[0].keys())
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(dicts)

        _write_split(tune_dicts, tune_dir / "labels.csv")
        _write_split(val_dicts, val_dir / "labels.csv")
        _write_split(holdout_dicts, answer_key_dir / "answer_key_holdout.csv")

        # Also write non-label versions for pipeline use
        def _strip_label(dicts: list[dict]) -> list[dict]:
            return [{k: v for k, v in d.items() if k != "true_category"} for d in dicts]

        _write_split(_strip_label(tune_dicts), tune_dir / "records.csv")
        _write_split(_strip_label(val_dicts), val_dir / "records.csv")
        _write_split(_strip_label(holdout_dicts), holdout_dir / "records.csv")

        logger.info(
            "Ground truth split | tune=%d | validation=%d | holdout=%d",
            len(tune), len(validation), len(holdout),
        )
        return {
            "tune_labels": tune_dir / "labels.csv",
            "val_labels": val_dir / "labels.csv",
            "holdout_answer_key": answer_key_dir / "answer_key_holdout.csv",
            "tune_records": tune_dir / "records.csv",
            "val_records": val_dir / "records.csv",
            "holdout_records": holdout_dir / "records.csv",
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(seed: int = RANDOM_SEED, output_dir: Optional[Path] = None) -> dict:
    """Generate synthetic data and return metadata dict."""
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "data"

    gen = SyntheticDataGenerator(seed=seed)
    datasets = gen.generate()
    paths = gen.save(datasets, output_dir)
    split_paths = gen.split_ground_truth(datasets["ground_truth"], output_dir, seed=seed)
    paths.update(split_paths)

    # Verify distribution
    gt = datasets["ground_truth"]
    from collections import Counter
    dist = Counter(r.true_category for r in gt)
    logger.info("Ground truth distribution: %s", dict(dist))

    # Verify no label leakage in text fields
    label_keywords = ["split_settlement", "refund_misattribution", "fee_tier", "near_duplicate"]
    leakage_found = False
    for row in datasets["payments"] + datasets["ledger"] + datasets["bank"]:
        for field_val in vars(row).values():
            if isinstance(field_val, str):
                for kw in label_keywords:
                    if kw.lower() in field_val.lower():
                        logger.error("LABEL LEAKAGE DETECTED: '%s' in %s", kw, field_val)
                        leakage_found = True
    if not leakage_found:
        logger.info("Label leakage check: PASSED")

    return {
        "paths": paths,
        "distribution": dict(dist),
        "total_ground_truth": len(gt),
        "label_leakage": leakage_found,
        "seed": seed,
    }


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else RANDOM_SEED
    result = main(seed=seed)
    print(json.dumps(result, indent=2, default=str))
