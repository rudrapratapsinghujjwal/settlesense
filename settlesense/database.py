"""
SettleSense — SQLite Database Layer
Schema creation, inserts, and queries.
All human UI actions update SQLite.
"""

from __future__ import annotations
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS source_records (
    id          TEXT PRIMARY KEY,
    source      TEXT NOT NULL,
    record_id   TEXT NOT NULL,
    raw_json    TEXT NOT NULL,           -- Full original record as JSON
    is_synthetic INTEGER NOT NULL DEFAULT 1,
    ingested_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS normalized_transactions (
    txn_id          TEXT PRIMARY KEY,
    source          TEXT NOT NULL,
    source_record_id TEXT NOT NULL,
    amount          REAL NOT NULL,       -- In base currency units (paise)
    currency        TEXT NOT NULL DEFAULT 'INR',
    transaction_date TEXT NOT NULL,      -- ISO8601
    description     TEXT NOT NULL,
    reference_ids   TEXT NOT NULL,       -- JSON array
    fee             REAL,
    tax             REAL,
    settlement_id   TEXT,
    order_id        TEXT,
    payment_id      TEXT,
    method          TEXT,
    status          TEXT,
    is_synthetic    INTEGER NOT NULL DEFAULT 1,
    raw_fields      TEXT NOT NULL        -- JSON
);

CREATE TABLE IF NOT EXISTS ground_truth_labels (
    record_id       TEXT PRIMARY KEY,
    true_category   TEXT NOT NULL,
    split           TEXT NOT NULL        -- 'tune' | 'validation' | 'holdout'
);

CREATE TABLE IF NOT EXISTS candidate_matches (
    id                  TEXT PRIMARY KEY,
    exception_record_id TEXT NOT NULL,
    candidate_id        TEXT NOT NULL,
    match_score         REAL NOT NULL,
    amount_proximity    REAL NOT NULL,
    date_proximity_days REAL NOT NULL,
    description_sim     REAL NOT NULL,
    shared_refs         TEXT NOT NULL,   -- JSON array
    created_at          TEXT NOT NULL,
    FOREIGN KEY (exception_record_id) REFERENCES normalized_transactions(txn_id)
);

CREATE TABLE IF NOT EXISTS ai_decisions (
    decision_id             TEXT PRIMARY KEY,
    record_id               TEXT NOT NULL,
    candidate_category      TEXT NOT NULL,
    proposed_linked_ids     TEXT NOT NULL,  -- JSON array
    evidence_used           TEXT NOT NULL,  -- JSON array
    raw_model_signal        REAL NOT NULL,
    recommended_action      TEXT NOT NULL,
    reasoning_summary       TEXT NOT NULL,
    is_valid                INTEGER NOT NULL DEFAULT 1,
    validation_error        TEXT,
    hallucinated_evidence   INTEGER NOT NULL DEFAULT 0,
    created_at              TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS confidence_signals (
    record_id               TEXT PRIMARY KEY,
    candidate_margin        REAL NOT NULL,
    rule_agreement          REAL NOT NULL,
    evidence_completeness   REAL NOT NULL,
    raw_model_signal        REAL NOT NULL,
    calibrated_confidence   REAL NOT NULL,
    threshold_used          REAL NOT NULL,
    decision_status         TEXT NOT NULL,  -- auto_resolved | human_review
    created_at              TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pipeline_decisions (
    decision_id             TEXT PRIMARY KEY,
    record_id               TEXT NOT NULL,
    category                TEXT NOT NULL,
    status                  TEXT NOT NULL,
    calibrated_confidence   REAL NOT NULL,
    threshold_used          REAL NOT NULL,
    timestamp               TEXT NOT NULL,
    pipeline_stage          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS human_overrides (
    override_id         TEXT PRIMARY KEY,
    decision_id         TEXT NOT NULL,
    record_id           TEXT NOT NULL,
    action              TEXT NOT NULL,  -- approved | rejected | overridden | escalated
    override_category   TEXT,
    notes               TEXT,
    timestamp           TEXT NOT NULL,
    FOREIGN KEY (decision_id) REFERENCES pipeline_decisions(decision_id)
);

CREATE TABLE IF NOT EXISTS evaluation_runs (
    run_id                      TEXT PRIMARY KEY,
    dataset_split               TEXT NOT NULL,
    model_name                  TEXT NOT NULL,
    prompt_version              TEXT NOT NULL,
    threshold                   REAL NOT NULL,
    random_seed                 INTEGER NOT NULL,
    timestamp                   TEXT NOT NULL,
    total_records               INTEGER NOT NULL,
    clean_records               INTEGER NOT NULL,
    exception_records           INTEGER NOT NULL,
    auto_resolved               INTEGER NOT NULL,
    human_review                INTEGER NOT NULL,
    correctly_classified        INTEGER NOT NULL,
    false_auto_resolved         INTEGER NOT NULL,
    automation_rate             REAL NOT NULL,
    escalation_rate             REAL NOT NULL,
    false_auto_resolve_rate     REAL NOT NULL,
    overall_accuracy            REAL NOT NULL,
    throughput_records_per_sec  REAL NOT NULL,
    avg_latency_ms              REAL NOT NULL,
    per_category_json           TEXT NOT NULL,  -- JSON
    confusion_matrix_json       TEXT NOT NULL,  -- JSON
    confusion_labels_json       TEXT NOT NULL,  -- JSON
    notes                       TEXT
);

CREATE TABLE IF NOT EXISTS audit_log (
    log_id      TEXT PRIMARY KEY,
    timestamp   TEXT NOT NULL,
    record_id   TEXT,
    stage       TEXT NOT NULL,
    action      TEXT NOT NULL,
    detail      TEXT NOT NULL,   -- JSON or plain description
    level       TEXT NOT NULL DEFAULT 'INFO'
);

CREATE INDEX IF NOT EXISTS idx_normalized_source ON normalized_transactions(source);
CREATE INDEX IF NOT EXISTS idx_normalized_date ON normalized_transactions(transaction_date);
CREATE INDEX IF NOT EXISTS idx_candidates_exception ON candidate_matches(exception_record_id);
CREATE INDEX IF NOT EXISTS idx_decisions_record ON pipeline_decisions(record_id);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_record ON audit_log(record_id);
"""


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

def get_connection(db_path: Path) -> sqlite3.Connection:
    """Open a SQLite connection with WAL mode and row factory."""
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database(db_path: Path) -> None:
    """Create all tables if they don't exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA_SQL)
    logger.info("Database initialized at %s", db_path)


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

def audit(
    conn: sqlite3.Connection,
    stage: str,
    action: str,
    detail: str | dict,
    record_id: Optional[str] = None,
    level: str = "INFO",
) -> None:
    """Insert one audit log entry. Never logs secrets."""
    import uuid
    detail_str = json.dumps(detail) if isinstance(detail, dict) else str(detail)
    conn.execute(
        """INSERT INTO audit_log (log_id, timestamp, record_id, stage, action, detail, level)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (str(uuid.uuid4()), datetime.utcnow().isoformat(), record_id,
         stage, action, detail_str, level),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

def insert_ground_truth(
    conn: sqlite3.Connection,
    records: list[dict],
) -> None:
    """Bulk-insert ground truth labels. Used only by data generator."""
    conn.executemany(
        """INSERT OR REPLACE INTO ground_truth_labels (record_id, true_category, split)
           VALUES (:record_id, :true_category, :split)""",
        records,
    )
    conn.commit()


def get_ground_truth(
    conn: sqlite3.Connection,
    split: Optional[str] = None,
) -> list[sqlite3.Row]:
    """Fetch ground truth labels for a given split."""
    if split:
        return conn.execute(
            "SELECT * FROM ground_truth_labels WHERE split = ?", (split,)
        ).fetchall()
    return conn.execute("SELECT * FROM ground_truth_labels").fetchall()


# ---------------------------------------------------------------------------
# Human override actions
# ---------------------------------------------------------------------------

def insert_human_override(
    conn: sqlite3.Connection,
    decision_id: str,
    record_id: str,
    action: str,
    override_category: Optional[str] = None,
    notes: Optional[str] = None,
) -> str:
    """Record a human approve/reject/override/escalate action."""
    import uuid
    override_id = str(uuid.uuid4())
    conn.execute(
        """INSERT INTO human_overrides
           (override_id, decision_id, record_id, action, override_category, notes, timestamp)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (override_id, decision_id, record_id, action,
         override_category, notes, datetime.utcnow().isoformat()),
    )
    # Update pipeline decision status
    status_map = {
        "approved": "human_approved",
        "rejected": "human_rejected",
        "overridden": "human_overridden",
        "escalated": "escalated",
    }
    new_status = status_map.get(action, action)
    conn.execute(
        "UPDATE pipeline_decisions SET status = ? WHERE decision_id = ?",
        (new_status, decision_id),
    )
    conn.commit()
    audit(conn, "human_review", action,
          {"decision_id": decision_id, "override_category": override_category},
          record_id=record_id)
    return override_id


# ---------------------------------------------------------------------------
# Evaluation run storage
# ---------------------------------------------------------------------------

def save_evaluation_run(conn: sqlite3.Connection, result: dict) -> None:
    """Persist a complete evaluation result. Called only by evaluation code."""
    conn.execute(
        """INSERT OR REPLACE INTO evaluation_runs VALUES (
            :run_id, :dataset_split, :model_name, :prompt_version, :threshold,
            :random_seed, :timestamp, :total_records, :clean_records, :exception_records,
            :auto_resolved, :human_review, :correctly_classified, :false_auto_resolved,
            :automation_rate, :escalation_rate, :false_auto_resolve_rate, :overall_accuracy,
            :throughput_records_per_sec, :avg_latency_ms,
            :per_category_json, :confusion_matrix_json, :confusion_labels_json, :notes
        )""",
        result,
    )
    conn.commit()


def get_latest_evaluation(conn: sqlite3.Connection, split: str = "holdout") -> Optional[sqlite3.Row]:
    """Fetch the most recent evaluation run for a given split."""
    return conn.execute(
        "SELECT * FROM evaluation_runs WHERE dataset_split = ? ORDER BY timestamp DESC LIMIT 1",
        (split,),
    ).fetchone()
