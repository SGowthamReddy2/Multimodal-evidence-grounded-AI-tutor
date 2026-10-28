"""SQLite WAL Learner Memory Controller with Confirmation Gate and Student CRUD.

Fulfills Milestone M5: Selective Learner Memory:
1. Multi-tier profile tracking (goal, mastery, misconception, preference)
2. Confirmation Gate: proposed insights require student approval
3. Exponential confidence decay with 30-day half-life
4. Relational audit trails & prompt injection sanitization
"""

import datetime
import json
import logging
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.config import (
    MEMORY_CONFIDENCE_HALF_LIFE_DAYS,
    MEMORY_DB_PATH,
    MEMORY_INJECTION_PATTERNS,
    MEMORY_MAX_RECORD_LENGTH,
)

logger = logging.getLogger(__name__)


class LearnerMemoryController:
    """Manages learner profile storage, confirmation gates, and audit trails."""

    def __init__(self, db_path: str = MEMORY_DB_PATH) -> None:
        self.db_path = db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create tables and indices if they do not exist."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_records (
                    memory_id TEXT PRIMARY KEY,
                    learner_id TEXT NOT NULL,
                    type TEXT CHECK(type IN ('goal', 'mastery', 'misconception', 'preference')),
                    value TEXT NOT NULL,
                    provenance TEXT NOT NULL,
                    write_authority TEXT CHECK(write_authority IN ('tutor_proposed', 'student_confirmed', 'student_authored')),
                    confidence REAL DEFAULT 0.8 CHECK(confidence BETWEEN 0.0 AND 1.0),
                    evidence TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    revision_state TEXT DEFAULT 'current' CHECK(revision_state IN ('current', 'revised', 'deleted')),
                    revision_history TEXT DEFAULT '[]'
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_mem_learner ON memory_records(learner_id);"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS version_registry (
                    source_id TEXT PRIMARY KEY,
                    current_version TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.commit()
            logger.info("Learner memory database initialized at %s", self.db_path)

    # ------------------------------------------------------------------
    # Sanitization & Security
    # ------------------------------------------------------------------

    def sanitize_value(self, raw_value: str) -> str:
        """Sanitize learner memory string to prevent prompt injection."""
        val = raw_value.strip()
        for pattern in MEMORY_INJECTION_PATTERNS:
            if pattern.lower() in val.lower():
                logger.warning("Sanitizing injection attempt in memory value: '%s'", val)
                val = val.replace(pattern, "[FILTERED]")

        if len(val) > MEMORY_MAX_RECORD_LENGTH:
            val = val[:MEMORY_MAX_RECORD_LENGTH]
        return val

    # ------------------------------------------------------------------
    # Confirmation Gate & Memory Lifecycle
    # ------------------------------------------------------------------

    def propose_memory(
        self,
        learner_id: str,
        memory_type: str,
        value: str,
        provenance: str,
        evidence: str,
        confidence: float = 0.8,
    ) -> str:
        """Tutor proposes an insight about the student (requires confirmation)."""
        clean_val = self.sanitize_value(value)
        memory_id = f"mem-{uuid.uuid4().hex[:8]}"

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO memory_records (
                    memory_id, learner_id, type, value, provenance,
                    write_authority, confidence, evidence, revision_state
                ) VALUES (?, ?, ?, ?, ?, 'tutor_proposed', ?, ?, 'current');
                """,
                (memory_id, learner_id, memory_type, clean_val, provenance, confidence, evidence),
            )
            conn.commit()

        logger.info("Proposed memory %s (%s: '%s') for learner %s", memory_id, memory_type, clean_val, learner_id)
        return memory_id

    def confirm_memory(self, memory_id: str) -> bool:
        """Student confirms a proposed memory, elevating its authority."""
        now_str = datetime.datetime.utcnow().isoformat()
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE memory_records
                SET write_authority = 'student_confirmed',
                    updated_at = ?
                WHERE memory_id = ? AND revision_state = 'current';
                """,
                (now_str, memory_id),
            )
            conn.commit()
            success = cursor.rowcount > 0

        if success:
            logger.info("Confirmed memory %s.", memory_id)
        return success

    def add_student_memory(
        self,
        learner_id: str,
        memory_type: str,
        value: str,
    ) -> str:
        """Directly insert a student-authored memory record."""
        clean_val = self.sanitize_value(value)
        memory_id = f"mem-{uuid.uuid4().hex[:8]}"

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO memory_records (
                    memory_id, learner_id, type, value, provenance,
                    write_authority, confidence, evidence, revision_state
                ) VALUES (?, ?, ?, ?, 'ui_student_direct', 'student_authored', 1.0, 'user_input', 'current');
                """,
                (memory_id, learner_id, memory_type, clean_val),
            )
            conn.commit()

        logger.info("Student-authored memory created: %s", memory_id)
        return memory_id

    def get_active_profile(self, learner_id: str) -> Dict[str, str]:
        """Fetch active, confirmed learner profile records with confidence decay."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT memory_id, type, value, write_authority, confidence, updated_at
                FROM memory_records
                WHERE learner_id = ? 
                  AND revision_state = 'current'
                  AND write_authority IN ('student_confirmed', 'student_authored');
                """,
                (learner_id,),
            ).fetchall()

        profile = {}
        now = datetime.datetime.utcnow()

        for r in rows:
            # Compute confidence decay: C(t) = C0 * 2^(-delta_t / 30)
            try:
                up_dt = datetime.datetime.fromisoformat(r["updated_at"].replace("Z", ""))
                days_elapsed = (now - up_dt).total_seconds() / 86400.0
                decay = 2 ** (-days_elapsed / MEMORY_CONFIDENCE_HALF_LIFE_DAYS)
                effective_conf = r["confidence"] * decay
            except Exception:
                effective_conf = r["confidence"]

            if effective_conf >= 0.35:
                profile[r["type"]] = r["value"]

        return profile

    def list_all_records(self, learner_id: str) -> List[Dict[str, Any]]:
        """List all non-deleted records for the student governance UI table."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT memory_id, type, value, write_authority, confidence, created_at, updated_at
                FROM memory_records
                WHERE learner_id = ? AND revision_state != 'deleted'
                ORDER BY created_at DESC;
                """,
                (learner_id,),
            ).fetchall()

        return [dict(r) for r in rows]

    def update_memory_value(self, memory_id: str, new_value: str) -> bool:
        """Update an existing memory value with audit trail preservation."""
        clean_val = self.sanitize_value(new_value)
        now_str = datetime.datetime.utcnow().isoformat()

        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT value, revision_history FROM memory_records WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
            if not row:
                return False

            old_val = row["value"]
            history = json.loads(row["revision_history"] or "[]")
            history.append({"old_value": old_val, "updated_at": now_str})

            cursor = conn.execute(
                """
                UPDATE memory_records
                SET value = ?,
                    revision_history = ?,
                    updated_at = ?
                WHERE memory_id = ?;
                """,
                (clean_val, json.dumps(history), now_str, memory_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def delete_memory(self, memory_id: str) -> bool:
        """Soft-delete a memory record."""
        now_str = datetime.datetime.utcnow().isoformat()
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE memory_records
                SET revision_state = 'deleted',
                    updated_at = ?
                WHERE memory_id = ?;
                """,
                (now_str, memory_id),
            )
            conn.commit()
            return cursor.rowcount > 0

