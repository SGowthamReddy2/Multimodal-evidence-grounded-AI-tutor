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
import re
import sqlite3
import uuid
from contextlib import contextmanager
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

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

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
            # Ephemeral Memory: Chat Sessions and Messages
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_id TEXT PRIMARY KEY,
                    learner_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_session_learner ON chat_sessions(learner_id);"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    message_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    learner_id TEXT NOT NULL,
                    role TEXT CHECK(role IN ('user', 'assistant', 'system')),
                    content TEXT NOT NULL,
                    intent TEXT DEFAULT 'academic',
                    metadata_json TEXT DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id)
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_msg_session ON chat_messages(session_id);"
            )
            # Archival Memory: Long-Term Unbounded Storage (MemGPT / Letta Tier 3)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS archival_memory (
                    archival_id TEXT PRIMARY KEY,
                    learner_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_archival_learner ON archival_memory(learner_id);"
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

    def get_active_profile(self, learner_id: str) -> Dict[str, Any]:
        """Fetch active, confirmed learner profile records with confidence decay.
        
        Provides backward-compatible single string keys ('goal', 'mastery', 'misconception', 'preference')
        while supporting full multi-item retention via plural list keys ('goals', 'masteries', 'misconceptions', 'preferences').
        """
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

        now = datetime.datetime.utcnow()
        goals = []
        masteries = []
        misconceptions = []
        preferences = []

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
                t = r["type"]
                val = r["value"]
                if t == "goal":
                    goals.append(val)
                elif t == "mastery":
                    masteries.append(val)
                elif t == "misconception":
                    misconceptions.append(val)
                elif t == "preference":
                    preferences.append(val)

        return {
            "goal": goals[0] if goals else "",
            "goals": goals,
            "mastery": masteries[0] if masteries else "",
            "masteries": masteries,
            "misconception": misconceptions[0] if misconceptions else "",
            "misconceptions": misconceptions,
            "preference": preferences[0] if preferences else "",
            "preferences": preferences,
        }

    def get_active_core_memory_blocks(self, learner_id: str) -> Dict[str, List[Dict[str, Any]]]:
        """Return structured Letta Core Memory Blocks with full metadata for the student.
        
        Blocks:
        - goals: Target deadlines, upcoming exams, study objectives
        - mastery: Concept proficiency levels (Novice, Developing, Proficient, Mastered)
        - misconceptions: Active false mental models (tracked until resolved)
        - preferences: Pedagogical learning modes (visual diagrams, code-first, etc.)
        """
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT memory_id, type, value, write_authority, confidence, evidence, provenance, created_at, updated_at, revision_state
                FROM memory_records
                WHERE learner_id = ? 
                  AND revision_state = 'current'
                ORDER BY created_at ASC;
                """,
                (learner_id,),
            ).fetchall()

        blocks: Dict[str, List[Dict[str, Any]]] = {
            "goals": [],
            "mastery": [],
            "misconceptions": [],
            "preferences": [],
        }

        block_key_map = {
            "goal": "goals",
            "mastery": "mastery",
            "misconception": "misconceptions",
            "preference": "preferences",
        }

        for r in rows:
            target_key = block_key_map.get(r["type"], "goals")
            blocks[target_key].append(dict(r))

        return blocks

    def core_memory_append(
        self,
        learner_id: str,
        block: str,
        value: str,
        authority: str = "student_authored",
        confidence: float = 0.85,
        evidence: str = "dialogue_input",
        provenance: str = "dialogue_turn",
    ) -> str:
        """MemGPT/Letta core_memory_append: Appends an item to a Core Memory Block.
        
        Args:
            learner_id: Identifier of student.
            block: 'goals' | 'mastery' | 'misconceptions' | 'preferences' (or singular).
            value: Item content.
            authority: 'student_authored' (direct declaration) or 'tutor_proposed'.
            confidence: Confidence score 0.0 to 1.0.
            evidence: Grounding quote or rationale.
            provenance: Extraction origin.
        """
        block_norm = block.lower().rstrip("s")
        if block_norm == "masterie":
            block_norm = "mastery"
        if block_norm not in ("goal", "mastery", "misconception", "preference"):
            block_norm = "goal"

        clean_val = self.sanitize_value(value)
        memory_id = f"mem-{uuid.uuid4().hex[:8]}"
        now_str = datetime.datetime.utcnow().isoformat()

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO memory_records (
                    memory_id, learner_id, type, value, provenance,
                    write_authority, confidence, evidence, revision_state,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'current', ?, ?);
                """,
                (memory_id, learner_id, block_norm, clean_val, provenance, authority, confidence, evidence, now_str, now_str),
            )
            conn.commit()

        logger.info("Core memory append [%s] %s: '%s' (%s)", block_norm, memory_id, clean_val, authority)
        return memory_id

    def core_memory_replace(
        self,
        memory_id: str,
        new_value: str,
        reason: str = "",
    ) -> bool:
        """MemGPT/Letta core_memory_replace: Updates an existing core memory item with audit history."""
        clean_val = self.sanitize_value(new_value)
        now_str = datetime.datetime.utcnow().isoformat()

        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT value, revision_history FROM memory_records WHERE memory_id = ? AND revision_state = 'current'",
                (memory_id,),
            ).fetchone()
            if not row:
                logger.warning("core_memory_replace failed: memory_id %s not found or not current", memory_id)
                return False

            old_val = row["value"]
            history = json.loads(row["revision_history"] or "[]")
            history.append({"old_value": old_val, "reason": reason, "updated_at": now_str})

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
            success = cursor.rowcount > 0

        if success:
            logger.info("Core memory replace %s: '%s' -> '%s' (Reason: %s)", memory_id, old_val, clean_val, reason)
        return success

    def core_memory_delete(
        self,
        memory_id: str,
        reason: str = "",
    ) -> bool:
        """MemGPT/Letta core_memory_delete: Soft-deletes a core memory item."""
        now_str = datetime.datetime.utcnow().isoformat()
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT revision_history FROM memory_records WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
            if not row:
                return False

            history = json.loads(row["revision_history"] or "[]")
            history.append({"action": "deleted", "reason": reason, "timestamp": now_str})

            cursor = conn.execute(
                """
                UPDATE memory_records
                SET revision_state = 'deleted',
                    revision_history = ?,
                    updated_at = ?
                WHERE memory_id = ?;
                """,
                (json.dumps(history), now_str, memory_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def core_memory_resolve(
        self,
        memory_id: str,
        resolution_note: str = "",
    ) -> bool:
        """Marks a misconception as resolved or an exam goal as achieved (unifies with delete)."""
        clean_note = resolution_note or "Resolved in pedagogical dialogue"
        return self.core_memory_delete(memory_id, reason=clean_note)

    def clear_all_learner_memory(self, learner_id: str, reason: str = "Cleared by student") -> int:
        """Soft-deletes all active core memory records for a student."""
        now_str = datetime.datetime.utcnow().isoformat()
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT memory_id, revision_history FROM memory_records WHERE learner_id = ? AND revision_state = 'current'",
                (learner_id,),
            ).fetchall()
            count = 0
            for r in rows:
                history = json.loads(r["revision_history"] or "[]")
                history.append({"action": "deleted", "reason": reason, "timestamp": now_str})
                conn.execute(
                    """
                    UPDATE memory_records
                    SET revision_state = 'deleted',
                        revision_history = ?,
                        updated_at = ?
                    WHERE memory_id = ?;
                    """,
                    (json.dumps(history), now_str, r["memory_id"]),
                )
                count += 1
            conn.commit()
            logger.info("Cleared %d active core memories for learner %s", count, learner_id)
            return count

    def get_active_memory_choices(self, learner_id: str) -> List[Tuple[str, str]]:
        """Return human-readable choice tuples for UI dropdowns: [(label, memory_id), ...]."""
        blocks = self.get_active_core_memory_blocks(learner_id)
        choices: List[Tuple[str, str]] = []

        type_icons = {
            "goals": "🎯 [Goal]",
            "mastery": "📈 [Mastery]",
            "misconceptions": "⚠️ [Hurdle]",
            "preferences": "🎨 [Preference]",
        }

        for block_name in ("goals", "mastery", "misconceptions", "preferences"):
            items = blocks.get(block_name, [])
            icon = type_icons.get(block_name, "💡")
            for item in items:
                m_id = item["memory_id"]
                val = item.get("value", "").strip()
                display_val = val if len(val) <= 60 else val[:57] + "..."
                label = f"{icon} {display_val}"
                choices.append((label, m_id))

        return choices

    def get_core_memory_prompt_block(self, learner_id: str) -> str:
        """Generate formatted Core Memory markdown block for LLM prompt injection."""
        blocks = self.get_active_core_memory_blocks(learner_id)
        lines = ["### Learner Core Memory (Active Working Context):"]

        # 1. Goals
        active_goals = [b["value"] for b in blocks["goals"]]
        if active_goals:
            lines.append("- 🎯 **Goals & Deadlines**:")
            for g in active_goals:
                lines.append(f"  * {g}")
        else:
            lines.append("- 🎯 **Goals & Deadlines**: None specified.")

        # 2. Mastery
        active_mastery = [b["value"] for b in blocks["mastery"]]
        if active_mastery:
            lines.append("- 📈 **Concept Mastery Levels**:")
            for m in active_mastery:
                lines.append(f"  * {m}")
        else:
            lines.append("- 📈 **Concept Mastery Levels**: Baseline MIT 6.006 student.")

        # 3. Misconceptions
        active_misconceptions = [b["value"] for b in blocks["misconceptions"]]
        if active_misconceptions:
            lines.append("- ⚠️ **Active Misconceptions & Hurdles**:")
            for misc in active_misconceptions:
                lines.append(f"  * {misc}")

        # 4. Preferences
        active_prefs = [b["value"] for b in blocks["preferences"]]
        if active_prefs:
            lines.append("- 🎨 **Pedagogical Preferences**:")
            for p in active_prefs:
                lines.append(f"  * {p}")

        return "\n".join(lines)

    def archival_memory_insert(
        self,
        learner_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """MemGPT/Letta archival_memory_insert: Persists long-term observations or study notes."""
        clean_content = self.sanitize_value(content)
        archival_id = f"arch-{uuid.uuid4().hex[:10]}"
        meta_str = json.dumps(metadata or {}, ensure_ascii=False)
        now_str = datetime.datetime.utcnow().isoformat()

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO archival_memory (archival_id, learner_id, content, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?);
                """,
                (archival_id, learner_id, clean_content, meta_str, now_str),
            )
            conn.commit()

        logger.info("Archival memory inserted %s for learner %s", archival_id, learner_id)
        return archival_id

    def archival_memory_search(
        self,
        learner_id: str,
        query: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """MemGPT/Letta archival_memory_search: Semantic/keyword search over longitudinal records."""
        tokens = [t.lower() for t in re.findall(r"\w+", query) if len(t) > 2]
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT archival_id, learner_id, content, metadata_json, created_at
                FROM archival_memory
                WHERE learner_id = ?
                ORDER BY created_at DESC
                LIMIT 50;
                """,
                (learner_id,),
            ).fetchall()

        results = []
        for r in rows:
            c = r["content"].lower()
            match_score = sum(1 for t in tokens if t in c)
            if not tokens or match_score > 0:
                meta = {}
                if r["metadata_json"]:
                    try:
                        meta = json.loads(r["metadata_json"])
                    except Exception:
                        pass
                results.append(
                    {
                        "archival_id": r["archival_id"],
                        "content": r["content"],
                        "metadata": meta,
                        "created_at": r["created_at"],
                        "score": match_score,
                    }
                )

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

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

    # ------------------------------------------------------------------
    # Ephemeral Memory: Session & Chat History Management
    # ------------------------------------------------------------------

    def create_session(self, learner_id: str, title: str = "New Session") -> str:
        """Create a new chat session for ephemeral dialogue isolation."""
        session_id = f"sess-{uuid.uuid4().hex[:10]}"
        now_str = datetime.datetime.utcnow().isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO chat_sessions (session_id, learner_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?);
                """,
                (session_id, learner_id, title, now_str, now_str),
            )
            conn.commit()
        logger.info("Created chat session %s for learner %s", session_id, learner_id)
        return session_id

    def get_or_create_session(self, learner_id: str, session_id: Optional[str] = None) -> str:
        """Retrieve existing session or create a new active session."""
        if session_id:
            with self._get_connection() as conn:
                row = conn.execute(
                    "SELECT session_id FROM chat_sessions WHERE session_id = ? AND learner_id = ?",
                    (session_id, learner_id),
                ).fetchone()
                if row:
                    return str(row["session_id"])

        # Fetch most recent session or create new
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT session_id FROM chat_sessions WHERE learner_id = ? ORDER BY updated_at DESC LIMIT 1",
                (learner_id,),
            ).fetchone()
            if row:
                return str(row["session_id"])

        return self.create_session(learner_id, title="Interactive Tutor Session")

    def add_chat_message(
        self,
        session_id: str,
        learner_id: str,
        role: str,
        content: str,
        intent: str = "academic",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Log a chat message to the ephemeral session buffer."""
        msg_id = f"msg-{uuid.uuid4().hex[:10]}"
        now_str = datetime.datetime.utcnow().isoformat()
        meta_str = json.dumps(metadata or {}, ensure_ascii=False)

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO chat_messages (message_id, session_id, learner_id, role, content, intent, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (msg_id, session_id, learner_id, role, content, intent, meta_str, now_str),
            )
            conn.execute(
                "UPDATE chat_sessions SET updated_at = ? WHERE session_id = ?",
                (now_str, session_id),
            )
            conn.commit()
        return msg_id

    def get_session_messages(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve the recent chronological dialogue turns for context assembly."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT message_id, session_id, learner_id, role, content, intent, metadata_json, created_at
                FROM chat_messages
                WHERE session_id = ?
                ORDER BY created_at ASC
                LIMIT ?;
                """,
                (session_id, limit),
            ).fetchall()

        messages = []
        for r in rows:
            meta = {}
            if r["metadata_json"]:
                try:
                    meta = json.loads(r["metadata_json"])
                except Exception:
                    pass
            messages.append(
                {
                    "message_id": r["message_id"],
                    "session_id": r["session_id"],
                    "learner_id": r["learner_id"],
                    "role": r["role"],
                    "content": r["content"],
                    "intent": r["intent"],
                    "metadata": meta,
                    "created_at": r["created_at"],
                }
            )
        return messages

    def list_sessions(self, learner_id: str) -> List[Dict[str, Any]]:
        """List all chat sessions for a learner."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT session_id, learner_id, title, created_at, updated_at
                FROM chat_sessions
                WHERE learner_id = ?
                ORDER BY updated_at DESC;
                """,
                (learner_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def update_session_title(self, session_id: str, new_title: str) -> bool:
        """Update the title of a chat session for easier discovery in UI dropdowns."""
        clean_title = new_title.strip()[:60]
        now_str = datetime.datetime.utcnow().isoformat()
        with self._get_connection() as conn:
            cursor = conn.execute(
                "UPDATE chat_sessions SET title = ?, updated_at = ? WHERE session_id = ?",
                (clean_title, now_str, session_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def delete_session(self, session_id: str) -> bool:
        """Delete a chat session and all its associated messages."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
            cursor = conn.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))
            conn.commit()
            return cursor.rowcount > 0



