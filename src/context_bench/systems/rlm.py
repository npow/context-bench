"""RLM-based memory system using smart retrieval + LLM answering.

Stores conversation data in LanceDB (embeddings) and DuckDB (structured).
On query, uses multi-strategy retrieval (semantic + keyword + entity) to
gather relevant context, then a single LLM call to answer concisely.

Requires: pip install lancedb duckdb sentence-transformers
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import time
import traceback
import urllib.error
import urllib.request
from typing import Any

from context_bench.memory_types import (
    ConversationTurn,
    Declaration,
    DocumentChunk,
    IngestResult,
    Item,
    PlatformEvent,
    QueryResult,
)

try:
    import lancedb
except ImportError:
    lancedb = None  # type: ignore[assignment]

try:
    import duckdb
except ImportError:
    duckdb = None  # type: ignore[assignment]

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore[assignment]


def _extract_entities(text: str) -> list[tuple[str, str, str]]:
    """Simple regex-based entity extraction."""
    entities: list[tuple[str, str, str]] = []
    for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", text):
        snippet = text[max(0, m.start() - 30):m.end() + 30]
        entities.append((m.group(0), "name", snippet))
    for m in re.finditer(
        r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|"
        r"(?:January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+\d{1,2},?\s*\d{4})\b",
        text,
    ):
        snippet = text[max(0, m.start() - 30):m.end() + 30]
        entities.append((m.group(0), "date", snippet))
    return entities


def _extract_keywords(question: str) -> list[str]:
    """Extract meaningful keywords from a question for SQL LIKE search."""
    # Remove common question words
    stop_words = {
        "what", "when", "where", "who", "why", "how", "which", "would",
        "could", "should", "does", "did", "will", "can", "is", "are",
        "was", "were", "has", "have", "had", "be", "been", "being",
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "as", "if", "that", "this",
        "it", "its", "not", "no", "do", "so", "up", "out", "about",
        "than", "then", "also", "just", "more", "most", "still",
        "likely", "probably", "considered", "she", "he", "her", "his",
        "they", "their", "s", "t", "re", "ve", "ll", "d",
    }
    # Split on non-alphanumeric and filter
    words = re.findall(r"[a-zA-Z]+", question.lower())
    keywords = [w for w in words if w not in stop_words and len(w) > 2]
    return keywords


class RLMSystem:
    """Memory system using smart retrieval + LLM answering.

    Stores conversation data in LanceDB (embeddings) and DuckDB (structured).
    Uses multi-strategy retrieval to gather relevant context, then
    answers with a single LLM call.
    """

    def __init__(
        self,
        base_url: str,
        model: str = "claude-haiku-4-5-20251001",
        embedding_model: str = "all-MiniLM-L6-v2",
        max_iterations: int = 8,
        max_llm_calls: int = 10,
        api_key: str | None = None,
        timeout: float = 120.0,
    ):
        if lancedb is None:
            raise ImportError("lancedb is required: pip install lancedb")
        if duckdb is None:
            raise ImportError("duckdb is required: pip install duckdb")
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is required: pip install sentence-transformers")

        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._timeout = timeout
        self._max_iterations = max_iterations
        self._max_llm_calls = max_llm_calls

        self._embedder = SentenceTransformer(embedding_model)
        self._tmpdir: str | None = None
        self._lance_db: Any = None
        self._lance_table: Any = None
        self._duck_conn: Any = None
        self._duck_path: str | None = None

    @property
    def name(self) -> str:
        return "rlm"

    def reset(self) -> None:
        if self._duck_conn is not None:
            try:
                self._duck_conn.close()
            except Exception:
                pass
            self._duck_conn = None
        self._lance_db = None
        self._lance_table = None
        self._duck_path = None
        if self._tmpdir and os.path.exists(self._tmpdir):
            shutil.rmtree(self._tmpdir, ignore_errors=True)
        self._tmpdir = None

    def ingest(self, items: list[Item]) -> IngestResult:
        t0 = time.perf_counter()
        self.reset()
        self._tmpdir = tempfile.mkdtemp(prefix="rlm_")

        lance_path = os.path.join(self._tmpdir, "lance")
        self._lance_db = lancedb.connect(lance_path)

        self._duck_path = os.path.join(self._tmpdir, "meta.duckdb")
        conn = duckdb.connect(self._duck_path)
        conn.execute(
            "CREATE TABLE turns("
            "  turn_id INTEGER, content TEXT, role TEXT, "
            "  speaker TEXT, timestamp TEXT, session_id TEXT)"
        )
        conn.execute(
            "CREATE TABLE entities("
            "  entity TEXT, entity_type TEXT, "
            "  turn_id INTEGER, context_snippet TEXT)"
        )
        conn.execute(
            "CREATE TABLE declarations("
            "  key TEXT, value TEXT, "
            "  source_turn_id TEXT, timestamp TEXT)"
        )

        lance_rows: list[dict[str, Any]] = []
        turn_id = 0

        for item in items:
            if isinstance(item, ConversationTurn):
                content = item.content
                ts = item.timestamp or ""
                sid = item.session_id or ""
                speaker = item.speaker or ""
                role = item.role

                vec = self._embedder.encode(content).tolist()
                lance_rows.append({
                    "text": content,
                    "vector": vec,
                    "timestamp": ts,
                    "session_id": sid,
                    "speaker": speaker,
                    "item_type": "conversation_turn",
                })
                conn.execute(
                    "INSERT INTO turns VALUES (?, ?, ?, ?, ?, ?)",
                    [turn_id, content, role, speaker, ts, sid],
                )
                for entity, etype, snippet in _extract_entities(content):
                    conn.execute(
                        "INSERT INTO entities VALUES (?, ?, ?, ?)",
                        [entity, etype, turn_id, snippet],
                    )
                turn_id += 1

            elif isinstance(item, DocumentChunk):
                vec = self._embedder.encode(item.content).tolist()
                lance_rows.append({
                    "text": item.content,
                    "vector": vec,
                    "timestamp": "",
                    "session_id": "",
                    "speaker": item.source or "",
                    "item_type": "document_chunk",
                })

            elif isinstance(item, PlatformEvent):
                vec = self._embedder.encode(item.content).tolist()
                lance_rows.append({
                    "text": item.content,
                    "vector": vec,
                    "timestamp": item.timestamp,
                    "session_id": "",
                    "speaker": item.author or "",
                    "item_type": "platform_event",
                })

            elif isinstance(item, Declaration):
                vec = self._embedder.encode(f"{item.key}: {item.value}").tolist()
                lance_rows.append({
                    "text": f"{item.key}: {item.value}",
                    "vector": vec,
                    "timestamp": "",
                    "session_id": "",
                    "speaker": "",
                    "item_type": "declaration",
                })
                conn.execute(
                    "INSERT INTO declarations VALUES (?, ?, ?, ?)",
                    [item.key, item.value, item.source_turn_id or "", ""],
                )

        if lance_rows:
            self._lance_table = self._lance_db.create_table(
                "semantic", data=lance_rows, mode="overwrite"
            )
        else:
            import pyarrow as pa
            schema = pa.schema([
                ("text", pa.string()),
                ("vector", pa.list_(pa.float32(), list_size=self._embedder.get_sentence_embedding_dimension())),
                ("timestamp", pa.string()),
                ("session_id", pa.string()),
                ("speaker", pa.string()),
                ("item_type", pa.string()),
            ])
            self._lance_table = self._lance_db.create_table(
                "semantic", schema=schema, mode="overwrite"
            )

        conn.close()
        self._duck_conn = None

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return IngestResult(
            num_items=len(items),
            latency_ms=elapsed_ms,
            details={"lance_rows": len(lance_rows), "turns": turn_id},
        )

    def query(self, question: str, budget: int | None = None) -> QueryResult:
        t0 = time.perf_counter()
        try:
            result = self._query_smart(question)
        except Exception as exc:
            print(f"[RLM] Query failed: {exc}", flush=True)
            traceback.print_exc()
            result = QueryResult(
                answer="",
                total_latency_ms=0,
                context_tokens=0,
                details={"method": "error", "error": str(exc)},
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        result.total_latency_ms = elapsed_ms
        return result

    # ------------------------------------------------------------------
    # Smart retrieval + answer
    # ------------------------------------------------------------------

    def _query_smart(self, question: str) -> QueryResult:
        """Multi-strategy retrieval followed by a single LLM answer call."""
        db_ro = duckdb.connect(self._duck_path, read_only=True)

        try:
            context_parts = self._retrieve(question, db_ro)
        finally:
            db_ro.close()

        # Deduplicate while preserving order
        seen = set()
        unique_parts = []
        for part in context_parts:
            key = part.strip()[:200]
            if key not in seen:
                seen.add(key)
                unique_parts.append(part)

        context = "\n---\n".join(unique_parts)
        context_tokens = len(context.split())

        # Truncate if too long (keep most relevant first)
        max_context_chars = 15000
        if len(context) > max_context_chars:
            context = context[:max_context_chars] + "\n[...truncated...]"

        answer = self._chat([
            {
                "role": "system",
                "content": (
                    "You answer questions about conversations. "
                    "Rules:\n"
                    "1. Answer in ONE short sentence or phrase (under 15 words)\n"
                    "2. No explanations, no elaboration, no dashes followed by reasoning\n"
                    "3. Just the direct answer, nothing else\n"
                    "4. Examples: 'Psychology, counseling certification' / 'Likely no' / "
                    "'Yes, since she collects classic children's books' / 'Liberal' / "
                    "'Thoughtful, authentic, driven' / 'National park; she likes the outdoors'"
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}\n\nShort answer:",
            },
        ])

        return QueryResult(
            answer=answer,
            total_latency_ms=0,
            context_tokens=context_tokens,
            details={
                "method": "smart_retrieval",
                "num_context_parts": len(unique_parts),
                "fallback_triggered": False,
            },
        )

    def _format_turn(self, content: str, speaker: str, ts: str) -> str:
        """Format a turn with optional timestamp and speaker prefix."""
        prefix = ""
        if ts:
            prefix += f"[{ts}] "
        if speaker:
            prefix += f"{speaker}: "
        return prefix + content

    def _get_context_window(self, db_ro: Any, turn_ids: set[int], window: int = 2) -> list[str]:
        """Get surrounding turns for a set of turn IDs (temporal context)."""
        if not turn_ids:
            return []
        # Build ranges around each turn
        expanded = set()
        for tid in turn_ids:
            for offset in range(-window, window + 1):
                expanded.add(tid + offset)
        expanded = {t for t in expanded if t >= 0}

        if not expanded:
            return []

        placeholders = ",".join(str(t) for t in sorted(expanded))
        try:
            rows = db_ro.execute(
                f"SELECT turn_id, content, speaker, timestamp FROM turns "
                f"WHERE turn_id IN ({placeholders}) "
                f"ORDER BY turn_id"
            ).fetchall()
            return [self._format_turn(content, speaker, ts) for _, content, speaker, ts in rows]
        except Exception:
            return []

    def _retrieve(self, question: str, db_ro: Any) -> list[str]:
        """Multi-strategy retrieval combining semantic, keyword, and entity search."""
        context_parts: list[str] = []
        relevant_turn_ids: set[int] = set()

        # Strategy 1: Semantic (vector) search via LanceDB
        try:
            query_vec = self._embedder.encode(question).tolist()
            results = self._lance_table.search(query_vec).limit(20).to_list()
            for r in results:
                context_parts.append(self._format_turn(
                    r.get("text", ""), r.get("speaker", ""), r.get("timestamp", "")
                ))
        except Exception as e:
            print(f"[RLM] Semantic search failed: {e}", flush=True)

        # Strategy 2: Keyword search in DuckDB
        keywords = _extract_keywords(question)
        if keywords:
            conditions = [f"LOWER(content) LIKE '%{kw}%'" for kw in keywords[:8]]
            if conditions:
                where = " OR ".join(conditions)
                try:
                    rows = db_ro.execute(
                        f"SELECT turn_id, content, speaker, timestamp FROM turns "
                        f"WHERE {where} "
                        f"ORDER BY turn_id LIMIT 30"
                    ).fetchall()
                    for tid, content, speaker, ts in rows:
                        relevant_turn_ids.add(tid)
                        context_parts.append(self._format_turn(content, speaker, ts))
                except Exception as e:
                    print(f"[RLM] Keyword search failed: {e}", flush=True)

        # Strategy 3: Entity-based search
        question_words = set(re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", question))
        if question_words:
            for entity_name in question_words:
                try:
                    rows = db_ro.execute(
                        "SELECT t.turn_id, t.content, t.speaker, t.timestamp "
                        "FROM turns t JOIN entities e ON t.turn_id = e.turn_id "
                        "WHERE e.entity = ? "
                        "ORDER BY t.turn_id LIMIT 10",
                        [entity_name],
                    ).fetchall()
                    for tid, content, speaker, ts in rows:
                        relevant_turn_ids.add(tid)
                        context_parts.append(self._format_turn(content, speaker, ts))
                except Exception:
                    pass

        # Strategy 4: Declarations
        try:
            decls = db_ro.execute("SELECT key, value FROM declarations").fetchall()
            for key, value in decls:
                context_parts.append(f"[Declaration] {key}: {value}")
        except Exception:
            pass

        # Strategy 5: If question mentions a specific speaker, get their key turns
        all_speakers = []
        try:
            all_speakers = [row[0] for row in db_ro.execute(
                "SELECT DISTINCT speaker FROM turns WHERE speaker IS NOT NULL AND speaker != ''"
            ).fetchall()]
        except Exception:
            pass

        for speaker_name in all_speakers:
            if speaker_name.lower() in question.lower():
                try:
                    rows = db_ro.execute(
                        "SELECT turn_id, content, timestamp FROM turns "
                        "WHERE speaker = ? ORDER BY turn_id LIMIT 20",
                        [speaker_name],
                    ).fetchall()
                    for tid, content, ts in rows:
                        relevant_turn_ids.add(tid)
                        context_parts.append(self._format_turn(content, speaker_name, ts))
                except Exception:
                    pass

        # Strategy 6: Context window - grab surrounding turns for temporal context
        if relevant_turn_ids:
            window_parts = self._get_context_window(db_ro, relevant_turn_ids, window=2)
            context_parts.extend(window_parts)

        return context_parts

    # ------------------------------------------------------------------
    # HTTP chat
    # ------------------------------------------------------------------

    def _chat(self, messages: list[dict[str, Any]]) -> str:
        url = f"{self._base_url}/v1/chat/completions"
        body = json.dumps({"model": self._model, "messages": messages}).encode()
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        for attempt in range(3):
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=300) as resp:
                    data = json.loads(resp.read().decode())
                    return data["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"HTTP {e.code}: {e.reason}") from e
            except urllib.error.URLError as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"Connection error: {e.reason}") from e

        raise RuntimeError("Failed after 3 retries")
