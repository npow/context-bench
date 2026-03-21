"""RLM-based memory system using DSPy's Recursive Language Model.

Stores conversation data in LanceDB (embeddings) and DuckDB (structured).
On query, runs the RLM loop: the model writes Python code to explore
the stores iteratively until it assembles the right context.

Requires: pip install context-bench[dspy] lancedb duckdb sentence-transformers
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import time
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
    import dspy
except ImportError:
    dspy = None  # type: ignore[assignment]

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


_STORE_DESCRIPTION = """\
Available data stores:

1. `semantic` -- A LanceDB table with embedded conversation chunks.
   Columns: text (str), vector (embedding), timestamp (str), session_id (str), speaker (str), item_type (str)
   Usage: results = semantic.search("query text").limit(10).to_list()
   Filter: results = semantic.search("query").where("timestamp > '2024-01-01'").limit(10).to_list()

2. `db` -- A DuckDB connection (read-only) with structured metadata.
   Tables:
     turns(turn_id INTEGER, content TEXT, role TEXT, speaker TEXT, timestamp TEXT, session_id TEXT)
     entities(entity TEXT, entity_type TEXT, turn_id INTEGER, context_snippet TEXT)
     declarations(key TEXT, value TEXT, source_turn_id TEXT, timestamp TEXT)
   Usage: results = db.execute("SELECT * FROM turns WHERE content LIKE '%topic%'").fetchall()
"""


def _extract_entities(text: str) -> list[tuple[str, str, str]]:
    """Simple regex-based entity extraction.

    Returns list of (entity, entity_type, context_snippet).
    """
    entities: list[tuple[str, str, str]] = []

    # Names: capitalized words that look like proper nouns (2+ capitalized words)
    for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", text):
        snippet = text[max(0, m.start() - 30):m.end() + 30]
        entities.append((m.group(0), "name", snippet))

    # Dates: various date formats
    for m in re.finditer(
        r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|"
        r"(?:January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+\d{1,2},?\s*\d{4})\b",
        text,
    ):
        snippet = text[max(0, m.start() - 30):m.end() + 30]
        entities.append((m.group(0), "date", snippet))

    return entities


class RLMSystem:
    """Memory system using DSPy's RLM for iterative code-generated retrieval.

    Stores conversation data in LanceDB (embeddings) and DuckDB (structured).
    On query, runs the RLM loop: the model writes Python code to explore
    the stores iteratively until it assembles the right context.

    Requires: pip install context-bench[dspy] lancedb duckdb sentence-transformers
    """

    def __init__(
        self,
        base_url: str,
        model: str = "claude-haiku-4-5-20251001",
        embedding_model: str = "all-MiniLM-L6-v2",
        max_iterations: int = 10,
        max_llm_calls: int = 20,
        api_key: str | None = None,
        timeout: float = 120.0,
    ):
        if dspy is None:
            raise ImportError("dspy is required: pip install dspy")
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

    # ------------------------------------------------------------------
    # MemorySystem protocol
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clean up temporary storage."""
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

        # Set up temp directory
        self.reset()
        self._tmpdir = tempfile.mkdtemp(prefix="rlm_")

        # --- LanceDB setup ---
        lance_path = os.path.join(self._tmpdir, "lance")
        self._lance_db = lancedb.connect(lance_path)

        # --- DuckDB setup ---
        self._duck_path = os.path.join(self._tmpdir, "meta.duckdb")
        conn = duckdb.connect(self._duck_path)
        conn.execute(
            "CREATE TABLE turns("
            "  turn_id INTEGER, content TEXT, role TEXT, "
            "  speaker TEXT, timestamp TEXT, session_id TEXT"
            ")"
        )
        conn.execute(
            "CREATE TABLE entities("
            "  entity TEXT, entity_type TEXT, "
            "  turn_id INTEGER, context_snippet TEXT"
            ")"
        )
        conn.execute(
            "CREATE TABLE declarations("
            "  key TEXT, value TEXT, "
            "  source_turn_id TEXT, timestamp TEXT"
            ")"
        )

        # --- Process items ---
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

                # Entity extraction
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

        # Create LanceDB table
        if lance_rows:
            self._lance_table = self._lance_db.create_table(
                "semantic", data=lance_rows, mode="overwrite"
            )
        else:
            # Create empty table with schema
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
            details={
                "lance_rows": len(lance_rows),
                "turns": turn_id,
            },
        )

    def query(self, question: str, budget: int | None = None) -> QueryResult:
        t0 = time.perf_counter()

        try:
            result = self._query_rlm(question)
        except Exception as exc:
            # Fallback to simple semantic search
            result = self._query_fallback(question, fallback_reason=str(exc))

        elapsed_ms = (time.perf_counter() - t0) * 1000
        result.total_latency_ms = elapsed_ms
        return result

    # ------------------------------------------------------------------
    # RLM query
    # ------------------------------------------------------------------

    def _query_rlm(self, question: str) -> QueryResult:
        """Run the RLM loop against the stores."""
        # Configure dspy
        lm = dspy.LM(
            model=f"openai/{self._model}",
            api_base=self._base_url,
            api_key=self._api_key or "unused",
        )
        dspy.configure(lm=lm)

        # Open DuckDB read-only for the RLM execution
        db_ro = duckdb.connect(self._duck_path, read_only=True)

        try:
            # Build context describing the stores
            context = _STORE_DESCRIPTION

            # Add sample data so the RLM knows what's there
            try:
                sample_turns = db_ro.execute(
                    "SELECT * FROM turns LIMIT 5"
                ).fetchall()
                if sample_turns:
                    context += f"\nSample turns data:\n{sample_turns}\n"
            except Exception:
                pass

            try:
                sample_decl = db_ro.execute(
                    "SELECT * FROM declarations LIMIT 5"
                ).fetchall()
                if sample_decl:
                    context += f"\nSample declarations data:\n{sample_decl}\n"
            except Exception:
                pass

            try:
                entity_count = db_ro.execute(
                    "SELECT COUNT(*) FROM entities"
                ).fetchone()[0]
                context += f"\nEntities table has {entity_count} rows.\n"
            except Exception:
                pass

            # Create RLM module. We pass semantic and db as additional inputs.
            rlm = dspy.RLM(
                signature="context, question, semantic, db -> answer",
                max_iterations=self._max_iterations,
                max_llm_calls=self._max_llm_calls,
                verbose=True,
            )

            result = rlm(
                context=context,
                question=question,
                semantic=self._lance_table,
                db=db_ro,
            )

            answer = result.answer
            trajectory = result.trajectory if hasattr(result, "trajectory") else []
            iterations = len(trajectory) if trajectory else 0

            # Estimate context tokens from trajectory
            traj_text = json.dumps(trajectory) if trajectory else ""
            context_tokens = len(traj_text.split())

            return QueryResult(
                answer=answer,
                total_latency_ms=0,  # filled by caller
                context_tokens=context_tokens,
                details={
                    "method": "rlm",
                    "iterations": iterations,
                    "trajectory": trajectory,
                    "fallback_triggered": False,
                },
            )
        finally:
            db_ro.close()

    # ------------------------------------------------------------------
    # Fallback: simple semantic search + LLM
    # ------------------------------------------------------------------

    def _query_fallback(self, question: str, fallback_reason: str = "") -> QueryResult:
        """Fallback to simple semantic search when RLM fails."""
        if self._lance_table is None:
            return QueryResult(
                answer="No data ingested.",
                total_latency_ms=0,
                context_tokens=0,
                details={"method": "fallback", "fallback_triggered": True, "reason": fallback_reason},
            )

        # Semantic search
        try:
            results = self._lance_table.search(question).limit(10).to_list()
        except Exception:
            results = []

        if not results:
            # Try to get all data from DuckDB
            try:
                conn = duckdb.connect(self._duck_path, read_only=True)
                rows = conn.execute("SELECT content FROM turns").fetchall()
                conn.close()
                context_parts = [r[0] for r in rows]
            except Exception:
                context_parts = []
        else:
            context_parts = [r["text"] for r in results]

        context = "\n---\n".join(context_parts)
        context_tokens = len(context.split())

        # Call LLM to answer
        system_prompt = (
            "You are a helpful assistant. Answer the question based only on the "
            "provided context. Be concise and accurate."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {question}\n\n"
                    "Answer based only on the context above:"
                ),
            },
        ]
        answer = self._chat(messages)

        return QueryResult(
            answer=answer,
            total_latency_ms=0,  # filled by caller
            context_tokens=context_tokens,
            details={
                "method": "fallback",
                "fallback_triggered": True,
                "reason": fallback_reason,
                "num_results": len(context_parts),
            },
        )

    # ------------------------------------------------------------------
    # HTTP chat (fallback only, no external deps)
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
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
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
