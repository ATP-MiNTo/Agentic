# Lightweight local session memory for the agentic loop
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import config
from src.utils.logger import get_logger

logger = get_logger()


class SessionMemoryStore:
    """Persist a small amount of local agent memory between turns."""

    def __init__(self, memory_file: Optional[Path] = None, max_entries: int = 20):
        self.memory_file = memory_file or (config.LOG_DIR / "session_memory.json")
        self.max_entries = max_entries
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        self._state = self._load()

    def _load(self) -> Dict:
        if not self.memory_file.exists():
            return {"turns": []}

        try:
            with open(self.memory_file, "r", encoding="utf-8") as handle:
                state = json.load(handle)
            if isinstance(state, dict) and isinstance(state.get("turns"), list):
                return state
        except Exception as exc:
            logger.warning(f"Could not load session memory: {exc}")

        return {"turns": []}

    def _save(self) -> None:
        try:
            with open(self.memory_file, "w", encoding="utf-8") as handle:
                json.dump(self._state, handle, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.warning(f"Could not save session memory: {exc}")

    @staticmethod
    def _keywords(text: str) -> List[str]:
        tokens = re.findall(r"[A-Za-z][A-Za-z\-]{3,}", text.lower())
        stopwords = {
            "what",
            "when",
            "where",
            "which",
            "about",
            "with",
            "from",
            "that",
            "this",
            "these",
            "those",
            "have",
            "there",
            "their",
            "would",
            "could",
            "should",
            "symptom",
            "symptoms",
        }
        return [token for token in tokens if token not in stopwords]

    def record_turn(
        self,
        query: str,
        response: str,
        doc_ids: List[str],
        critic_score: float,
        subqueries: Optional[List[str]] = None,
        clarifying_question: str = "",
    ) -> None:
        turn = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "query": query,
            "response_preview": response[:180],
            "doc_ids": doc_ids[:5],
            "critic_score": round(float(critic_score or 0.0), 3),
            "subqueries": (subqueries or [])[:5],
            "clarifying_question": clarifying_question,
            "keywords": self._keywords(query + " " + " ".join(subqueries or [])),
        }

        self._state.setdefault("turns", []).append(turn)
        self._state["turns"] = self._state["turns"][-self.max_entries :]
        self._save()

    def build_summary(self, limit: int = 5) -> str:
        turns = self._state.get("turns", [])
        if not turns:
            return "No prior session memory yet."

        recent = turns[-limit:]
        lines = []
        for turn in reversed(recent):
            docs = ", ".join(turn.get("doc_ids", [])[:3]) or "none"
            subqueries = turn.get("subqueries", [])
            subquery_text = f" | subqueries: {'; '.join(subqueries[:2])}" if subqueries else ""
            clarifier = turn.get("clarifying_question", "")
            clarifier_text = f" | clarify: {clarifier}" if clarifier else ""
            lines.append(
                f"- {turn.get('timestamp', '')} | score={turn.get('critic_score', 0.0):.2f} | docs: {docs}{subquery_text}{clarifier_text}"
            )

        keyword_counts = Counter()
        for turn in recent:
            keyword_counts.update(turn.get("keywords", []))

        top_keywords = [keyword for keyword, count in keyword_counts.most_common(5) if count > 1]
        if top_keywords:
            lines.append(f"Recurring signals: {', '.join(top_keywords)}")

        return "\n".join(lines)
