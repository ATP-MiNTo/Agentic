# Core RAG Agent implementation
import json
import time
from typing import List, Dict, Tuple, Optional

import config
from src.utils.logger import get_logger
from src.utils.prompt_templates import (
    SYSTEM_PROMPT,
    PLANNER_PROMPT,
    DECOMPOSITION_PROMPT,
    QUERY_REWRITE_PROMPT,
    CRITIC_PROMPT,
    REASONING_PROMPT,
    GENERATION_PROMPT,
    MEDICAL_DISCLAIMER,
)
from src.agent.embedding_model import EmbeddingModel
from src.agent.llm_interface import LLMInterface
from src.utils.session_memory import SessionMemoryStore

logger = get_logger()


class MedicalRAGAgent:
    """Medical Information RAG Agent with an iterative agentic loop."""

    def __init__(self, faiss_index=None, use_faiss: bool = True):
        logger.info("Initializing Medical RAG Agent...")

        self.embedding_model = EmbeddingModel()
        self.llm = LLMInterface()
        self.faiss_index = faiss_index
        self.use_faiss = use_faiss
        self.metadata = {}

        self.max_steps = getattr(config, "AGENT_MAX_STEPS", 4)
        self.min_evidence_docs = getattr(config, "AGENT_MIN_EVIDENCE_DOCS", 2)
        self.confidence_threshold = getattr(config, "AGENT_CONFIDENCE_THRESHOLD", 0.7)
        self.force_clarify_low_evidence = getattr(config, "AGENT_FORCE_CLARIFY_ON_LOW_EVIDENCE", True)
        self.max_subqueries = getattr(config, "AGENT_MAX_SUBQUERIES", 3)
        self.session_memory_enabled = getattr(config, "ENABLE_SESSION_MEMORY", True)
        self.session_memory = SessionMemoryStore() if self.session_memory_enabled else None

        if use_faiss and faiss_index is None:
            self._load_faiss_index()

        logger.info("Agent mode: iterative planner-tool-critic loop enabled")
        logger.info("✓ RAG Agent initialized successfully")

    def _load_faiss_index(self):
        """Load FAISS index from disk."""
        try:
            import faiss

            index_path = config.FAISS_INDEX_DIR / "index.faiss"
            metadata_path = config.FAISS_INDEX_DIR / "metadata.json"

            if not index_path.exists():
                logger.error(f"FAISS index not found at {index_path}")
                logger.info("Run: python src/utils/index_builder.py")
                raise FileNotFoundError("FAISS index not found")

            logger.debug(f"Loading FAISS index from {index_path}")
            self.faiss_index = faiss.read_index(str(index_path))

            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

            logger.info(f"✓ FAISS index loaded ({len(self.metadata)} documents)")

        except ImportError:
            logger.error("faiss-cpu not installed. Run: pip install faiss-cpu")
            raise
        except Exception as e:
            logger.error(f"Error loading FAISS index: {str(e)}")
            raise

    def retrieve(self, query: str, top_k: int = None) -> Tuple[List[str], List[float], List[str]]:
        """Retrieve relevant documents using semantic search."""
        if top_k is None:
            top_k = config.TOP_K

        logger.debug(f"Retrieving top {top_k} documents for query")
        start_time = time.time()

        try:
            query_embedding = self.embedding_model.embed_text(query).reshape(1, -1)
            distances, indices = self.faiss_index.search(query_embedding, top_k)

            documents: List[str] = []
            scores: List[float] = []
            doc_ids: List[str] = []

            for distance, idx in zip(distances[0], indices[0]):
                if idx == -1:
                    continue

                similarity = 1 / (1 + distance)
                metadata_idx = str(idx)
                if metadata_idx in self.metadata:
                    doc_id = self.metadata[metadata_idx]["id"]
                    content = self.metadata[metadata_idx]["content"]
                    documents.append(content)
                    scores.append(similarity)
                    doc_ids.append(doc_id)
                    logger.debug(f"Retrieved: {doc_id} (score: {similarity:.4f})")

            elapsed = time.time() - start_time
            logger.info(f"Retrieval completed in {elapsed:.2f}s ({len(documents)} docs)")
            logger.log_retrieval(doc_ids, scores)
            return documents, scores, doc_ids

        except Exception as e:
            logger.error(f"Error during retrieval: {str(e)}")
            raise

    def _evidence_lines(self, state: Dict) -> str:
        """Build concise evidence summary for planner/critic prompts."""
        if not state["documents"]:
            return "No evidence yet."

        lines = []
        for doc_id, entry in state["documents"].items():
            lines.append(f"- {doc_id} (score={entry['score']:.2f})")
        return "\n".join(lines)

    def _trace_lines(self, state: Dict) -> str:
        if not state["agent_trace"]:
            return "No actions yet."
        return "\n".join(
            f"- Step {item['step']}: {item['action']} -> {item['observation']}"
            for item in state["agent_trace"]
        )

    def _memory_lines(self) -> str:
        if not self.session_memory:
            return "Session memory disabled."
        return self.session_memory.build_summary()

    def _record_documents(self, state: Dict, docs: List[str], scores: List[float], doc_ids: List[str]):
        """Merge retrieved documents into state, keeping the best score per doc."""
        for doc_id, doc, score in zip(doc_ids, docs, scores):
            existing = state["documents"].get(doc_id)
            if existing is None or score > existing["score"]:
                state["documents"][doc_id] = {"content": doc, "score": score}

    def _decompose_query(self, query: str, state: Dict) -> Dict:
        prompt = DECOMPOSITION_PROMPT.format(
            query=query,
            memory=self._memory_lines(),
            max_subqueries=self.max_subqueries,
        )
        fallback = {"subqueries": [query], "strategy": "single-query retrieval"}
        plan = self.llm.generate_json(prompt, fallback=fallback, max_tokens=220, temperature=0.2)

        raw_subqueries = plan.get("subqueries", [])
        if not isinstance(raw_subqueries, list):
            raw_subqueries = [query]

        cleaned: List[str] = []
        for subquery in raw_subqueries:
            text = str(subquery).strip()
            if text and text not in cleaned:
                cleaned.append(text)
        if not cleaned:
            cleaned = [query]

        state["subqueries"] = cleaned[: self.max_subqueries]
        state["retrieval_strategy"] = str(plan.get("strategy", fallback["strategy"])).strip() or fallback["strategy"]
        return {"subqueries": state["subqueries"], "strategy": state["retrieval_strategy"]}

    def _retrieve_multi_query(self, queries: List[str], top_k: int, state: Dict) -> str:
        used_queries = []
        for subquery in queries[: self.max_subqueries]:
            docs, scores, doc_ids = self.retrieve(subquery, top_k)
            self._record_documents(state, docs, scores, doc_ids)
            used_queries.append(subquery)

        state["subqueries"] = used_queries
        state["retrieval_strategy"] = "multi-query retrieval"
        return f"Retrieved and merged evidence across {len(used_queries)} focused queries."

    def _rewrite_query(self, query: str, state: Dict) -> str:
        prompt = QUERY_REWRITE_PROMPT.format(
            query=query,
            evidence=self._evidence_lines(state),
        )
        rewritten = self.llm.generate(prompt, max_tokens=64, temperature=0.2).strip()
        return rewritten or query

    def _plan_next_action(self, query: str, state: Dict, step: int, top_k: int) -> Dict:
        fallback = {
            "action": "retrieve_top_k" if not state["documents"] else "finalize_answer",
            "arguments": {"query": query, "top_k": top_k},
            "expected_outcome": "Get sufficient evidence for answer.",
        }
        prompt = PLANNER_PROMPT.format(
            query=query,
            step=step,
            max_steps=self.max_steps,
            memory=self._memory_lines(),
            evidence=self._evidence_lines(state),
            trace=self._trace_lines(state),
        )
        plan = self.llm.generate_json(prompt, fallback=fallback, max_tokens=220, temperature=0.2)
        action = plan.get("action", fallback["action"])
        if action not in {
            "retrieve_top_k",
            "reretrieve_refined_query",
            "decompose_query",
            "retrieve_multi_query",
            "extract_evidence_by_topic",
            "clarify_user",
            "finalize_answer",
        }:
            action = fallback["action"]
        plan["action"] = action
        plan.setdefault("arguments", {})
        plan.setdefault("expected_outcome", fallback["expected_outcome"])
        return plan

    def _execute_tool(self, action: str, arguments: Dict, query: str, top_k: int, state: Dict) -> str:
        """Execute one planner-selected tool and update state."""
        if action == "retrieve_top_k":
            retrieval_query = arguments.get("query", query)
            retrieve_k = int(arguments.get("top_k", top_k))
            docs, scores, doc_ids = self.retrieve(retrieval_query, retrieve_k)
            self._record_documents(state, docs, scores, doc_ids)
            return f"Retrieved {len(doc_ids)} documents using query '{retrieval_query[:80]}'."

        if action == "reretrieve_refined_query":
            rewritten = self._rewrite_query(query, state)
            retrieve_k = int(arguments.get("top_k", top_k))
            docs, scores, doc_ids = self.retrieve(rewritten, retrieve_k)
            self._record_documents(state, docs, scores, doc_ids)
            return f"Re-retrieved {len(doc_ids)} documents using refined query '{rewritten[:80]}'."

        if action == "decompose_query":
            decomposition = self._decompose_query(query, state)
            return f"Decomposed query into {len(decomposition['subqueries'])} focused retrieval prompts."

        if action == "retrieve_multi_query":
            subqueries = arguments.get("subqueries")
            if not isinstance(subqueries, list) or not subqueries:
                subqueries = state.get("subqueries") or self._decompose_query(query, state)["subqueries"]
            retrieve_k = int(arguments.get("top_k", max(2, top_k // 2 or 1)))
            return self._retrieve_multi_query([str(item) for item in subqueries], retrieve_k, state)

        if action == "extract_evidence_by_topic":
            topic = str(arguments.get("topic", "")).strip().lower()
            if not state["documents"]:
                return "No evidence available to extract from."

            matched = []
            for doc_id, entry in state["documents"].items():
                content = entry["content"]
                if topic and topic in content.lower():
                    matched.append((doc_id, content))

            if not matched:
                top_docs = sorted(
                    state["documents"].items(),
                    key=lambda item: item[1]["score"],
                    reverse=True,
                )[:2]
                matched = [(doc_id, item["content"]) for doc_id, item in top_docs]

            snippets = [f"{doc_id}: {content[:180]}" for doc_id, content in matched]
            state["focused_evidence"] = snippets
            return f"Extracted focused evidence from {len(matched)} documents for topic '{topic or 'general'}'."

        if action == "clarify_user":
            question = str(
                arguments.get("question")
                or arguments.get("clarifying_question")
                or "Could you clarify your question so I can retrieve the right medical evidence?"
            ).strip()
            state["clarifying_question"] = question
            return question

        if action == "finalize_answer":
            return "Planner selected finalization."

        docs, scores, doc_ids = self.retrieve(query, top_k)
        self._record_documents(state, docs, scores, doc_ids)
        return f"Unknown action fallback executed; retrieved {len(doc_ids)} documents."

    def _critic_review(self, query: str, state: Dict, last_observation: str) -> Dict:
        fallback = {
            "score": 0.0,
            "needs_more_evidence": True,
            "missing_points": ["insufficient structured evidence"],
            "confidence": "low",
            "clarifying_question": "",
        }
        prompt = CRITIC_PROMPT.format(
            query=query,
            evidence=self._evidence_lines(state),
            trace=self._trace_lines(state),
            last_observation=last_observation,
        )
        critic = self.llm.generate_json(prompt, fallback=fallback, max_tokens=220, temperature=0.2)

        score = critic.get("score", 0.0)
        try:
            critic["score"] = max(0.0, min(1.0, float(score)))
        except (TypeError, ValueError):
            critic["score"] = 0.0

        critic["needs_more_evidence"] = bool(critic.get("needs_more_evidence", True))
        critic["confidence"] = str(critic.get("confidence", "low")).lower()
        if critic["confidence"] not in {"low", "medium", "high"}:
            critic["confidence"] = "low"
        if not isinstance(critic.get("missing_points"), list):
            critic["missing_points"] = []
        critic["clarifying_question"] = str(critic.get("clarifying_question", "")).strip()
        return critic

    def reason(self, query: str, documents: List[str], doc_ids: List[str]) -> str:
        """Summarize how retrieved evidence supports an answer."""
        logger.debug("Starting reasoning step")

        context = ""
        for doc_id, doc_content in zip(doc_ids, documents):
            context += f"[From: {doc_id}]\n{doc_content[:300]}...\n\n"

        prompt = REASONING_PROMPT.format(
            query=query,
            strategy=getattr(self, "retrieval_strategy", "agent-planned retrieval"),
            documents=context,
        )

        try:
            reasoning = self.llm.generate(prompt, max_tokens=220, temperature=0.5)
            logger.log_reasoning(reasoning)
            return reasoning
        except Exception as e:
            logger.error(f"Error during reasoning: {str(e)}")
            return (
                f"Evidence reviewed from {len(documents)} documents. "
                f"Top sources: {', '.join(doc_ids[:3]) or 'none'}"
            )

    def generate_response(
        self,
        query: str,
        documents: List[str],
        doc_ids: List[str],
        scores: List[float],
        reasoning: str,
        agent_trace: str = "",
    ) -> str:
        """Generate final response using LLM."""
        logger.debug("Starting response generation")
        start_time = time.time()

        context_parts = []
        for doc_id, doc_content, score in zip(doc_ids, documents, scores):
            context_parts.append(f"[Source: {doc_id} | Confidence: {score:.1%}]\n{doc_content}\n")
        context = "\n---\n".join(context_parts)

        prompt = GENERATION_PROMPT.format(
            system_prompt=SYSTEM_PROMPT,
            query=query,
            context=context,
            agent_trace=agent_trace or "No intermediate tool trace available.",
            reasoning=reasoning,
        )

        try:
            response = self.llm.generate(
                prompt,
                max_tokens=config.MAX_TOKENS,
                temperature=config.TEMPERATURE,
            )
            full_response = f"{MEDICAL_DISCLAIMER}\n\n{response}"
            elapsed = time.time() - start_time
            logger.info(f"Response generated in {elapsed:.2f}s")
            return full_response
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return (
                f"{MEDICAL_DISCLAIMER}\n\n"
                "I encountered an error while generating a response. Please try again."
            )

    def chat(self, query: str, top_k: int = None, return_details: bool = False) -> Dict:
        """Main chat method with iterative plan-act-observe-critic behavior."""
        logger.log_query(query)
        start_time = time.time()

        if top_k is None:
            top_k = config.TOP_K

        state = {
            "documents": {},
            "agent_trace": [],
            "critic_history": [],
            "focused_evidence": [],
            "subqueries": [],
            "retrieval_strategy": "",
            "clarifying_question": "",
        }

        try:
            for step in range(1, self.max_steps + 1):
                plan = self._plan_next_action(query, state, step, top_k)
                action = plan["action"]
                arguments = plan.get("arguments", {})

                logger.info(f"Step {step}/{self.max_steps} planning action: {action}")
                observation = self._execute_tool(action, arguments, query, top_k, state)

                trace_item = {
                    "step": step,
                    "action": action,
                    "arguments": arguments,
                    "expected_outcome": plan.get("expected_outcome", ""),
                    "observation": observation,
                }
                state["agent_trace"].append(trace_item)

                critic = self._critic_review(query, state, observation)
                critic["step"] = step
                state["critic_history"].append(critic)

                if action == "clarify_user" and not state.get("clarifying_question"):
                    state["clarifying_question"] = observation

                enough_docs = len(state["documents"]) >= self.min_evidence_docs
                high_confidence = critic["score"] >= self.confidence_threshold

                logger.info(
                    f"Step {step} critic: score={critic['score']:.2f}, "
                    f"needs_more_evidence={critic['needs_more_evidence']}, confidence={critic['confidence']}"
                )

                if action == "finalize_answer":
                    break

                if action == "clarify_user":
                    break

                if enough_docs and high_confidence and not critic["needs_more_evidence"]:
                    logger.info("Critic indicates sufficient evidence; stopping early")
                    break

            if not state["documents"]:
                clarifier = state.get("clarifying_question", "")
                if not clarifier and state["critic_history"]:
                    clarifier = state["critic_history"][-1].get("clarifying_question", "")
                response = "I couldn't find relevant medical information to answer your question."
                if clarifier:
                    response = f"{response} Could you clarify: {clarifier}"
                if self.session_memory:
                    self.session_memory.record_turn(
                        query=query,
                        response=response,
                        doc_ids=[],
                        critic_score=float(state["critic_history"][-1].get("score", 0.0)) if state["critic_history"] else 0.0,
                        subqueries=state.get("subqueries", []),
                        clarifying_question=clarifier,
                    )
                return {
                    "response": response,
                    "success": False,
                }

            sorted_docs = sorted(
                state["documents"].items(),
                key=lambda item: item[1]["score"],
                reverse=True,
            )
            doc_ids = [doc_id for doc_id, _ in sorted_docs]
            documents = [item["content"] for _, item in sorted_docs]
            scores = [item["score"] for _, item in sorted_docs]

            reasoning = self.reason(query, documents, doc_ids)
            self.retrieval_strategy = state.get("retrieval_strategy", "agent-planned retrieval")
            agent_trace_text = "\n".join(
                f"Step {item['step']}: {item['action']} -> {item['observation']}"
                for item in state["agent_trace"]
            )
            response = self.generate_response(
                query,
                documents,
                doc_ids,
                scores,
                reasoning,
                agent_trace=agent_trace_text,
            )

            last_critic = state["critic_history"][-1] if state["critic_history"] else {}
            if (
                self.force_clarify_low_evidence
                and float(last_critic.get("score", 0.0)) < self.confidence_threshold
                and last_critic.get("clarifying_question")
            ):
                response += (
                    "\n\nI may need one clarification to be more precise: "
                    f"{last_critic['clarifying_question']}"
                )

            logger.log_response(response, tokens=len(response.split()))
            elapsed = time.time() - start_time
            logger.log_execution_time("Total query", elapsed)

            if self.session_memory:
                self.session_memory.record_turn(
                    query=query,
                    response=response,
                    doc_ids=doc_ids,
                    critic_score=float(last_critic.get("score", 0.0)),
                    subqueries=state.get("subqueries", []),
                    clarifying_question=last_critic.get("clarifying_question", ""),
                )

            result = {
                "response": response,
                "success": True,
                "execution_time": elapsed,
            }

            if return_details:
                result.update(
                    {
                        "retrieved_documents": [
                            {"id": doc_id, "score": score, "preview": doc[:200]}
                            for doc_id, doc, score in zip(doc_ids, documents, scores)
                        ],
                        "reasoning": reasoning,
                        "agent_trace": state["agent_trace"],
                        "critic_history": state["critic_history"],
                        "iterations": len(state["agent_trace"]),
                        "final_confidence": float(last_critic.get("score", 0.0)),
                        "query": query,
                        "subqueries": state.get("subqueries", []),
                        "retrieval_strategy": state.get("retrieval_strategy", "agent-planned retrieval"),
                    }
                )

            return result

        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            return {
                "response": f"An error occurred: {str(e)}",
                "success": False,
                "error": str(e),
            }
