import textwrap

# Prompt templates for the agentic RAG system

SYSTEM_PROMPT = textwrap.dedent("""
You are a knowledgeable medical information assistant. Your role is to:

1. Answer questions accurately using only provided medical documents
2. Cite sources when providing information
3. Be explicit about uncertainty and information gaps
4. Prioritize user safety and encourage professional medical consultation
5. Structure responses clearly with bullet points or sections when useful

IMPORTANT DISCLAIMERS:
- Always include a medical safety disclaimer
- Encourage users to consult healthcare professionals
- Do not provide diagnosis or treatment plans
- Focus on educational information only

Medical Topics Covered:
- Migraine: symptoms, triggers, treatments, prevention
- Diabetes: types, symptoms, management, complications
- Cancer: overview, staging, symptoms, treatments
""").strip()


PLANNER_PROMPT = textwrap.dedent("""
You are the planning module for a medical RAG agent.

Question: {query}
Current Step: {step}
Max Steps: {max_steps}

Recent Memory:
{memory}

Current Evidence:
{evidence}

Previous Tool Trace:
{trace}

You must choose exactly one next action from:
1) retrieve_top_k
2) reretrieve_refined_query
3) decompose_query
4) retrieve_multi_query
5) extract_evidence_by_topic
6) clarify_user
7) finalize_answer

Return strict JSON only:
{{
    "action": "retrieve_top_k",
    "arguments": {{
        "query": "optional query text",
        "top_k": 5,
        "topic": "optional topic string",
        "subqueries": ["optional subquery 1", "optional subquery 2"]
    }},
    "expected_outcome": "short sentence"
}}

Rules:
- Use retrieve_top_k if evidence is empty.
- Use reretrieve_refined_query if evidence is weak or off-topic.
- Use decompose_query if the question is compound, broad, or has multiple intents.
- Use retrieve_multi_query after decomposition when multiple focused searches are needed.
- Use extract_evidence_by_topic when you have evidence but need precision.
- Use clarify_user if the question is too vague to answer safely.
- Use finalize_answer only when evidence is sufficient.
""").strip()


DECOMPOSITION_PROMPT = textwrap.dedent("""
Break the medical question into focused retrieval subqueries.

Question: {query}

Recent Memory:
{memory}

Return strict JSON only:
{{
    "subqueries": ["focused subquery 1", "focused subquery 2"],
    "strategy": "short sentence"
}}

Rules:
- Create 2 to {max_subqueries} subqueries when possible.
- Keep each subquery short, specific, and retrieval-friendly.
- If the question is already simple, return a single subquery.
""").strip()


QUERY_REWRITE_PROMPT = textwrap.dedent("""
Rewrite the user question for semantic retrieval.

Original question: {query}
Current evidence summary:
{evidence}

Return only one rewritten query line that is:
- specific
- medically relevant
- concise
- optimized for retrieval
""").strip()


CRITIC_PROMPT = textwrap.dedent("""
You are a strict critic for a medical RAG agent.

Question: {query}
Evidence:
{evidence}

Agent trace:
{trace}

Last observation:
{last_observation}

Return strict JSON only:
{{
    "score": 0.0,
    "needs_more_evidence": true,
    "missing_points": ["point 1", "point 2"],
    "confidence": "low",
    "clarifying_question": ""
}}

Rules:
- score is from 0.0 to 1.0
- confidence must be one of: low, medium, high
- If evidence is insufficient, set needs_more_evidence=true
- If the question is ambiguous, add a short clarifying question
""").strip()


REASONING_PROMPT = textwrap.dedent("""
Analyze the following medical question and retrieved documents.

Question: {query}

Retrieval strategy used:
{strategy}

Retrieved Documents:
{documents}

Please provide concise reasoning:
1. Which parts of the question are addressed by evidence?
2. What gaps remain?
3. What is your confidence and why?

Reasoning:
""").strip()


GENERATION_PROMPT = textwrap.dedent("""
{system_prompt}

---

USER QUESTION: {query}

RETRIEVED CONTEXT (from medical documents):
{context}

AGENT TRACE SUMMARY:
{agent_trace}

REASONING SUMMARY:
{reasoning}

---

Write a final answer based only on the retrieved context.

Guidelines:
1. Keep the response medically safe and educational
2. Cite specific sources by document name
3. Use concise bullet points when appropriate
4. Clearly call out uncertainty or missing evidence
5. End by advising professional consultation when relevant

Answer:
""").strip()


# Medical disclaimer
MEDICAL_DISCLAIMER = textwrap.dedent("""
⚠️ MEDICAL DISCLAIMER:
This information is for educational purposes only and is NOT a substitute for professional 
medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider 
before making any medical decisions or changes to your healthcare routine.
""").strip()


# Function to format retrieved documents
def format_documents_for_context(docs: list, scores: list = None) -> str:
    """Format retrieved documents for inclusion in prompts.
    
    Args:
        docs: List of document contents
        scores: List of similarity scores
    
    Returns:
        Formatted string for including in prompts
    """
    if not docs:
        return "No documents retrieved"
    
    formatted = []
    for i, doc in enumerate(docs, 1):
        score_str = f" (Score: {scores[i-1]:.2%})" if scores and i <= len(scores) else ""
        formatted.append(f"[Document {i}]{score_str}:\n{doc}\n")
    
    return "\n".join(formatted)


# Function to format reasoning
def format_reasoning(reasoning: str) -> str:
    """Format reasoning for inclusion in prompts.
    
    Args:
        reasoning: Reasoning text from agent
    
    Returns:
        Formatted reasoning string
    """
    if not reasoning:
        return "No reasoning available"
    
    return reasoning


# Prompt for indexing step (system message)
INDEXING_PROMPT = textwrap.dedent("""
You are analyzing medical documents for a retrieval system.
Your task is to understand the semantic meaning of documents to enable accurate retrieval.

Document: {document}

This document will be indexed and used to answer medical questions.
""").strip()


# Prompt for evaluation (used during development)
EVAL_PROMPT = textwrap.dedent("""
Evaluate the quality of the following medical response.

Question: {query}
Response: {response}
Retrieved Documents: {docs}

Rate on:
1. Accuracy (0-10): Based on retrieved documents
2. Completeness (0-10): Addresses all aspects of question
3. Clarity (0-10): Easy to understand
4. Safety (0-10): Includes appropriate disclaimers

Provide brief feedback:
""").strip()
