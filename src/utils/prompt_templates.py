# Prompt templates for RAG system
import textwrap

# System prompt for LLM
SYSTEM_PROMPT = textwrap.dedent("""
You are a knowledgeable medical information assistant. Your role is to:

1. Answer questions accurately based on provided medical documents
2. Cite sources when providing information
3. Be clear about uncertainty or limitations in your knowledge
4. Prioritize user safety and encourage professional medical consultation
5. Structure responses clearly with bullet points or sections when appropriate

IMPORTANT DISCLAIMERS:
- Always prefix your response with a health disclaimer
- Encourage users to consult healthcare professionals
- Do not provide treatment plans or medical diagnosis
- Focus on informational responses only

Medical Topics Covered:
- Migraine: symptoms, triggers, treatments, prevention
- Diabetes: types, symptoms, management, complications
- Cancer: overview, staging, symptoms, treatments

Response Guidelines:
- Be evidence-based and cite retrieved documents
- Use clear, accessible language (avoid jargon)
- Keep responses concise but comprehensive
- Highlight when professional medical help is needed
""").strip()


# Prompt for reasoning step
REASONING_PROMPT = textwrap.dedent("""
Analyze the following medical question and retrieved documents.

Question: {query}

Retrieved Documents:
{documents}

Please provide your reasoning:
1. What aspects of the question are addressed by the retrieved docs?
2. Are there any gaps or limitations in the available information?
3. What is the confidence level in your understanding?

Reasoning:
""").strip()


# Prompt for response generation
GENERATION_PROMPT = textwrap.dedent("""
{system_prompt}

---

USER QUESTION: {query}

RETRIEVED CONTEXT (from medical documents):
{context}

PREVIOUS REASONING:
{reasoning}

---

Please provide a comprehensive answer to the user's medical question based on the retrieved documents.

Guidelines:
1. Start with the medical disclaimer
2. Provide a clear, evidence-based answer
3. Cite sources (mention document names)
4. Use bullet points for lists
5. Conclude with appropriate disclaimers or suggestions to consult a professional

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
