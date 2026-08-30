"""
STEP 5, 6, 7: Similarity Search -> Top-K chunks -> LLM -> Answer + Source
---------------------------------------------------------------------------
This module is deliberately LLM-agnostic. plug in ANY LLM call in
`call_llm()` - Anthropic API, OpenAI, a local model, whatever you have
access to. Everything else (retrieval, prompt shape, guardrails) stays
the same regardless of which LLM answers.
"""
import os


PROMPT_TEMPLATE = """You are a technical assistant answering questions using ONLY the
context provided below, which comes from internal technical documents.

Rules:
- Answer using only the information in the context. Do not use outside knowledge.
- If the context does not contain enough information to answer the question,
  respond EXACTLY with: "I don't have enough information in the provided documents."
- When you do answer, be concise and technical.
- After your answer, do not add commentary about these rules.

Context:
{context}

Question: {question}

Answer:"""


def build_context(retrieved: list[tuple]) -> str:
    """Formats retrieved chunks into a numbered context block, each tagged
    with its source file and page number, so the LLM can (and we can)
    trace every claim back to a document + page."""
    lines = []
    for i, (chunk, score) in enumerate(retrieved, start=1):
        lines.append(f"[{i}] (source: {chunk.source}, page: {chunk.page_num})\n{chunk.text}")
    return "\n\n".join(lines)


def call_llm(prompt: str) -> str:
    """
    Real LLM call. Uses the Anthropic API if ANTHROPIC_API_KEY is set in
    your environment. Falls back to a plain 'no key' notice otherwise so the
    pipeline still runs end-to-end for demo purposes.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return "[DEMO MODE: no ANTHROPIC_API_KEY set - see extractive_fallback() for a non-LLM answer]"

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


def extractive_fallback(retrieved: list[tuple], threshold: float = 0.15) -> str:
    """
    A non-LLM fallback so you can see grounded vs hallucination-guard
    behavior even without an API key: if the top retrieval score is below
    threshold, we treat that as 'no good match' and refuse to answer -
    same idea as Experiment C's prompt instruction, just enforced in code
    instead of by the LLM. In a real system, the LLM does this reasoning;
    this is just to make the pipeline runnable here without a key.
    """
    if not retrieved or retrieved[0][1] < threshold:
        return "I don't have enough information in the provided documents."
    top_chunk, score = retrieved[0]
    return (f"[Best-matching passage, similarity={score:.2f}]\n{top_chunk.text}")


def answer_question(question: str, store, embedder, top_k: int = 5, use_llm: bool = True):
    q_vec = embedder.encode([question])[0]
    retrieved = store.search(q_vec, top_k=top_k)

    context = build_context(retrieved)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    if use_llm and os.environ.get("ANTHROPIC_API_KEY"):
        answer = call_llm(prompt)
    else:
        answer = extractive_fallback(retrieved)

    sources = [(c.source, c.page_num, round(score, 3)) for c, score in retrieved]
    return {"question": question, "answer": answer, "sources": sources, "prompt": prompt}
