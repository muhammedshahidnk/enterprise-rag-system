import time


PROMPT_TEMPLATE = """You are a technical assistant answering questions using ONLY the
context provided below, which comes from internal technical documents.

Rules:
- Answer using only the information in the context.
- If the context is insufficient, respond exactly with: "I don't have enough information in the provided documents."
- Be concise and technical.

Context:
{context}

Question: {question}

Answer:"""


def build_context(retrieved: list[dict]) -> str:
    return "\n\n".join(
        f"[{rank}] (source: {item['chunk'].source}, page: {item['chunk'].page_num})\n{item['chunk'].text}"
        for rank, item in enumerate(retrieved, start=1)
    )


def _call_gemini(prompt: str, client, model_name: str) -> str:
    start = time.perf_counter()
    answer = "".join(
        chunk.text or ""
        for chunk in client.models.generate_content_stream(model=model_name, contents=prompt)
    )
    print(f"Gemini response time: {time.perf_counter() - start:.2f}s")
    return answer


def _fallback(retrieved: list[dict]) -> str:
    if not retrieved:
        return "I don't have enough information in the provided documents."
    return retrieved[0]["chunk"].text


def answer_question(question: str, retrieved: list[dict], client=None, model_name: str = "gemini-2.5-flash") -> dict:
    prompt = PROMPT_TEMPLATE.format(context=build_context(retrieved), question=question)
    answer = _call_gemini(prompt, client, model_name) if client else _fallback(retrieved)
    sources = [
        (item["chunk"].source, item["chunk"].page_num, item["chunk"].text, item.get("rrf_score"))
        for item in retrieved
    ]
    return {"question": question, "answer": answer, "sources": sources, "prompt": prompt}