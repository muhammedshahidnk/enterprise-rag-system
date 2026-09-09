from extract import extract_pdfs
from chunk import chunk_pages
from embed_store import SentenceTransformerEmbedder,build_store
from rag import answer_question
from transformers import AutoTokenizer , AutoModelForCausalLM, BitsAndBytesConfig
import torch
from google import genai
import time
from keyword_search import BM25Retriever
from rag import compare_retrievers


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


pages = extract_pdfs("documents")

client = genai.Client()


# # -----------------------------------------------------------------------
# section("EXPERIMENT A: chunk_size 300 vs 800")
# # -----------------------------------------------------------------------
# query = "How many leave will get for the husband of pregnant wife?"
# # query = "Does paternity leave available for husband? "

# for size in (300,800):
#     chunks = chunk_pages(pages, chunk_size=size, overlap=50)
#     embedder = SentenceTransformerEmbedder()
#     store = build_store(chunks, embedder)
#     result = answer_question(query, store, embedder, top_k=3, use_gemini_llm=True, client=client)
#     print(f"\n--- chunk_size={size} ({len(chunks)} total chunks) ---")
#     print("Query:", query)
#     for src, page, score in result["sources"]:
#         print(f"   match: {src} p.{page}  score={score}")
#     print("Top chunk text snippet:")
#     print("  ", result["answer"][:220].replace("\n", " "))

# # ----------------------------------------------------------------------
# section("EXPERIMENT B: top_k 3 vs 5 vs 10")
# # -----------------------------------------------------------------------
chunks = chunk_pages(pages, chunk_size=500, overlap=50)
embedder = SentenceTransformerEmbedder()
store = build_store(chunks, embedder)

bm25 = BM25Retriever(chunks)
# query2 = "What is the insurance policy you have in your company and number and usage?"

# for k in (3, 5, 10):
#     q_vec = embedder.encode([query2])[0]
#     hits = store.search(q_vec, top_k=k)
#     print(f"\n--- top_k={k} ---")
#     for c, score in hits:
#         print(f"   {score:.3f}  {c.source} p.{c.page_num}  \"{c.text[:60].strip()}...\"")

# -----------------------------------------------------------------------
# section("EXPERIMENT C: question with NO answer in the documents")
# # -----------------------------------------------------------------------
# query3 = "Use of 04066274205 number and what is this for?"
# result = answer_question(query3, store, embedder, top_k=3, use_gemini_llm=True, client=client)
# print("Query:", query3)
# print("Top match score:", result["sources"][0][2] if result["sources"] else None)
# print("Response with grounding guard:", result["answer"])

test_queries = [
    "What is the use of 575758 sms number?",
    "Do you know anything about the number 575758?",
    "How can I track my claim?",
    "How do I track claim status?",
    "What application can be used for claim tracking?",
]

compare_retrievers(test_queries, store, bm25, embedder, top_k=5)


# query4 = "Do you know anything about the number 575758?"
# query4 = "What is the use of 575758 sms number?"
# for que in test_queries:
#     query4 = que

#     search = "bm25"  # or "semantic"

#     result = answer_question(query4, store,bm25, embedder, top_k=5, use_gemini_llm=True, search=search, client=client)
#     print("Query:", query4)
#     print("Answer (extractive, top match):")
#     print(" ", result["answer"][:300].replace("\n", " "))
#     print("\nSources returned to user:")
#     for src, page,text, score in result["sources"]:
#         print(f"   - {src}, page {page} (similarity {score})")
        # print("     snippet:", text)



















# -----------------------------------------------------------------------
# section("EXPERIMENT D: which document and page did this come from?")
# # # -----------------------------------------------------------------------
# # query4 = "If an employee joined company , any evaluation process after some months are there and how?"

# query4 = "What is the use of 575758 sms number?"

# result = answer_question(query4, store, embedder, top_k=3, use_gemini_llm=True, client=client)
# print("Query:", query4)
# print("Answer (extractive, top match):")
# print(" ", result["answer"][:300].replace("\n", " "))
# print("\nSources returned to user:")
# for src, page,text, score in result["sources"]:
#     print(f"   - {src}, page {page} (similarity {score})")
#     print("     snippet:", text)