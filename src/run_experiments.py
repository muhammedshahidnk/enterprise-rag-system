from src.extract import extract_pdfs
from documents.chunk import chunk_pages
from src.embed_store import TfidfEmbedder, build_store
from src.rag import answer_question


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


pages = extract_pdfs("documents")

# -----------------------------------------------------------------------
section("EXPERIMENT A: chunk_size 300 vs 800")
# -----------------------------------------------------------------------
query = "How often should the spindle be lubricated?"

for size in (300, 800):
    chunks = chunk_pages(pages, chunk_size=size, overlap=50)
    embedder = TfidfEmbedder()
    store = build_store(chunks, embedder)
    result = answer_question(query, store, embedder, top_k=3, use_llm=False)
    print(f"\n--- chunk_size={size} ({len(chunks)} total chunks) ---")
    print("Query:", query)
    for src, page, score in result["sources"]:
        print(f"   match: {src} p.{page}  score={score}")
    print("Top chunk text snippet:")
    print("  ", result["answer"][:220].replace("\n", " "))

# -----------------------------------------------------------------------
section("EXPERIMENT B: top_k 3 vs 5 vs 10")
# -----------------------------------------------------------------------
chunks = chunk_pages(pages, chunk_size=500, overlap=50)
embedder = TfidfEmbedder()
store = build_store(chunks, embedder)
query2 = "What is the acceptance criteria for incoming material inspection?"

for k in (3, 5, 10):
    q_vec = embedder.encode([query2])[0]
    hits = store.search(q_vec, top_k=k)
    print(f"\n--- top_k={k} ---")
    for c, score in hits:
        print(f"   {score:.3f}  {c.source} p.{c.page_num}  \"{c.text[:60].strip()}...\"")

# -----------------------------------------------------------------------
section("EXPERIMENT C: question with NO answer in the documents")
# -----------------------------------------------------------------------
query3 = "What is the company's maternity leave policy?"
result = answer_question(query3, store, embedder, top_k=3, use_llm=False)
print("Query:", query3)
print("Top match score:", result["sources"][0][2] if result["sources"] else None)
print("Response with grounding guard:", result["answer"])

# -----------------------------------------------------------------------
section("EXPERIMENT D: which document and page did this come from?")
# -----------------------------------------------------------------------
query4 = "What should I do if I find non-conforming material during inspection?"
result = answer_question(query4, store, embedder, top_k=3, use_llm=False)
print("Query:", query4)
print("Answer (extractive, top match):")
print(" ", result["answer"][:300].replace("\n", " "))
print("\nSources returned to user:")
for src, page, score in result["sources"]:
    print(f"   - {src}, page {page} (similarity {score})")
