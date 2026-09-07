from extract import extract_pdfs
from chunk import chunk_pages
from embed_store import SentenceTransformerEmbedder,build_store
from rag import answer_question
from transformers import AutoTokenizer , AutoModelForCausalLM, BitsAndBytesConfig
import torch
from google import genai
import time


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


pages = extract_pdfs("documents")

client = genai.Client()

# model_name = "Qwen/Qwen2.5-7B-Instruct"
# quant_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_use_double_quant=True,
# )

# print("[rag] Loading tokenizer for", model_name)
# try:
#         tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# except Exception as e:
#         print(f"[rag] Tokenizer load failed: {e}")
#         raise

# print("[rag] Loading model for", model_name)
# try:
#         llm_model = AutoModelForCausalLM.from_pretrained(
#             model_name,
#             dtype = torch.float16 if torch.cuda.is_available() else torch.float32,
#             quantization_config=quant_config,
#             device_map = "cuda",
#             trust_remote_code=True
#         )
# except Exception as e:
#         print(f"[rag] Model load failed: {e}")
#         raise

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
# query2 = "What is the insurance policy you have in your company and number and usage?"

# for k in (3, 5, 10):
#     q_vec = embedder.encode([query2])[0]
#     hits = store.search(q_vec, top_k=k)
#     print(f"\n--- top_k={k} ---")
#     for c, score in hits:
#         print(f"   {score:.3f}  {c.source} p.{c.page_num}  \"{c.text[:60].strip()}...\"")

# -----------------------------------------------------------------------
section("EXPERIMENT C: question with NO answer in the documents")
# -----------------------------------------------------------------------
query3 = "What is rag and computer vision and difference?"
result = answer_question(query3, store, embedder, top_k=3, use_gemini_llm=True, client=client)
print("Query:", query3)
print("Top match score:", result["sources"][0][2] if result["sources"] else None)
print("Response with grounding guard:", result["answer"])

# -----------------------------------------------------------------------
section("EXPERIMENT D: which document and page did this come from?")
# -----------------------------------------------------------------------
query4 = "If an employee joined company , any evaluation process after some months are there and how?"
result = answer_question(query4, store, embedder, top_k=3, use_gemini_llm=True, client=client)
print("Query:", query4)
print("Answer (extractive, top match):")
print(" ", result["answer"][:300].replace("\n", " "))
print("\nSources returned to user:")
for src, page, score in result["sources"]:
    print(f"   - {src}, page {page} (similarity {score})")
