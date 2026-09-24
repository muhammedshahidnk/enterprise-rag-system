if __package__:
    from src.enterprise_rag.rag_system import RagSystem
else:
    from enterprise_rag.rag_system import RagSystem


if __name__ == "__main__":
    rag_system = RagSystem()

    query = "What is the use of 575758 sms number?"
    # result = rag_system.answer_query(
    #     query,
    #     use_gemini_llm=True,
    # )

# pages = extract_pdfs("Data")

# client = genai.Client()


# chunks = chunk_pages(pages, chunk_size=800, overlap=70)


# embedder = SentenceTransformerEmbedder()
# store = build_store(chunks, embedder)

# bm25 = BM25Retriever(chunks)

# hybrid_retriever = HybridRetriever(store, bm25, embedder)


# test_queries = [
#     "What is the use of 575758 sms number?",
#     "Do you know anything about the number 575758?",
#     "How can I track my claim?",
#     "How do I track claim status?",
#     "What application can be used for claim tracking?",
#     "Which company shahid  working currently and his previous company ?",
#     "How many experience years does shahid have and in which field and company?",
#     "Date of releaving in previous and date of joining in current company of shahid?"
# ]

# compare_retrievers(test_queries, store, bm25, embedder, top_k=5)


# query4 = "Do you know anything about the number 575758?"
# query4 = "What is the use of 575758 sms number?"



# for que in test_queries:
#     query4 = que

#     retrieved = hybrid_retriever.search(query4,candidate_k=5, top_k=10)

#     result = answer_question(query4, use_gemini_llm=True, retrieved=retrieved, client=client)

#     print("Query:", query4)
#     print("Answer :")
#     print(" ", result["answer"])
#     print("\nSources returned to user:")
#     for src, page,text, score in result["sources"]:
#         print(f"   - {src}, page {page} (similarity {score})")
        # print("     snippet:", text)






    # print("Query:", query4)
    # for rank,results in enumerate(retrieved, start=1):
    #     print(f"\nRank {rank} ")
    #     print(f"RRF Score: {results['rrf_score']:.4f}")
    #     print(f"Semantic Score: {results['semantic_score']}")   
    #     print(f"BM25 Score: {results['bm25_score']}")
    #     print(f"Source: {results['chunk'].source} | Page: {results['chunk'].page_num}")
    #     print(f"Text: {results['chunk'].text[:100].replace(chr(10), ' ')}...")

    

