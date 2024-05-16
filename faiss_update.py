def update_faiss_database(embeddings, vectorStore_a, vectorStore_b):
    vectorStore_a.merge_from(vectorStore_b)