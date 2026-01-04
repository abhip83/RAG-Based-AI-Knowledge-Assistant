def retrieve_docs(vectorstore, query, k):
    return vectorstore.similarity_search(query, k=k)
