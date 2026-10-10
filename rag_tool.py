def search_policies(vector_store, question, k=3):
    """Return the top-k most relevant policy chunks for the given question."""
    results = vector_store.similarity_search(question, k=k)
    return [
        {"content": doc.page_content, "metadata": doc.metadata}
        for doc in results
    ]