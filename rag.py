from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from policy_loader import load_policy_documents


def build_vector_store():
    """Build a Chroma vector store from the university IT policy documents."""
    documents = load_policy_documents()
    print(f"Loaded {len(documents)} policy documents.")

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://127.0.0.1:11434",  
    )

    vector_store = Chroma(
        collection_name="university_it_policies",
        embedding_function=embeddings,
        persist_directory="./chroma_db",     
    )

    BATCH = 5
    for i in range(0, len(documents), BATCH):
        batch = documents[i:i + BATCH]
        print(f"  embedding batch {i//BATCH + 1}/{(len(documents) + BATCH - 1)//BATCH} ...")
        vector_store.add_documents(batch)

    return vector_store