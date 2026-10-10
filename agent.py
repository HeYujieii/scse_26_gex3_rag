from ollama import chat

from rag import build_vector_store
from rag_tool import search_policies


QUESTION = "What exact phone number should I call for the University IT Service Desk?"


def main():
    print("Building vector store...")
    vector_store = build_vector_store()

    print("Searching policies...")
    hits = search_policies(vector_store, QUESTION, k=3)

    context_text = "\n\n---\n\n".join(
        f"[{h['metadata']['policy_id']} | topic={h['metadata']['topic']}]\n{h['content']}"
        for h in hits
    )

    print(f"\nRetrieved {len(hits)} chunks.")

    system_prompt = (
        "You are a university IT support assistant.\n"
        "Answer ONLY using the provided context.\n"
        "If the exact information is NOT present in the context, "
        "explicitly say that you do not know the phone number.\n"
        "Do NOT invent phone numbers, emails, or other contact details.\n"
        "Do NOT provide unrelated instructions."
    )

    user_prompt = (
        f"Context:\n{context_text}\n\n"
        f"Question: {QUESTION}\n\n"
        "Answer using ONLY the context above."
    )

    response = chat(
        model="qwen3:0.6b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    print("\n=== MODEL ANSWER ===")
    print(response.message.content)


if __name__ == "__main__":
    main()