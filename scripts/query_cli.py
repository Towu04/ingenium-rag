from ingenium_rag.api.app import store


def main() -> None:
    question = input("Question: ")
    for result in store.hybrid_search(question):
        print(f"[{result.score:.0f}] {result.chunk.text}")
