<<<<<<< HEAD
from ingenium_rag.api.app import pipeline
=======
from ingenium_rag.api.app import store
>>>>>>> d8558de83a60ee17885c0c11a7d88e8a92704c68


def main() -> None:
    question = input("Question: ")
<<<<<<< HEAD
    answer = pipeline.run(question)
    print(answer.text)
=======
    for result in store.hybrid_search(question):
        print(f"[{result.score:.0f}] {result.chunk.text}")
>>>>>>> d8558de83a60ee17885c0c11a7d88e8a92704c68
