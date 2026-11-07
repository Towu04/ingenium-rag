from api.app import pipeline


def main() -> None:
    question = input("Question: ")
    answer = pipeline.run(question)
    print(answer.text)
