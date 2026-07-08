from rag_query import rag_query

def main():
    print("Interactive RAG Chat. Type 'exit' to quit.")
    while True:
        question = input("You: ")
        if question.strip().lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        answer, context = rag_query(question)
        print("Assistant:", answer)
        if context:
            print("Context:", context)
        print('\n\n')

if __name__ == "__main__":
    main()


