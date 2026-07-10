from rag_query import rag_query

def main():
    print("Interactive RAG Chat. Type 'exit' to quit.")
    while True:
        question = input("You: ")
        if question.strip().lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        answer, contexts = rag_query(question)
        print("Assistant:", answer)
        if contexts:
            print("Context(s):")
            if isinstance(contexts, list):
                for doc in contexts:
                    meta = getattr(doc, 'metadata', {})
                    content = getattr(doc, 'page_content', str(doc))
                    print(f"- [Metadata: {meta}]\n  Content: {content}")
            else:
                print(contexts)
        print('\n\n')

if __name__ == "__main__":
    main()


