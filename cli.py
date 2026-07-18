from rag import rag_query
from misc import make_joke, make_invitation


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
            with open('./contexts.log', 'wt') as f:
                f.seek(0)
                f.truncate()
                f.write('Contexts:\n\n')
                for i, doc in enumerate(contexts):
                    meta = getattr(doc, 'metadata', {})
                    content = getattr(doc, 'page_content', str(doc))
                    f.write(f"  - ({i}) [Metadata: {meta}]\n    Content: {content}\n")
        print('\n')
        invitation = make_invitation(answer)
        print(f' - Invitation: {invitation}')
        print('\n')
        joke = make_joke(question)
        print(f' - Joke: {joke}')
        print('\n\n')

if __name__ == "__main__":
    main()


