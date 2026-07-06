from langchain_community.llms import Ollama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.load import loads


# Load the saved ensemble retriever
with open('ensemble_retriever.json', 'rb') as f:
    retriever = loads(f.read())

# Initialize Ollama LLM
llm = Ollama(model="llama3.2:3b", temperature=0.1)

# Create a prompt template for RAG with system and human messages
prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um assistente acadêmico especializado no curso de Ciências de Dados e Inteligência Artificial. Suas respostas devem ser baseadas estritamente no contexto fornecido e devem ser as mais precisas, claras e objetivas possível. Caso o contexto não contenha a resposta, afirme que não sabe a resposta."),
    ("human", "Contexto:\n{context}\n\nPergunta: {input}\nResposta:")
])

# Create the document chain
doc_chain = create_stuff_documents_chain(llm, prompt)

# Create the retrieval chain
rag_chain = create_retrieval_chain(retriever, doc_chain)

def rag_query(question: str):
    """Query the retriever-augmented generation pipeline."""
    result = rag_chain({"input": question})
    return result['answer'], result.get('context', None)

if __name__ == "__main__":
    question = input("Enter your question: ")
    answer, context = rag_query(question)
    print("Answer:", answer)
    if context:
        print("Context:", context)
