import json

from langchain_community.llms import Ollama
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.load import loads

from vectorize_documents import build_ensemble_retriever


# Load the ensemble retriever using the builder function
retriever = build_ensemble_retriever()

# Initialize Ollama LLM
llm = Ollama(model="llama3.2:3b", temperature=0.1)

with open('rag_prompt.txt', 'r', encoding='utf-8') as f:
    prompt_data = json.load(f)

prompt = ChatPromptTemplate.from_messages([
    ("system", prompt_data["system"]),
    ("human", prompt_data["human"])
])

# Create the document chain
doc_chain = create_stuff_documents_chain(llm, prompt)

# Create the retrieval chain
rag_chain = create_retrieval_chain(retriever, doc_chain)

def rag_query(question: str):
    result = rag_chain.invoke({"input": question})
    return result['answer'], result.get('context', None)

if __name__ == "__main__":
    question = input("Enter your question: ")
    answer, context = rag_query(question)
    print("Answer:", answer)
    if context:
        print("Context:", context)
