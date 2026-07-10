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


def llm_classification(question: str):
	prompt = ChatPromptTemplate.from_messages([
		("system", prompt_data["classification"]),
		("user", "{pergunta}")
	])

	chain_classificacao = prompt | llm | StrOutputParser()
	resposta_bruta = chain_classificacao.invoke({"pergunta": pergunta_usuario})
	decisao = resposta_bruta.strip().upper()

	if "SIM" in decisao:
		return True
	else:
		return False


def rag_query(question: str):
	in_scope = llm_classification(question)
	if not in_scope:
		return 'Essa pergunta não está no escopo das minhas atividades como assistente...'

    result = rag_chain.invoke({"input": question})
    return result['answer'], result.get('context', None)

if __name__ == "__main__":
    question = input("Enter your question: ")
    answer, context = rag_query(question)
    print("Answer:", answer)
    if context:
        print("Context:", context)
