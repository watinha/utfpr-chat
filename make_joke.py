import sys
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def make_joke(question):
    llm = OllamaLLM(model="llama3.2:3b", temperature=0.7)
    system_prompt = f"Generate a joke about: {question}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{pergunta}")
    ])

    chain_joke = prompt | llm | StrOutputParser()
    joke = chain_joke.invoke({ 'pergunta': question })
    return joke


