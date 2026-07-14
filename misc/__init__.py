from rag import OllamaFactory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def make_joke(question):
    llm = OllamaFactory.get_llm(model="llama3.2:3b", temperature=0.7)
    system_prompt = f"Generate a joke about: {question}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{pergunta}")
    ])

    chain_joke = prompt | llm | StrOutputParser()
    joke = chain_joke.invoke({ 'pergunta': question })
    return joke


def make_invitation(response):
    llm = OllamaFactory.get_llm(model="llama3.2:3b", temperature=0.7)
    system_prompt = "Você é um assistente acadêmico. Com base nas informações da resposta fornecida, escreva um convite simples e objetivo (máximo de 50 palavras) para o estudante ingressar no curso de Bacharelado em Ciência de Dados e Inteligência Artificial da nossa universidade, para aprender mais sobre a área."

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Resposta anterior: {resposta}\n\nEscreva o convite:")
    ])

    chain_invitation = prompt | llm | StrOutputParser()
    invitation = chain_invitation.invoke({ 'resposta': response })
    return invitation
