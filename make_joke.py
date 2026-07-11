import sys
from langchain_ollama import OllamaLLM

def make_joke(question):
    llm = OllamaLLM(model="llama3.2:3b", temperature=0.7)
    prompt = f"Generate a joke about: {question}"
    joke = llm(prompt)
    return joke


