from langchain_ollama import OllamaLLM, OllamaEmbeddings

class OllamaFactory:
    @staticmethod
    def get_llm(model: str = "llama3.2:3b", temperature: float = 0.1, **kwargs) -> OllamaLLM:
        return OllamaLLM(model=model, temperature=temperature, **kwargs)

    @staticmethod
    def get_embeddings(model: str = "bge-m3", **kwargs) -> OllamaEmbeddings:
        return OllamaEmbeddings(model=model, **kwargs)
