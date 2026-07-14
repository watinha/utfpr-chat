from langchain_classic.retrievers import EnsembleRetriever

class EnsembleRetrieverBuilder:
    @staticmethod
    def build(retrievers) -> EnsembleRetriever:
        return EnsembleRetriever(retrievers=retrievers)
