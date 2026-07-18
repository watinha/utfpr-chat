from rag.ollama_factory import OllamaFactory

from .vector_retriever import VectorRetrieverBuilder
from .bm25_retriever import BM25RetrieverBuilder
from .ensemble_retriever import EnsembleRetrieverBuilder
from .multiquery_retriever import MultiQueryRetrieverBuilder

def build_ensemble_retriever(split_docs=[]):
    embeddings = OllamaFactory.get_embeddings(model="bge-m3")
    vector_retriever = VectorRetrieverBuilder.build(split_docs, embeddings)
    bm25_retriever = BM25RetrieverBuilder.build(split_docs)
    ensemble_retriever = EnsembleRetrieverBuilder.build([vector_retriever, bm25_retriever])

    return ensemble_retriever
