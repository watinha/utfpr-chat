import os

from llms import OllamaFactory

from .vector_retriever import VectorRetrieverBuilder
from .bm25_retriever import BM25RetrieverBuilder
from .ensemble_retriever import EnsembleRetrieverBuilder
from .multiquery_retriever import MultiQueryRetrieverBuilder

ENV = os.getenv("ENV", "development")
k = 5 if ENV.lower() == "production" else 3

def build_ensemble_retriever(split_docs=[]):
    embeddings = OllamaFactory.get_embeddings(model="bge-m3")
    vector_retriever = VectorRetrieverBuilder.build(split_docs, embeddings, k=k)
    bm25_retriever = BM25RetrieverBuilder.build(split_docs, k=k)
    ensemble_retriever = EnsembleRetrieverBuilder.build([vector_retriever, bm25_retriever])
    #multiquery_retriever = MultiQueryRetrieverBuilder.build(ensemble_retriever)

    #return multiquery_retriever
    return ensemble_retriever
