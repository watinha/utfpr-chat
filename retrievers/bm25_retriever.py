import os
import pickle
from langchain_community.retrievers import BM25Retriever

class BM25RetrieverBuilder:
    @staticmethod
    def build(split_docs, cache_path: str = './retrievers/bm25_retriever.pkl'):
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                bm25_retriever = pickle.load(f)
        else:
            bm25_retriever = BM25Retriever.from_documents(split_docs)
            with open(cache_path, 'wb') as f:
                pickle.dump(bm25_retriever, f)
        bm25_retriever.k = 2
        return bm25_retriever
