import os
from langchain_core.vectorstores import InMemoryVectorStore

class VectorRetrieverBuilder:
    @staticmethod
    def build(split_docs, embeddings, store_path: str = './retrievers/vectorstore.json'):
        if os.path.exists(store_path):
            vectorstore = InMemoryVectorStore.load(store_path, embeddings)
        else:
            vectorstore = InMemoryVectorStore.from_documents(split_docs, embeddings)
            vectorstore.dump(store_path)
        return vectorstore.as_retriever()
