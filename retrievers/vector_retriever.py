import os
from langchain_community.vectorstores import FAISS

class VectorRetrieverBuilder:
    @staticmethod
    def build(split_docs, embeddings, store_path: str = './retrievers/faiss_index'):
        if os.path.exists(store_path) and os.listdir(store_path):
            vectorstore = FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)
        else:
            vectorstore = FAISS.from_documents(split_docs, embeddings)
            vectorstore.save_local(store_path)
        return vectorstore.as_retriever(search_kwargs={"k": 2})
