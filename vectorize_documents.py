import os
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import LatexTextSplitter

def build_ensemble_retriever():
    tex_dir = './tex'
    tex_files = [os.path.join(tex_dir, f) for f in os.listdir(tex_dir) if f.endswith('.tex')]

    all_docs = []
    for file_path in tex_files:
        loader = TextLoader(file_path)
        docs = loader.load()
        all_docs.extend(docs)

    splitter = LatexTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(all_docs)

    embeddings = OllamaEmbeddings(model="bge-m3")
    vectorstore = InMemoryVectorStore.from_documents(split_docs, embeddings)
    vector_retriever = vectorstore.as_retriever()
    bm25_retriever = BM25Retriever.from_documents(split_docs)
    ensemble_retriever = EnsembleRetriever(retrievers=[vector_retriever, bm25_retriever])
    return ensemble_retriever


