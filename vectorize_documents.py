import os, pickle
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import LatexTextSplitter

# List of .tex files
tex_files = [
    './tex/unidades-curriculares.tex',
    './tex/1_Apresentacao.tex',
    './tex/unidades-optativas-humanidades.tex',
    './tex/3_corpo_docente.tex',
    './tex/2_Organização_Didático_Pedagógica.tex',
    './tex/4_infraestrutura.tex',
    './tex/Apendices.tex',
    './tex/unidades-optativas-especificas.tex',
]

all_docs = []
for file_path in tex_files:
    loader = TextLoader(file_path)
    docs = loader.load()
    all_docs.extend(docs)

# Split documents using LatexTextSplitter
splitter = LatexTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = splitter.split_documents(all_docs)

# Create embeddings
embeddings = OllamaEmbeddings(model="bge-m3")

# Vector store
vectorstore = InMemoryVectorStore.from_documents(split_docs, embeddings)

# Create retrievers
vector_retriever = vectorstore.as_retriever()
bm25_retriever = BM25Retriever.from_documents(split_docs)

# EnsembleRetriever combines both
ensemble_retriever = EnsembleRetriever(retrievers=[vector_retriever, bm25_retriever])

print("EnsembleRetriever created with vector and BM25 retrievers.")

with open('ensemble_retriever.pkl', 'wb') as f:
    pickle.dump(ensemble_retriever, f)

print("EnsembleRetriever saved to ensemble_retriever.pkl.")
