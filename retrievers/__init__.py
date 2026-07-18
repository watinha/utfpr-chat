import os
from langchain_community.document_loaders.text import TextLoader
from langchain_text_splitters import LatexTextSplitter
from pylatexenc.latex2text import LatexNodes2Text
from rag.ollama_factory import OllamaFactory

from .vector_retriever import VectorRetrieverBuilder
from .bm25_retriever import BM25RetrieverBuilder
from .ensemble_retriever import EnsembleRetrieverBuilder
from .multiquery_retriever import MultiQueryRetrieverBuilder

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 300

def build_ensemble_retriever(tex_dir: str = './tex'):
    tex_files = [os.path.join(tex_dir, f) for f in os.listdir(tex_dir) if f.endswith('.tex')]
    conversor = LatexNodes2Text()

    all_docs = []
    for file_path in tex_files:
        loader = TextLoader(file_path)
        docs = loader.load()

        if 'unidades-' in file_path:
            for doc in docs:
                doc.page_content = conversor.latex_to_text(doc.page_content)

        all_docs.extend(docs)

    splitter = LatexTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    split_docs = splitter.split_documents(all_docs)

    embeddings = OllamaFactory.get_embeddings(model="bge-m3")
    vector_retriever = VectorRetrieverBuilder.build(split_docs, embeddings)
    bm25_retriever = BM25RetrieverBuilder.build(split_docs)
    ensemble_retriever = EnsembleRetrieverBuilder.build([vector_retriever, bm25_retriever])

    return ensemble_retriever
