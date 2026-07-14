import os
from langchain_community.document_loaders.text import TextLoader
from langchain_text_splitters import LatexTextSplitter
from pylatexenc.latex2text import LatexNodes2Text
from rag import OllamaFactory
from retrievers import (
    VectorRetrieverBuilder,
    BM25RetrieverBuilder,
    EnsembleRetrieverBuilder,
    MultiQueryRetrieverBuilder,
)


def build_ensemble_retriever():
    tex_dir = './tex'
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

    splitter = LatexTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(all_docs)

    embeddings = OllamaFactory.get_embeddings(model="bge-m3")
    vector_retriever = VectorRetrieverBuilder.build(split_docs, embeddings)
    bm25_retriever = BM25RetrieverBuilder.build(split_docs)
    ensemble_retriever = EnsembleRetrieverBuilder.build([vector_retriever, bm25_retriever])

    #multiquery_retriever = MultiQueryRetrieverBuilder.build(
    #    retriever=ensemble_retriever
    #)

    #return multiquery_retriever
    return ensemble_retriever


if __name__ == "__main__":
    build_ensemble_retriever()


