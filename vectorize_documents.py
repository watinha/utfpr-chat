import os
from langchain_community.document_loaders.text import TextLoader
from langchain_text_splitters import LatexTextSplitter
from pylatexenc.latex2text import LatexNodes2Text
from retrievers import build_ensemble_retriever

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 300

def load_and_split_documents(tex_dir: str = './tex'):
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
    return splitter.split_documents(all_docs)

if __name__ == "__main__":
    split_docs = load_and_split_documents()
    build_ensemble_retriever(split_docs)
