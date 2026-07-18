import os
from langchain_community.document_loaders import UnstructuredPDFLoader
from retrievers import build_ensemble_retriever

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 300

def load_and_split_documents(pdf_dir: str = './docs'):
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir, exist_ok=True)
        
    pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    
    all_docs = []
    for file_path in pdf_files:
        loader = UnstructuredPDFLoader(
            file_path, 
            mode="elements",
            chunking_strategy="by_title",
            max_characters=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
            languages=["pt"]
        )
        docs = loader.load()
        all_docs.extend(docs)
        
    return all_docs

if __name__ == "__main__":
    split_docs = load_and_split_documents()
    build_ensemble_retriever(split_docs)
