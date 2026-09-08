import os
from langchain_community.document_loaders import UnstructuredPDFLoader
from retrievers import build_ensemble_retriever
from llms import OllamaFactory
from chunking import process_table_documents, apply_contextual_chunking

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 500

def load_and_split_documents(pdf_dir: str = './docs'):
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir, exist_ok=True)
        
    pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    
    llm = OllamaFactory.get_llm(model="llama3.2:3b", temperature=0.1)
    
    all_docs = []
    for file_path in pdf_files:
        filename = os.path.basename(file_path)
        print(f"Carregando e processando PDF: {file_path}")
        loader = UnstructuredPDFLoader(
            file_path, 
            mode="elements",
            strategy="hi_res",
            infer_table_structure=True,
            chunking_strategy="by_title",
            max_characters=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
            languages=["pt"]
        )
        docs = loader.load()
        docs = process_table_documents(docs, llm)
        docs = apply_contextual_chunking(docs, llm, document_title=filename, chunk_size=CHUNK_SIZE)
        all_docs.extend(docs)
        
    return all_docs

if __name__ == "__main__":
    split_docs = load_and_split_documents()
    build_ensemble_retriever(split_docs)
