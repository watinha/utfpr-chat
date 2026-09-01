import os
from langchain_community.document_loaders import UnstructuredPDFLoader
from retrievers import build_ensemble_retriever
from llms import OllamaFactory
from langchain_core.prompts import PromptTemplate

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 750

TABLE_SUMMARY_PROMPT = PromptTemplate(
    input_variables=["table_content"],
    template=(
        "Você é um assistente especialista em análise de documentos.\n"
        "Analise a seguinte tabela extraída de um PDF e gere um resumo claro, conciso e estruturado "
        "destacando as principais informações, colunas, métricas e dados relevantes:\n\n"
        "Tabela:\n{table_content}\n\n"
        "Resumo:"
    )
)

def summarize_table(table_content: str, llm) -> str:
    """Gera um resumo textual para o conteúdo de uma tabela usando o LLM."""
    if not table_content or len(table_content.strip()) < 10:
        return ""
    
    prompt = TABLE_SUMMARY_PROMPT.format(table_content=table_content)
    try:
        summary = llm.invoke(prompt)
        if isinstance(summary, str):
            return summary.strip()
        return getattr(summary, 'content', str(summary)).strip()
    except Exception as e:
        print(f"Erro ao gerar resumo da tabela com LLM: {e}")
        return ""

def process_table_documents(docs, llm):
    """Identifica elementos de tabelas, gera resumos usando o LLM e atualiza os documentos."""
    processed_docs = []
    
    for doc in docs:
        category = str(doc.metadata.get("category", "")).lower()
        element_type = str(doc.metadata.get("element_type", "")).lower()
        is_table = category == "table" or "table" in element_type
        
        if is_table:
            table_content = doc.metadata.get("text_as_html") or doc.page_content
            summary = summarize_table(table_content, llm)
            
            if summary:
                doc.page_content = (
                    f"RESUMO DA TABELA:\n{summary}\n\n"
                    f"DADOS DA TABELA:\n{doc.page_content}"
                )
                doc.metadata["is_table"] = True
                doc.metadata["table_summary"] = summary
        
        processed_docs.append(doc)
        
    return processed_docs

def load_and_split_documents(pdf_dir: str = './docs'):
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir, exist_ok=True)
        
    pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    
    llm = OllamaFactory.get_llm(model="llama3.2:3b", temperature=0.1)
    
    all_docs = []
    for file_path in pdf_files:
        print(f"Carregando e processando PDF: {file_path}")
        loader = UnstructuredPDFLoader(
            file_path, 
            mode="elements",
            chunking_strategy="by_title",
            max_characters=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
            languages=["pt"]
        )
        docs = loader.load()
        docs = process_table_documents(docs, llm)
        all_docs.extend(docs)
        
    return all_docs

if __name__ == "__main__":
    split_docs = load_and_split_documents()
    build_ensemble_retriever(split_docs)

