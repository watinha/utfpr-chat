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

CHUNK_CONTEXT_PROMPT = PromptTemplate(
    input_variables=["doc_title", "section_path", "chunk_content"],
    template=(
        "Aqui está um trecho de um documento na seção '{section_path}' do documento '{doc_title}'.\n"
        "Escreva um parágrafo curto (2-3 frases) contextualizando sobre o que trata este trecho no âmbito da seção e do documento:\n\n"
        "Trecho:\n{chunk_content}\n\n"
        "Parágrafo de Contexto:"
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

def generate_chunk_context(doc_title: str, section_path: str, chunk_content: str, llm) -> str:
    """Gera uma descrição em parágrafo do contexto de um chunk usando o LLM."""
    if not chunk_content or len(chunk_content.strip()) < 15:
        return ""
    
    sample_content = chunk_content[:1500]
    prompt = CHUNK_CONTEXT_PROMPT.format(
        doc_title=doc_title,
        section_path=section_path,
        chunk_content=sample_content
    )
    try:
        response = llm.invoke(prompt)
        if isinstance(response, str):
            return response.strip()
        return getattr(response, 'content', str(response)).strip()
    except Exception as e:
        print(f"Erro ao gerar descrição de contexto do chunk com LLM: {e}")
        return ""

def process_table_documents(docs, llm):
    """Identifica elementos de tabelas, gera resumos usando o LLM e atualiza os documentos."""
    processed_docs = []
    
    for doc in docs:
        category = str(doc.metadata.get("category", "")).lower()
        element_type = str(doc.metadata.get("element_type", "")).lower()
        is_table = category == "table" or "table" in element_type
        
        if is_table:
            print(f"[Tabela Identificada] Documento: {doc.metadata.get('filename', '')} | Página: {doc.metadata.get('page_number', 'N/A')}")
            table_content = doc.metadata.get("text_as_html") or doc.page_content
            summary = summarize_table(table_content, llm)
            
            if summary:
                print(f"[Resumo da Tabela Gerado]: {summary[:120]}...")
                doc.page_content = (
                    f"RESUMO DA TABELA:\n{summary}\n\n"
                    f"DADOS DA TABELA:\n{doc.page_content}"
                )
                doc.metadata["is_table"] = True
                doc.metadata["table_summary"] = summary
        
        processed_docs.append(doc)
        
    return processed_docs


def apply_contextual_chunking(docs, llm, document_title: str = ""):
    """
    Enriquece cada chunk de documento com o contexto da seção em que está inserido
    e uma descrição em parágrafo gerada por LLM (Contextual Chunking).
    """
    processed_docs = []
    current_section_stack = []
    
    for doc in docs:
        category = str(doc.metadata.get("category", "")).lower()
        element_type = str(doc.metadata.get("element_type", "")).lower()
        content = doc.page_content.strip()
        
        # Verifica se o Unstructured já forneceu lista de seções nos metadados
        unstructured_sections = doc.metadata.get("sections") or doc.metadata.get("section")
        
        # Atualiza a pilha de seções se o elemento for um Título/Cabeçalho
        if (category in ("title", "header") or "title" in element_type or "header" in element_type) and content:
            if len(content) < 300:
                category_depth = doc.metadata.get("category_depth", 0)
                if isinstance(category_depth, int) and category_depth < len(current_section_stack):
                    current_section_stack = current_section_stack[:category_depth]
                    current_section_stack.append(content)
                else:
                    if category == "title" and not current_section_stack:
                        current_section_stack = [content]
                    else:
                        current_section_stack.append(content)
        
        # Constrói o caminho da seção
        if unstructured_sections:
            if isinstance(unstructured_sections, list):
                section_path = " > ".join([str(s) for s in unstructured_sections if str(s).strip()])
            else:
                section_path = str(unstructured_sections)
        elif current_section_stack:
            dedup_sections = []
            for sec in current_section_stack:
                if not dedup_sections or dedup_sections[-1] != sec:
                    dedup_sections.append(sec)
            section_path = " > ".join(dedup_sections)
        else:
            section_path = "Geral"
            
        doc_label = document_title if document_title else "Documento PDF"
        
        # Gera descrição do contexto em parágrafo via LLM
        context_paragraph = generate_chunk_context(
            doc_title=doc_label,
            section_path=section_path,
            chunk_content=doc.page_content,
            llm=llm
        )
        
        # Armazena metadados
        doc.metadata["section"] = section_path
        doc.metadata["section_context"] = f"Documento: {doc_label} | Seção: {section_path}"
        if context_paragraph:
            doc.metadata["context_paragraph"] = context_paragraph
            
        # Formata o conteúdo do chunk preapendando o contexto e a descrição gerada pelo LLM
        context_prefix = f"CONTEXTO DA SEÇÃO: [Documento: {doc_label} | Seção: {section_path}]"
        if context_paragraph:
            context_prefix += f"\nDESCRIÇÃO DO CONTEXTO: {context_paragraph}"
            
        doc.page_content = f"{context_prefix}\n\nCONTEÚDO:\n{doc.page_content}"
        processed_docs.append(doc)
        
    return processed_docs

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
            chunking_strategy="by_title",
            max_characters=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
            languages=["pt"]
        )
        docs = loader.load()
        docs = process_table_documents(docs, llm)
        docs = apply_contextual_chunking(docs, llm, document_title=filename)
        all_docs.extend(docs)
        
    return all_docs

if __name__ == "__main__":
    split_docs = load_and_split_documents()
    build_ensemble_retriever(split_docs)


