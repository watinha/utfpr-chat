import os

from langchain_community.document_loaders import UnstructuredPDFLoader
from retrievers import build_ensemble_retriever
from llms import OllamaFactory
from langchain_core.prompts import PromptTemplate

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 500
SAMPLE_SECTION_SIZE = 4000
SAMPLE_CHUNK_SIZE = CHUNK_SIZE
MIN_CHUNK_LEN_FOR_LLM_CONTEXT = 150

TABLE_SUMMARY_PROMPT = PromptTemplate(
    input_variables=["table_content"],
    template=(
        "Você é um assistente especialista em análise de documentos.\n"
        "Analise a seguinte tabela extraída de um PDF e gere um resumo claro, conciso e estruturado "
        "destacando os principais dados e informações:\n\n"
        "Tabela:\n{table_content}\n\n"
        "Resumo:"
    )
)

CHUNK_CONTEXT_PROMPT = PromptTemplate(
    input_variables=["doc_title", "section_path", "section_content", "chunk_content"],
    template=(
        "Aqui está um trecho de um documento na seção '{section_path}' do documento '{doc_title}'.\n\n"
        "Conteúdo completo da seção:\n{section_content}\n\n"
        "Trecho a ser contextualizado:\n{chunk_content}\n\n"
        "Escreva um parágrafo curto (2-3 frases) contextualizando o trecho acima no âmbito do conteúdo completo da seção e do documento:\n\n"
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

def generate_chunk_context(doc_title: str, section_path: str, section_content: str, chunk_content: str, llm) -> str:
    """Gera uma descrição em parágrafo do contexto de um chunk utilizando o conteúdo completo da seção."""
    if not chunk_content or len(chunk_content.strip()) < MIN_CHUNK_LEN_FOR_LLM_CONTEXT:
        return ""
    
    sample_section = section_content[:SAMPLE_SECTION_SIZE]
    sample_chunk = chunk_content[:SAMPLE_CHUNK_SIZE]
    prompt = CHUNK_CONTEXT_PROMPT.format(
        doc_title=doc_title,
        section_path=section_path,
        section_content=sample_section,
        chunk_content=sample_chunk
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
    """Identifica elementos de tabelas, gera resumos usando o LLM e preserva os dados completos da tabela (HTML e Texto)."""
    processed_docs = []
    
    for doc in docs:
        category = str(doc.metadata.get("category", "")).lower()
        element_type = str(doc.metadata.get("element_type", "")).lower()
        is_table = category == "table" or "table" in element_type
        
        if is_table:
            page_num = doc.metadata.get('page_number', 'N/A')
            filename = doc.metadata.get('filename', '')
            print(f"[Tabela Identificada] Documento: {filename} | Página: {page_num}")
            
            html_content = doc.metadata.get("text_as_html")
            raw_content = doc.page_content
            
            # Utiliza HTML para estruturação visual da tabela se disponível
            table_content = html_content if html_content else raw_content
            summary = summarize_table(table_content, llm)
            
            full_table_data = f"{raw_content}"
            else:
                full_table_data = table_content
            
            if summary:
                print(f"[Resumo da Tabela Gerado]: {summary[:120]}...")
                doc.page_content = (
                    f"RESUMO DA TABELA:\n{summary}\n\n"
                    f"DADOS COMPLETOS DA TABELA:\n{full_table_data}"
                )
                doc.metadata["is_table"] = True
                doc.metadata["table_summary"] = summary
            else:
                doc.page_content = f"DADOS COMPLETOS DA TABELA:\n{full_table_data}"
        
        processed_docs.append(doc)
        
    return processed_docs

def apply_contextual_chunking(docs, llm, document_title: str = ""):
    """
    Enriquece cada chunk de documento com o contexto da seção em que está inserido
    e uma descrição em parágrafo gerada por LLM utilizando o conteúdo completo da seção.
    """
    doc_section_info = []
    section_chunks = {}
    current_section_stack = []
    
    # Passo 1: Determina o caminho da seção para cada chunk e agrupa o conteúdo completo de cada seção
    for doc in docs:
        category = str(doc.metadata.get("category", "")).lower()
        element_type = str(doc.metadata.get("element_type", "")).lower()
        content = doc.page_content.strip()
        
        unstructured_sections = doc.metadata.get("sections") or doc.metadata.get("section")
        
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
            
        doc_section_info.append((doc, section_path))
        
        if section_path not in section_chunks:
            section_chunks[section_path] = []
        section_chunks[section_path].append(doc.page_content)
        
    # Constrói o texto completo de cada seção
    complete_section_contents = {
        sec_path: "\n\n".join(chunks) for sec_path, chunks in section_chunks.items()
    }
    
    # Passo 2: Gera a descrição contextual para cada chunk utilizando o conteúdo completo da seção
    processed_docs = []
    doc_label = document_title if document_title else "Documento PDF"
    
    for doc, section_path in doc_section_info:
        full_section_text = complete_section_contents.get(section_path, "")
        
        # Gera descrição do contexto em parágrafo via LLM apenas para chunks com conteúdo substancial e não-tabelas
        context_paragraph = ""
        if len(doc.page_content.strip()) >= MIN_CHUNK_LEN_FOR_LLM_CONTEXT and not doc.metadata.get("is_table"):
            context_paragraph = generate_chunk_context(
                doc_title=doc_label,
                section_path=section_path,
                section_content=full_section_text,
                chunk_content=doc.page_content,
                llm=llm
            )
        
        doc.metadata["section"] = section_path
        doc.metadata["section_context"] = f"Documento: {doc_label} | Seção: {section_path}"
        if context_paragraph:
            doc.metadata["context_paragraph"] = context_paragraph
            
        context_prefix = f"CONTEXTO DA SEÇÃO: [{doc_label} > {section_path}]"
        if context_paragraph:
            doc.page_content = f"{context_prefix}\nDESCRIÇÃO DO CONTEXTO: {context_paragraph}\n\n--- CONTEÚDO INTEGRAL DO CHUNK ---\n{doc.page_content}"
        else:
            doc.page_content = f"{context_prefix}\n\n--- CONTEÚDO INTEGRAL DO CHUNK ---\n{doc.page_content}"
            
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
            strategy="hi_res",
            infer_table_structure=True,
            chunking_strategy="by_title",
            max_characters=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
            combine_under_n_chars=500,
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



