import re
from langchain_core.prompts import PromptTemplate

SAMPLE_SECTION_SIZE = 4000
SAMPLE_CHUNK_SIZE = 2000
MIN_CHUNK_LEN_FOR_LLM_CONTEXT = 150

NUMBERED_HEADING_PATTERN = re.compile(
    r'^(?:'
    r'(\d+(?:\.\d+)*)\.?\s+[A-ZÀ-Úa-zà-ú]'  # 1., 1.2, 1.2.3 Title
    r'|(?:CAPÍTULO|CAPITULO|SEÇÃO|SECAO|ITEM|ANEXO|APÊNDICE|APENDICE)\s+[0-9A-ZIVXLCDM]+\.?\b'  # CAPÍTULO I, SEÇÃO 2
    r')',
    re.IGNORECASE
)

CHUNK_CONTEXT_PROMPT = PromptTemplate(
    input_variables=["doc_title", "section_path", "section_content", "chunk_content"],
    template=(
        "Aqui está um trecho de um documento inserido na subseção/seção específica '{section_path}' do documento '{doc_title}'.\n\n"
        "Conteúdo da subseção/seção:\n{section_content}\n\n"
        "Trecho a ser contextualizado:\n{chunk_content}\n\n"
        "Escreva um parágrafo curto (2-3 frases) contextualizando exatamente sobre o que trata este trecho específico no âmbito desta subseção e do documento:\n\n"
        "Parágrafo de Contexto:"
    )
)

def detect_heading_level(content: str, category: str):
    """Detecta se o texto é um título/cabeçalho de seção ou subseção e calcula seu nível hierárquico (1, 2, 3...)."""
    text = content.strip()
    if not text or len(text) > 250:
        return False, 0
    
    cat = category.lower()
    is_unstructured_header = cat in ("title", "header", "subtitle", "subheader")
    
    match = NUMBERED_HEADING_PATTERN.match(text)
    if match:
        num_part = match.group(1)
        if num_part:
            level = num_part.count('.') + 1
            return True, level
        return True, 1
        
    if is_unstructured_header:
        if text.isupper() or len(text) < 80:
            return True, 1
            
    return False, 0

def generate_chunk_context(doc_title: str, section_path: str, section_content: str, chunk_content: str, llm) -> str:
    """Gera uma descrição em parágrafo do contexto de um chunk utilizando o conteúdo completo da subseção."""
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

def apply_contextual_chunking(docs, llm, document_title: str = ""):
    """
    Enriquece cada chunk de documento com o contexto da seção/subseção em que está inserido
    e uma descrição em parágrafo gerada por LLM utilizando o conteúdo da seção.
    """
    doc_section_info = []
    section_chunks = {}
    current_section_stack = []
    
    # Passo 1: Determina o caminho hierárquico da seção/subseção para cada chunk e agrupa o conteúdo
    for doc in docs:
        category = str(doc.metadata.get("category", "")).lower()
        content = doc.page_content.strip()
        
        unstructured_sections = doc.metadata.get("sections") or doc.metadata.get("section")
        
        is_heading, level = detect_heading_level(content, category)
        if is_heading and level > 0:
            current_section_stack = current_section_stack[:level - 1]
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
        
    complete_section_contents = {
        sec_path: "\n\n".join(chunks) for sec_path, chunks in section_chunks.items()
    }
    
    # Passo 2: Gera a descrição contextual para cada chunk utilizando o conteúdo completo da subseção
    processed_docs = []
    doc_label = document_title if document_title else "Documento PDF"
    
    for doc, section_path in doc_section_info:
        full_section_text = complete_section_contents.get(section_path, "")
        
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
