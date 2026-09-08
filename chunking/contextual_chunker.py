from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

SAMPLE_SECTION_SIZE = 4000
SAMPLE_CHUNK_SIZE = 2000
MIN_CHUNK_LEN_FOR_LLM_CONTEXT = 150

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

def generate_chunk_context(doc_title: str, section_path: str, section_content: str, chunk_content: str, llm) -> str:
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

def remove_header_documents(docs):
    """Remove da lista os documentos que possuem a categoria 'Header' (ex: números de página e cabeçalhos)."""
    return [
        doc for doc in docs
        if str((doc.metadata or {}).get("category", "") or "").strip().lower() != "header"
    ]

def join_chunks_by_size(docs, chunk_size: int = SAMPLE_CHUNK_SIZE):
    """
    Gera uma nova lista de documentos unindo chunks sequencialmente até que
    o tamanho especificado (chunk_size) seja alcançado.
    Chunks com a categoria 'Title' não são unidos a outros chunks, mas são
    mantidos individualmente na lista final de documentos.
    """
    if not docs:
        return []

    new_docs = []
    current_docs = []
    current_length = 0

    def flush_current():
        nonlocal current_docs, current_length
        if not current_docs:
            return
        if len(current_docs) == 1:
            new_docs.append(current_docs[0])
        else:
            combined_text = "\n\n".join(d.page_content for d in current_docs)
            combined_metadata = dict(current_docs[0].metadata)
            new_docs.append(Document(page_content=combined_text, metadata=combined_metadata))
        current_docs = []
        current_length = 0

    for doc in docs:
        category = str((doc.metadata or {}).get("category", "") or "").strip().lower()
        # Chunks com categoria Title permanecem na lista, mas individuais (sem junção)
        if category == "title":
            flush_current()
            new_docs.append(doc)
            continue

        current_docs.append(doc)
        current_length += len(doc.page_content)

        if current_length >= chunk_size:
            flush_current()

    flush_current()

    return new_docs


def apply_contextual_chunking(docs, llm, document_title: str = "", chunk_size: int = SAMPLE_CHUNK_SIZE):
    """
    Aplica contextual chunking aos documentos:
    1. Remove cabeçalhos irrelevantes (Header).
    2. Agrupa chunks sequencialmente até atingir chunk_size (mantendo Titles individuais).
    3. Itera pelos documentos resultantes armazenando o último Title como nome e o conteúdo
       da seção para gerar o contexto dos demais chunks via LLM.
    """
    docs = remove_header_documents(docs)
    docs = join_chunks_by_size(docs, chunk_size=chunk_size)

    processed_docs = []
    doc_label = document_title if document_title else "Documento PDF"

    current_title_doc = None
    current_section_name = "Geral"
    current_section_docs = []

    def process_section(title_doc, section_name, section_docs):
        if title_doc is not None:
            title_doc.metadata["section"] = section_name
            processed_docs.append(title_doc)

        if not section_docs:
            return

        section_content = "\n\n".join(d.page_content for d in section_docs)

        for doc in section_docs:
            context_paragraph = ""
            if llm and len(doc.page_content.strip()) >= MIN_CHUNK_LEN_FOR_LLM_CONTEXT and not doc.metadata.get("is_table"):
                context_paragraph = generate_chunk_context(
                    doc_title=doc_label,
                    section_path=section_name,
                    section_content=section_content,
                    chunk_content=doc.page_content,
                    llm=llm
                )

            doc.metadata["section"] = section_name
            doc.metadata["section_context"] = f"Documento: {doc_label} | Seção: {section_name}"
            if context_paragraph:
                doc.metadata["context_paragraph"] = context_paragraph

            context_prefix = f"CONTEXTO DA SEÇÃO: [{doc_label} > {section_name}]"
            if context_paragraph:
                doc.page_content = f"{context_prefix}\nDESCRIÇÃO DO CONTEXTO: {context_paragraph}\n\n--- CONTEÚDO INTEGRAL DO CHUNK ---\n{doc.page_content}"
            else:
                doc.page_content = f"{context_prefix}\n\n--- CONTEÚDO INTEGRAL DO CHUNK ---\n{doc.page_content}"

            processed_docs.append(doc)

    for doc in docs:
        category = str((doc.metadata or {}).get("category", "") or "").strip().lower()
        if category == "title":
            # Ao encontrar uma nova seção, processa os chunks acumulados da seção anterior
            process_section(current_title_doc, current_section_name, current_section_docs)
            current_title_doc = doc
            current_section_name = doc.page_content.strip() or "Geral"
            current_section_docs = []
        else:
            current_section_docs.append(doc)

    # Processa os chunks da última seção
    process_section(current_title_doc, current_section_name, current_section_docs)

    return processed_docs



