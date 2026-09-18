from langchain_core.prompts import PromptTemplate

TABLE_SUMMARY_PROMPT = PromptTemplate(
    input_variables=["table_content"],
    template=(
        "Você é um assistente especialista em análise de documentos.\n"
        "Analise a seguinte tabela extraída de um PDF e seu contexto/legenda e gere um resumo claro, "
        "conciso e estruturado, "
        "destacando os principais dados e informações:\n\n"
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
    """
    Identifica elementos de tabelas, remove o chunk anterior do fluxo individual,
    mescla o chunk anterior com a tabela e utiliza ambos para gerar a descrição da tabela via LLM.
    """
    processed_docs = []
    
    for doc in docs:
        category = str(doc.metadata.get("category", "")).lower()
        element_type = str(doc.metadata.get("element_type", "")).lower()
        is_table = category == "table" or "table" in element_type
        
        if is_table:
            page_num = doc.metadata.get('page_number', 'N/A')
            filename = doc.metadata.get('filename', '')
            print(f"[Tabela Identificada] Documento: {filename} | Página: {page_num}")
            
            # Captura e remove o chunk anterior para mesclá-lo com a tabela
            prev_chunk_text = ""
            if processed_docs:
                prev_doc = processed_docs.pop()
                prev_chunk_text = prev_doc.page_content.strip()
            
            html_content = doc.metadata.get("text_as_html")
            raw_content = doc.page_content
            
            table_raw_data = html_content if html_content else raw_content
            
            # Inclui o chunk anterior junto à tabela para o LLM gerar o resumo/descrição
            if prev_chunk_text:
                table_content_for_llm = f"CONTEXTO DA TABELA / LEGENDA:\n{prev_chunk_text}\n\nCONTEÚDO DA TABELA:\n{table_raw_data}"
            else:
                table_content_for_llm = table_raw_data
                
            summary = summarize_table(table_content_for_llm, llm)
            
            if html_content and html_content != raw_content:
                full_table_data = f"{raw_content}"
            else:
                full_table_data = table_raw_data
            
            # Unifica o chunk anterior, o resumo gerado e os dados completos da tabela no mesmo Document
            content_parts = []
            if prev_chunk_text:
                content_parts.append(f"CONTEXTO DA TABELA / LEGENDA:\n{prev_chunk_text}")
                doc.metadata["table_caption"] = prev_chunk_text
                
            if summary:
                print(f"[Resumo da Tabela Gerado]: {summary[:120]}...")
                content_parts.append(f"RESUMO DA TABELA:\n{summary}")
                doc.metadata["table_summary"] = summary
                
            content_parts.append(f"DADOS COMPLETOS DA TABELA:\n{full_table_data}")
            
            doc.page_content = "\n\n".join(content_parts)
            doc.metadata["is_table"] = True
            
        processed_docs.append(doc)
        
    return processed_docs

