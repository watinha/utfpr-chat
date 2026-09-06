from langchain_core.prompts import PromptTemplate

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
            
            table_content = html_content if html_content else raw_content
            summary = summarize_table(table_content, llm)
            
            if html_content and html_content != raw_content:
                full_table_data = f"ESTRUTURA HTML DA TABELA:\n{html_content}\n\nTEXTO DA TABELA:\n{raw_content}"
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
