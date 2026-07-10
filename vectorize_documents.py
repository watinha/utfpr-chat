import os
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts.prompt import PromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import LatexTextSplitter

def build_ensemble_retriever():
    tex_dir = './tex'
    tex_files = [os.path.join(tex_dir, f) for f in os.listdir(tex_dir) if f.endswith('.tex')]

    all_docs = []
    for file_path in tex_files:
        loader = TextLoader(file_path)
        docs = loader.load()
        all_docs.extend(docs)

    splitter = LatexTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(all_docs)

    embeddings = OllamaEmbeddings(model="bge-m3")
    vectorstore = InMemoryVectorStore.from_documents(split_docs, embeddings)
    vector_retriever = vectorstore.as_retriever()
    bm25_retriever = BM25Retriever.from_documents(split_docs)
    ensemble_retriever = EnsembleRetriever(retrievers=[vector_retriever, bm25_retriever])

    llm_retriever = Ollama(model="llama3.2:3b", temperature=0.1)
    template_academico = """Você é um assistente acadêmico de Inteligência Artificial.
    Sua tarefa é ajudar a encontrar documentos oficiais do curso de Bacharelado em Ciência de Dados e IA.
    O usuário fará uma pergunta curta. Você deve gerar 3 versões diferentes dessa pergunta para melhorar a busca no banco de dados vetorial (que contém ementas, PPC e regulamentos).

    Use sinônimos acadêmicos adequados (ex: 'matéria' -> 'disciplina' ou 'unidade curricular', 'estágio' -> 'estágio supervisionado').
    Forneça apenas as perguntas, uma por linha.

    Pergunta Original: {question}
    """

    prompt_expansao = PromptTemplate(
        input_variables=["question"],
        template=template_academico
    )

    # Inicializa o retriever passando o prompt customizado
    multiquery_retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(),
        llm=llm,
        prompt=prompt_expansao
    )

    return multiquery_retriever


