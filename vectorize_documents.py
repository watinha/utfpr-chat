import os, pickle
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_ollama import OllamaLLM
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
    if os.path.exists('./retrievers/vectorstore.json'):
        vectorstore = InMemoryVectorStore.load('./retrievers/vectorstore.json', embeddings)
    else:
        vectorstore = InMemoryVectorStore.from_documents(split_docs, embeddings)
        vectorstore.dump('./retrievers/vectorstore.json')
    vector_retriever = vectorstore.as_retriever()

    if os.path.exists('./retrievers/bm25_retriever.pkl'):
        with open('./retrievers/bm25_retriever.pkl', 'rb') as f:
            bm25_retriever = pickle.load(f)
    else:
        bm25_retriever = BM25Retriever.from_documents(split_docs)
        with open('./retrievers/bm25_retriever.pkl', 'wb') as f:
            pickle.dump(bm25_retriever, f)

    ensemble_retriever = EnsembleRetriever(retrievers=[vector_retriever, bm25_retriever])

    llm_retriever = OllamaLLM(model="llama3.2:3b", temperature=0.1)
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
        retriever=ensemble_retriever,
        llm=llm_retriever,
        prompt=prompt_expansao
    )

    return multiquery_retriever


