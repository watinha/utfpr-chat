from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts.prompt import PromptTemplate
from rag import OllamaFactory

class MultiQueryRetrieverBuilder:
    @staticmethod
    def build(retriever, model: str = "llama3.2:3b", temperature: float = 0.1) -> MultiQueryRetriever:
        llm_retriever = OllamaFactory.get_llm(model=model, temperature=temperature)
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

        return MultiQueryRetriever.from_llm(
            retriever=retriever,
            llm=llm_retriever,
            prompt=prompt_expansao
        )
