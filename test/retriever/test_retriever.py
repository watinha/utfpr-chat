import os, sys, unittest, urllib.request

from rag import rag_query
from retrievers import build_ensemble_retriever


def is_ollama_available():
    try:
        # Check if local Ollama server is running and accessible
        with urllib.request.urlopen("http://localhost:11434/", timeout=2) as response:
            return response.status == 200
    except Exception:
        return False

class TestEnsembleRetriever(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not is_ollama_available():
            raise unittest.SkipTest("Ollama is not running on localhost:11434. Skipping integration retriever tests.")
        
        # Build the actual retriever configured in vectorize_documents
        cls.retriever = build_ensemble_retriever()

    def test_retriever_chunks_presence(self):
        """Validates that specific queries retrieve documents containing the expected target chunks."""
        # List of question/term check cases
        test_cases = [
            {
                "question": "Qual a qualificação dos docentes?",
                "expected_substrings": ["docente", "corpo docente", "titulação", "professor"],
                "source_file": "3_corpo_docente.tex"
            },
            {
                "question": "Como é a infraestrutura do curso?",
                "expected_substrings": ["infraestrutura", "laboratório", "equipamentos", "bloco"],
                "source_file": "4_infraestrutura.tex"
            },
            {
                "question": "Qual o período do curso?",
                "expected_substrings": ["período", "semestre", "integral", "noturno"],
                "source_file": "2_Organização_Didático_Pedagógica.tex"
            },
            {
                "question": "O curso possui a disciplina de Estrutura de Dados?",
                "expected_substrings": ["estrutura de dados", "árvores", "binárias", "listas"],
                "source_file": "unidades-curriculares.tex"
            },
            {
                "question": "Quais os conteúdos das disciplinas de Estrutura de Dados?",
                "expected_substrings": ["estrutura de dados", "ementa", "conteúdo", "listas", "lineares", "ordenação"],
                "source_file": "unidades-curriculares.tex"
            },
            {
                "question": "Quais às habilitações profissionais do egresso do curso?",
                "expected_substrings": ["egresso", "habilitações", "atividades profissionais", "atuação"],
                "source_file": "2_Organização_Didático_Pedagógica.tex"
            }
        ]

        for case in test_cases:
            with self.subTest(question=case["question"]):
                # Retrieve relevant documents for the query
                retrieved_docs = self.retriever.invoke(case["question"])
                
                # Check that we retrieved documents
                self.assertTrue(len(retrieved_docs) > 0, f"No documents retrieved for question: {case['question']}")

                # Check if at least one retrieved document matches the expected criteria
                match_found = False
                matched_content = []
                
                for doc in retrieved_docs:
                    content_lower = doc.page_content.lower()
                    source_meta = doc.metadata.get("source", "").lower()
                    
                    # Verify if the doc content contains all of the expected substrings
                    has_content_match = all(sub in content_lower for sub in case["expected_substrings"])
                    
                    if has_content_match or has_source_match:
                        match_found = True
                        matched_content.append(doc.page_content)

                # Assert that we found the relevant chunk
                self.assertTrue(
                    match_found, 
                    f"Could not find the expected chunk or source file for question: '{case['question']}'.\n"
                    f"Expected substrings: {case['expected_substrings']}\n"
                    f"Expected source: {case['source_file']}\n"
                    f"Retrieved docs sources: {[d.metadata.get('source') for d in retrieved_docs]}"
                )

    def test_retriever_no_matches(self):
        """Validates that completely unrelated queries do not retrieve chunks containing query-specific terms."""
        out_of_scope_cases = [
            {
                "question": "Qual a receita de bolo de chocolate?",
                "unrelated_words": ["receita", "bolo", "chocolate"]
            },
            {
                "question": "Como trocar o pneu de um carro?",
                "unrelated_words": ["trocar", "pneu", "carro"]
            },
            {
                "question": "Quem ganhou a copa do mundo de 1994?",
                "unrelated_words": ["copa", "futebol", "1994"]
            }
        ]

        for case in out_of_scope_cases:
            with self.subTest(question=case["question"]):
                retrieved_docs = self.retriever.invoke(case["question"])
                
                # Check that none of the retrieved documents contain the query-specific unrelated words
                for doc in retrieved_docs:
                    content_lower = doc.page_content.lower()
                    for word in case["unrelated_words"]:
                        self.assertNotIn(
                            word, 
                            content_lower, 
                            f"Retrieved document unexpectedly contained the unrelated word '{word}' from query '{case['question']}'."
                        )

if __name__ == '__main__':
    unittest.main()
