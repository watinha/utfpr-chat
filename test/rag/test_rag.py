import unittest
import urllib.request
from rag.rag_query import rag_query

def is_ollama_available():
    try:
        # Check if local Ollama server is running and accessible
        with urllib.request.urlopen("http://localhost:11434/", timeout=2) as response:
            return response.status == 200
    except Exception:
        return False

class TestRagQuery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not is_ollama_available():
            raise unittest.SkipTest("Ollama is not running on localhost:11434. Skipping integration RAG query tests.")

    def test_example_questions_success(self):
        """Validates that all example questions from the frontend return valid RAG responses."""
        example_questions = [
            {
                "question": "Qual a qualificação dos docentes?",
                "expected_words": ["docente", "doutor", "professor", "mestre", "qualificação"]
            },
            {
                "question": "Como é a infraestrutura do curso?",
                "expected_words": ["laboratório", "equipamentos", "biblioteca", "infraestrutura", "bloco"]
            },
            {
                "question": "Quais as atividades profissionais exercidas pelo egresso do curso?",
                "expected_words": ["egresso", "profissional", "dados", "empresa", "analista", "cientista"]
            },
            {
                "question": "Qual o período do curso?",
                "expected_words": ["período", "integral", "semestre", "matutino", "noturno"]
            },
            {
                "question": "Matemática é um conteúdo importante para entrar no curso?",
                "expected_words": ["matemática", "disciplina", "álgebra", "cálculo", "base"]
            },
            {
                "question": "O curso tem Trabalho de Conclusão de Curso?",
                "expected_words": ["tcc", "trabalho de conclusão", "curso", "tcc1", "tcc2"]
            },
            {
                "question": "Como funcionam as atividades de extensão no curso?",
                "expected_words": ["extensão", "atividades", "projeto", "comunidade"]
            },
            {
                "question": "O curso tem atividades complementares?",
                "expected_words": ["atividades", "complementares", "horas", "grupo"]
            },
            {
                "question": "Qual a importância do estágio?",
                "expected_words": ["estágio", "supervisionado", "profissional", "prática", "empresa"]
            }
        ]

        for case in example_questions:
            question = case["question"]
            with self.subTest(question=question):
                answer, contexts = rag_query(question)
                
                # Check response structure
                self.assertIsInstance(answer, str)
                self.assertTrue(len(answer.strip()) > 0, f"Empty RAG answer returned for: '{question}'")
                
                # Verify that at least one of the expected keywords is present in the LLM response
                has_keyword = any(word.lower() in answer.lower() for word in case["expected_words"])
                self.assertTrue(
                    has_keyword, 
                    f"RAG answer did not contain any of the expected topical keywords {case['expected_words']}.\nAnswer: '{answer}'"
                )
                
                # Since these are in-scope course syllabus questions, they should match documents
                self.assertTrue(len(contexts) > 0, f"No contexts returned for in-scope question: '{question}'")

    def test_out_of_scope_query(self):
        """Validates that out-of-scope queries are correctly filtered and return no contexts."""
        out_of_scope_question = "Qual a receita de bolo de chocolate?"
        answer, contexts = rag_query(out_of_scope_question)
        
        # Should return a rejection string
        self.assertIsInstance(answer, str)
        self.assertTrue(len(answer.strip()) > 0)
        
        # Should return empty context list
        self.assertEqual(contexts, [], "Out-of-scope query returned contexts unexpectedly.")

if __name__ == '__main__':
    unittest.main()
