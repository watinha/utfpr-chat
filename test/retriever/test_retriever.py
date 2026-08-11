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
                "question": "Qual a qualificação, titulação e adequação dos docentes?",
                "expected_substrings": ["docente", "corpo", "titulação", "doutorado"],
                "source_file": "./docs/PPC_CDIA_LD_COGEP_2026.pdf"
            },
            {
                "question": "Como é a infraestrutura do curso?",
                "expected_substrings": ["infraestrutura", "laboratório", "equipamentos"],
                "source_file": "./docs/PPC_CDIA_LD_COGEP_2026.pdf"
            },
            {
                "question": "Qual o período do curso?",
                "expected_substrings": ["período", "noturno"],
                "source_file": "./docs/PPC_CDIA_LD_COGEP_2026.pdf"
            },
            {
                "question": "O curso possui a disciplina de Estrutura de Dados?",
                "expected_substrings": ["estrutura de dados", "árvores", "binárias", "listas"],
                "source_file": "./docs/PPC_CDIA_LD_COGEP_2026.pdf"
            },
            {
                "question": "Quais os conteúdos das disciplinas de Estrutura de Dados?",
                "expected_substrings": ["estrutura de dados", "ementa", "conteúdo", "listas", "lineares"],
                "source_file": "./docs/PPC_CDIA_LD_COGEP_2026.pdf"
            },
            {
                "question": "Quais às habilitações profissionais do egresso do curso?",
                "expected_substrings": ["egresso", "profissionais", "atuação"],
                "source_file": "./docs/PPC_CDIA_LD_COGEP_2026.pdf"
            }
        ]

        for case in test_cases:
            with self.subTest(question=case["question"]):
                # Retrieve relevant documents for the query
                retrieved_docs = self.retriever.invoke(case["question"])
                
                # Check that we retrieved documents
                self.assertTrue(len(retrieved_docs) > 0, f"No documents retrieved for question: {case['question']}")

                # Identify expected words that were not present in any retrieved chunk
                missing_words = [
                    sub for sub in case["expected_substrings"]
                    if not any(sub in doc.page_content.lower() for doc in retrieved_docs)
                ]

                # Build detailed info for each chunk
                chunks_info = []
                for idx, doc in enumerate(retrieved_docs):
                    content_lower = doc.page_content.lower()
                    found_in_chunk = [sub for sub in case["expected_substrings"] if sub in content_lower]
                    missing_in_chunk = [sub for sub in case["expected_substrings"] if sub not in content_lower]
                    chunks_info.append(
                        f" - Chunk ({idx + 1}): Found: {found_in_chunk} | Missing: {missing_in_chunk}\n"
                        f"   Content: {doc.page_content}"
                    )

                # Assert that all expected words appeared across the retrieved chunks
                self.assertEqual(
                    missing_words, [],
                    f"Expected words were missing across all retrieved chunks for question: '{case['question']}'.\n"
                    f"Words not present in any chunk: {missing_words}\n"
                    f"All expected words: {case['expected_substrings']}\n"
                    f"Expected source: {case['source_file']}\n"
                    f"Chunks:\n" + "\n".join(chunks_info) + "\n"
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
