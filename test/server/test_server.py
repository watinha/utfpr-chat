import sys
import os
from unittest import TestCase, main
from unittest.mock import MagicMock

# Walk up directories to locate the project root containing server.py
current_dir = os.path.dirname(os.path.abspath(__file__)) if __file__ else os.getcwd()
while current_dir and current_dir != os.path.dirname(current_dir):
    if os.path.exists(os.path.join(current_dir, 'server.py')):
        sys.path.insert(0, current_dir)
        break
    current_dir = os.path.dirname(current_dir)

# Create mocks for backend dependencies
mock_rag = MagicMock()
mock_misc = MagicMock()

# Inject mocks into sys.modules BEFORE importing server to isolate Flask testing
sys.modules['rag'] = mock_rag
sys.modules['misc'] = mock_misc

# Import Flask app and validation schemas
from server import app

class TestServerAPI(TestCase):
    def setUp(self):
        self.client = app.test_client()
        # Reset mocks before each test
        mock_rag.reset_mock()
        mock_misc.reset_mock()

    def test_index_route(self):
        """Test that the index route is accessible."""
        with self.client.get('/') as response:
            self.assertIn(response.status_code, [200, 404])

    def test_answer_route_success(self):
        """Test /answer endpoint with a valid query parameter."""
        # Mock document structure returned by RAG
        mock_doc = MagicMock()
        mock_doc.metadata = {"source": "manual.txt", "page": 4}
        mock_doc.page_content = "Conteúdo sobre ementas de CDIA."
        
        mock_rag.rag_query.return_value = ("Esta é a resposta simulada do RAG.", [mock_doc])

        with self.client.get('/answer?question=Qual%20o%20periodo%20do%20curso%3F') as response:
            self.assertEqual(response.status_code, 200)
            data = response.get_json()
        
        self.assertEqual(data["answer"], "Esta é a resposta simulada do RAG.")
        self.assertEqual(len(data["contexts"]), 1)
        self.assertEqual(data["contexts"][0]["metadata"]["source"], "manual.txt")
        self.assertEqual(data["contexts"][0]["page_content"], "Conteúdo sobre ementas de CDIA.")
        
        # Verify mock call
        mock_rag.rag_query.assert_called_once_with("Qual o periodo do curso?")

    def test_answer_route_missing_query(self):
        """Test /answer validation when question parameter is missing."""
        with self.client.get('/answer') as response:
            self.assertEqual(response.status_code, 400)

    def test_answer_route_empty_query(self):
        """Test /answer validation when question parameter is empty."""
        with self.client.get('/answer?question=') as response:
            self.assertEqual(response.status_code, 400)

    def test_joke_route_success(self):
        """Test /joke endpoint with mock make_joke call."""
        mock_misc.make_joke.return_value = "Esta é uma piada sobre computação."

        with self.client.get('/joke?question=Me%20conte%20uma%20piada') as response:
            self.assertEqual(response.status_code, 200)
            data = response.get_json()
        
        self.assertEqual(data["joke"], "Esta é uma piada sobre computação.")
        mock_misc.make_joke.assert_called_once_with("Me conte uma piada")

    def test_invitation_route_success(self):
        """Test /invitation endpoint calling rag_query followed by make_invitation."""
        mock_rag.rag_query.return_value = ("Resposta do RAG para convite.", [])
        mock_misc.make_invitation.return_value = "Venha estudar no curso de CDIA da UTFPR!"

        with self.client.get('/invitation?question=Quero%20visitar%20a%20universidade') as response:
            self.assertEqual(response.status_code, 200)
            data = response.get_json()
        
        self.assertEqual(data["invitation"], "Venha estudar no curso de CDIA da UTFPR!")
        
        mock_rag.rag_query.assert_called_once_with("Quero visitar a universidade")
        mock_misc.make_invitation.assert_called_once_with("Resposta do RAG para convite.")

if __name__ == '__main__':
    main()
