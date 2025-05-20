"""Unit tests for Nebius retrievers."""

import unittest
from unittest.mock import MagicMock, patch
import pytest
import openai
from langchain_core.documents import Document
from pydantic import SecretStr

from langchain_nebius.embeddings import NebiusEmbeddings
from langchain_nebius.retrievers import NebiusRetriever


class TestNebiusRetriever(unittest.TestCase):
    """Test Nebius retriever unit tests."""

    def setUp(self):
        """Set up test data."""
        # Create a mock for the embeddings class
        self.mock_embeddings = MagicMock(spec=NebiusEmbeddings)
        self.mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
        self.mock_embeddings.aembed_query.return_value = [0.1, 0.2, 0.3]
        self.mock_embeddings.embed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        
        # Add required attributes for the NebiusEmbeddings
        self.mock_embeddings.nebius_api_key = None
        self.mock_embeddings.nebius_api_base = "https://api.studio.nebius.ai/v1/"
        self.mock_embeddings.client = None
        self.mock_embeddings.async_client = None
        self.mock_embeddings.request_timeout = None
        self.mock_embeddings.max_retries = 2
        self.mock_embeddings.default_headers = None
        self.mock_embeddings.default_query = None
        self.mock_embeddings.http_client = None
        self.mock_embeddings.http_async_client = None
        
        # Create sample documents
        self.docs = [
            Document(page_content="Paris is the capital of France"),
            Document(page_content="Berlin is the capital of Germany"),
            Document(page_content="Rome is the capital of Italy"),
        ]
        
        # Patch OpenAI client creation to avoid actual API calls
        self.openai_patcher = patch('openai.OpenAI')
        self.mock_openai = self.openai_patcher.start()
        
        # Mock the embeddings property on OpenAI
        self.mock_openai_instance = MagicMock()
        self.mock_openai.return_value = self.mock_openai_instance
        self.mock_openai_instance.embeddings = MagicMock()
        
        # Patch AsyncOpenAI as well
        self.async_openai_patcher = patch('openai.AsyncOpenAI')
        self.mock_async_openai = self.async_openai_patcher.start()
        
        # Mock the embeddings property on AsyncOpenAI
        self.mock_async_openai_instance = MagicMock()
        self.mock_async_openai.return_value = self.mock_async_openai_instance
        self.mock_async_openai_instance.embeddings = MagicMock()
        
        # Patch the post_init method of NebiusRetriever
        self.post_init_patcher = patch('langchain_nebius.retrievers.NebiusRetriever.post_init')
        self.mock_post_init = self.post_init_patcher.start()
        self.mock_post_init.return_value = None  # Return self

    def tearDown(self):
        """Clean up after tests."""
        # Stop patchers
        self.openai_patcher.stop()
        self.async_openai_patcher.stop()
        self.post_init_patcher.stop()

    def test_initialization(self):
        """Test retriever initialization with default parameters."""
        retriever = NebiusRetriever(
            embeddings=self.mock_embeddings,
            docs=self.docs
        )
        
        # Manually set doc_embeddings since we're mocking post_init
        retriever.doc_embeddings = [[0.1, 0.2, 0.3]] * len(self.docs)
        
        # Test default values
        self.assertEqual(retriever.k, 3)
        self.assertEqual(len(retriever.docs), 3)
        self.assertEqual(retriever.nebius_api_base, "https://api.studio.nebius.ai/v1/")

    def test_initialization_with_custom_parameters(self):
        """Test retriever initialization with custom parameters."""
        retriever = NebiusRetriever(
            embeddings=self.mock_embeddings,
            docs=self.docs,
            k=5,
            api_key=SecretStr("fake-api-key"),
            base_url="https://custom-api.example.com/"
        )
        
        # Manually set doc_embeddings since we're mocking post_init
        retriever.doc_embeddings = [[0.1, 0.2, 0.3]] * len(self.docs)
        
        # Test custom values
        self.assertEqual(retriever.k, 5)
        self.assertEqual(retriever.nebius_api_base, "https://custom-api.example.com/")
        self.assertEqual(retriever.nebius_api_key.get_secret_value(), "fake-api-key")

    def test_empty_docs(self):
        """Test retriever with empty documents."""
        retriever = NebiusRetriever(
            embeddings=self.mock_embeddings,
            docs=[]
        )
        
        # Test that get_relevant_documents returns empty list when docs is empty
        with patch('langchain_nebius.retrievers.CallbackManagerForRetrieverRun', MagicMock()):
            results = retriever._get_relevant_documents("test query", run_manager=MagicMock())
            self.assertEqual(results, [])

    def test_similarity_search(self):
        """Test similarity search functionality."""
        retriever = NebiusRetriever(
            embeddings=self.mock_embeddings,
            docs=self.docs,
        )
        
        # Setup mock embeddings for similarity computation
        retriever.doc_embeddings = [
            [0.9, 0.1, 0.1],  # High similarity to query [0.1, 0.2, 0.3]
            [0.1, 0.1, 0.1],  # Low similarity
            [0.5, 0.2, 0.4],  # Medium similarity
        ]
        
        # Test similarity search with k=2
        query_embedding = [0.1, 0.2, 0.3]
        results = retriever._similarity_search(query_embedding, k=2)
        
        # Should return most similar docs
        self.assertEqual(len(results), 2)
        
        # Testing document contents rather than specific ordering
        # This makes the test more robust to changes in similarity calculation
        document_contents = [doc.page_content for doc in results]
        self.assertIn("Berlin is the capital of Germany", document_contents)
        self.assertIn("Rome is the capital of Italy", document_contents)

    def test_get_relevant_documents(self):
        """Test get_relevant_documents method."""
        retriever = NebiusRetriever(
            embeddings=self.mock_embeddings,
            docs=self.docs,
        )
        
        # Set up mock embeddings response
        self.mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
        retriever.doc_embeddings = [[0.1, 0.2, 0.3]] * len(self.docs)
        
        # Mock the _similarity_search method to return predictable results
        with patch.object(retriever, '_similarity_search', return_value=self.docs[:2]) as mock_search:
            with patch('langchain_nebius.retrievers.CallbackManagerForRetrieverRun', MagicMock()):
                results = retriever._get_relevant_documents("test query", run_manager=MagicMock())
                
                # Verify _similarity_search was called with the right parameters
                mock_search.assert_called_once()
                self.assertEqual(len(results), 2)
    
    def test_retriever_with_custom_k(self):
        """Test retriever with custom k parameter."""
        retriever = NebiusRetriever(
            embeddings=self.mock_embeddings,
            docs=self.docs,
            k=2
        )
        
        # Set up mock embeddings response
        retriever.doc_embeddings = [[0.1, 0.2, 0.3]] * len(self.docs)
        
        # Test with custom k parameter in the method call
        with patch.object(retriever, '_similarity_search', return_value=self.docs[:1]) as mock_search:
            with patch('langchain_nebius.retrievers.CallbackManagerForRetrieverRun', MagicMock()):
                results = retriever._get_relevant_documents("test query", run_manager=MagicMock(), k=1)
                
                # Verify _similarity_search was called with k=1
                mock_search.assert_called_once_with([0.1, 0.2, 0.3], 1)
                self.assertEqual(len(results), 1)

    @pytest.mark.asyncio
    async def test_aget_relevant_documents(self):
        """Test async get_relevant_documents method."""
        retriever = NebiusRetriever(
            embeddings=self.mock_embeddings,
            docs=self.docs,
        )
        
        # Set up mock embeddings response
        self.mock_embeddings.aembed_query.return_value = [0.1, 0.2, 0.3]
        retriever.doc_embeddings = [[0.1, 0.2, 0.3]] * len(self.docs)
        
        # Mock the _similarity_search method to return predictable results
        with patch.object(retriever, '_similarity_search', return_value=self.docs[:2]) as mock_search:
            with patch('langchain_nebius.retrievers.CallbackManagerForRetrieverRun', MagicMock()):
                results = await retriever._aget_relevant_documents("test query", run_manager=MagicMock())
                
                # Verify _similarity_search was called with the right parameters
                mock_search.assert_called_once()
                self.assertEqual(len(results), 2)


if __name__ == "__main__":
    unittest.main()
