"""Integration tests for Nebius retrievers."""

import os
import pytest
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain_nebius.embeddings import NebiusEmbeddings
from langchain_nebius.chat_models import ChatNebius
from langchain_nebius.retrievers import NebiusRetriever

# Skip tests if NEBIUS_API_KEY is not set
requires_api_key = pytest.mark.skipif(
    not os.environ.get("NEBIUS_API_KEY"),
    reason="NEBIUS_API_KEY environment variable not set"
)


@requires_api_key
class TestNebiusRetrieverIntegration:
    """Test Nebius retriever with actual API calls."""

    def setup_method(self):
        """Set up test data."""
        self.docs = [
            Document(page_content="Paris is the capital of France and has a population of about 2.1 million."),
            Document(page_content="Berlin is the capital and largest city of Germany with a population of about 3.7 million."),
            Document(page_content="Rome is the capital city of Italy with a population of about 2.8 million people."),
            Document(page_content="Madrid is the capital and most populous city of Spain with around 3.3 million residents."),
            Document(page_content="London is the capital and largest city of England and the United Kingdom with a population over 9 million."),
        ]
        self.embeddings = NebiusEmbeddings()
        self.llm = ChatNebius()

    def test_retriever_initialization(self):
        """Test retriever initialization with real embeddings."""
        retriever = NebiusRetriever(
            embeddings=self.embeddings,
            docs=self.docs
        )
        
        # Check the retriever was properly initialized
        assert len(retriever.docs) == 5
        assert isinstance(retriever.embeddings, NebiusEmbeddings)
        assert retriever.k == 3  # Default value

    def test_get_relevant_documents(self):
        """Test retrieving relevant documents."""
        retriever = NebiusRetriever(
            embeddings=self.embeddings,
            docs=self.docs
        )
        
        # Get documents relevant to a query about France
        query = "What is the capital of France?"
        results = retriever.get_relevant_documents(query)
        
        # Should return the default number of documents (k=3)
        assert len(results) == 3
        
        # The first result should be about Paris
        assert "Paris" in results[0].page_content
        assert "France" in results[0].page_content

    def test_get_relevant_documents_with_custom_k(self):
        """Test retrieving a custom number of documents."""
        retriever = NebiusRetriever(
            embeddings=self.embeddings,
            docs=self.docs
        )
        
        # Get documents with custom k parameter
        query = "What are the populations of European capitals?"
        results = retriever.get_relevant_documents(query, k=2)
        
        # Should return exactly 2 documents
        assert len(results) == 2
        
        # Results should contain population information
        assert any("population" in doc.page_content for doc in results)

    @pytest.mark.asyncio
    async def test_aget_relevant_documents(self):
        """Test async retrieval of documents."""
        retriever = NebiusRetriever(
            embeddings=self.embeddings,
            docs=self.docs
        )
        
        # Get documents relevant to a query about Germany
        query = "Tell me about Berlin"
        results = await retriever.aget_relevant_documents(query)
        
        # Should return the default number of documents (k=3)
        assert len(results) == 3
        
        # One of the results should be about Berlin
        assert any("Berlin" in doc.page_content for doc in results)
        assert any("Germany" in doc.page_content for doc in results)

    def test_in_rag_chain(self):
        """Test the retriever works in a RAG chain."""
        retriever = NebiusRetriever(
            embeddings=self.embeddings,
            docs=self.docs,
            k=2  # Limit to 2 docs for faster testing
        )
        
        # Create a simple RAG prompt
        prompt = ChatPromptTemplate.from_template(
            """Answer the question based only on the following context:
            
            Context: {context}
            
            Question: {question}
            """
        )
        
        # Helper function to format documents
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # Create a RAG chain
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        # Test the chain with a query about France
        query = "What is the capital of France and what is its population?"
        response = rag_chain.invoke(query)
        
        # The response should mention Paris and the population
        assert "Paris" in response
        assert "2.1 million" in response or "2.1M" in response 