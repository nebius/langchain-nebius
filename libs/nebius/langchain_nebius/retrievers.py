"""Nebius retrievers."""

from typing import Any, Dict, List, Optional

import openai
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.utils import from_env, secret_from_env
from pydantic import ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_nebius.embeddings import NebiusEmbeddings


class NebiusRetriever(BaseRetriever):
    """Nebius retriever.

    Setup:
        Install ``langchain-nebius`` and set environment variable
        ``NEBIUS_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-nebius
            export NEBIUS_API_KEY="your-api-key"

    Key init args:
        embeddings: NebiusEmbeddings
            The Nebius embeddings model to use
        docs: List[Document]
            The documents to search through
        k: int
            Number of documents to return

    Instantiate:
        .. code-block:: python

            from langchain_nebius import NebiusRetriever, NebiusEmbeddings

            embeddings = NebiusEmbeddings()
            docs = [
                Document(page_content="Document 1 content"),
                Document(page_content="Document 2 content"),
                # ...
            ]
            
            retriever = NebiusRetriever(
                embeddings=embeddings,
                docs=docs,
                k=3
            )

    Usage:
        .. code-block:: python

            query = "What is the capital of France?"

            retrieved_docs = retriever.invoke(query)
            print(retrieved_docs)

        .. code-block:: none

            [Document(page_content='Document about Paris being the capital of France')]

    Use within a chain:
        .. code-block:: python

            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.runnables import RunnablePassthrough
            from langchain_nebius import ChatNebius

            prompt = ChatPromptTemplate.from_template(
                \"\"\"Answer the question based only on the context provided.

            Context: {context}

            Question: {question}\"\"\"
            )

            llm = ChatNebius(model="meta-llama/Llama-3.3-70B-Instruct-fast")

            def format_docs(docs):
                return "\\n\\n".join(doc.page_content for doc in docs)

            chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            chain.invoke("What is the capital of France?")

        .. code-block:: none

             "The capital of France is Paris."

    """

    embeddings: NebiusEmbeddings = Field(..., description="Nebius embeddings to use")
    docs: List[Document] = Field(default_factory=list, description="Documents to search through")
    k: int = Field(default=3, description="Number of documents to return")
    
    nebius_api_key: Optional[SecretStr] = Field(
        alias="api_key",
        default_factory=secret_from_env("NEBIUS_API_KEY", default=None),
        description="Nebius AI API key",
    )
    
    nebius_api_base: str = Field(
        default_factory=from_env(
            "NEBIUS_API_BASE", default="https://api.studio.nebius.ai/v1/"
        ),
        alias="base_url",
        description="Endpoint URL to use",
    )
    
    # Vector store to hold document embeddings
    doc_embeddings: List[List[float]] = Field(default_factory=list, exclude=True)
    
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        protected_namespaces=(),
    )
    
    @model_validator(mode="after")
    def post_init(self) -> Self:
        """Initialize client and compute embeddings for documents."""
        if not self.nebius_api_key:
            self.nebius_api_key = SecretStr(
                secret_from_env("NEBIUS_API_KEY", default="")
            )
        
        # Set up the client for the embeddings if needed
        if self.nebius_api_key and not self.embeddings.client:
            self.embeddings.nebius_api_key = self.nebius_api_key
            self.embeddings.nebius_api_base = self.nebius_api_base
        
        # Compute embeddings for the documents if we have any
        if self.docs and not self.doc_embeddings:
            texts = [doc.page_content for doc in self.docs]
            self.doc_embeddings = self.embeddings.embed_documents(texts)
        
        return self
        
    def _similarity_search(self, query_embedding: List[float], k: int) -> List[Document]:
        """Return documents most similar to the query embedding."""
        if not self.docs or not self.doc_embeddings:
            return []
            
        # Compute similarity scores for each document
        similarities = []
        for i, doc_embedding in enumerate(self.doc_embeddings):
            # Compute cosine similarity
            dot_product = sum(q * d for q, d in zip(query_embedding, doc_embedding))
            query_norm = sum(q ** 2 for q in query_embedding) ** 0.5
            doc_norm = sum(d ** 2 for d in doc_embedding) ** 0.5
            similarity = dot_product / (query_norm * doc_norm) if query_norm * doc_norm > 0 else 0
            similarities.append((i, similarity))
            
        # Sort by similarity and return top k documents
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k_indices = [idx for idx, _ in similarities[:k]]
        return [self.docs[idx] for idx in top_k_indices]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any
    ) -> List[Document]:
        """Get documents relevant to the query."""
        k = kwargs.get("k", self.k)
        
        # If we have no documents, return empty list
        if not self.docs:
            return []
            
        # Get query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Return most similar documents
        return self._similarity_search(query_embedding, k)

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        """Asynchronously get documents relevant to the query."""
        k = kwargs.get("k", self.k)
        
        # If we have no documents, return empty list
        if not self.docs:
            return []
            
        # Get query embedding
        query_embedding = await self.embeddings.aembed_query(query)
        
        # Return most similar documents
        return self._similarity_search(query_embedding, k)
