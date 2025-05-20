"""Nebius tools."""

from typing import List, Optional, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.documents import Document
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from langchain_nebius.retrievers import NebiusRetriever


class NebiusRetrievalToolInput(BaseModel):
    """Input schema for Nebius retrieval tool.

    This docstring is **not** part of what is sent to the model when performing tool
    calling. The Field default values and descriptions **are** part of what is sent to
    the model when performing tool calling.
    """

    query: str = Field(..., description="The search query to look up relevant documents")
    k: int = Field(3, description="Number of documents to return (default: 3)")


class NebiusRetrievalTool(BaseTool):
    """Tool for retrieving documents from Nebius.

    Setup:
        Install ``langchain-nebius`` and set environment variable ``NEBIUS_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-nebius
            export NEBIUS_API_KEY="your-api-key"

    Instantiation:
        .. code-block:: python

            from langchain_nebius.embeddings import NebiusEmbeddings
            from langchain_nebius.tools import NebiusRetrievalTool
            from langchain_core.documents import Document

            # Prepare your documents
            docs = [
                Document(page_content="Paris is the capital of France"),
                Document(page_content="Berlin is the capital of Germany"),
                Document(page_content="Rome is the capital of Italy"),
            ]

            # Create the tool with your documents and embeddings
            embeddings = NebiusEmbeddings()
            retrieval_tool = NebiusRetrievalTool(
                retriever=NebiusRetriever(
                    embeddings=embeddings,
                    docs=docs
                ),
                name="nebius_search",
                description="Search for information in the document collection"
            )

    Invocation with args:
        .. code-block:: python

            result = retrieval_tool.invoke({
                "query": "What is the capital of France?",
                "k": 1
            })
            print(result)

        .. code-block:: python

            "Paris is the capital of France"
    """

    name: str = "nebius_retrieval"
    """The name that is passed to the model when performing tool calling."""
    description: str = "Search for relevant documents based on a query using Nebius embeddings."
    """The description that is passed to the model when performing tool calling."""
    args_schema: Type[BaseModel] = NebiusRetrievalToolInput
    """The schema that is passed to the model when performing tool calling."""

    retriever: NebiusRetriever
    """The Nebius retriever instance to use for document retrieval."""
    return_direct: bool = False
    """Whether to return the results directly or as a string."""

    def _run(
        self, query: str, k: int = 3, *, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Run the tool to retrieve documents.

        Args:
            query: The search query.
            k: Number of documents to return.

        Returns:
            A string containing the content of the retrieved documents.
        """
        docs = self.retriever.get_relevant_documents(query, k=k)
        if not docs:
            return "No relevant documents found."
        
        return self._format_docs(docs)

    async def _arun(
        self,
        query: str,
        k: int = 3,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Run the tool asynchronously.

        Args:
            query: The search query.
            k: Number of documents to return.

        Returns:
            A string containing the content of the retrieved documents.
        """
        docs = await self.retriever.aget_relevant_documents(query, k=k)
        if not docs:
            return "No relevant documents found."
        
        return self._format_docs(docs)
    
    def _format_docs(self, docs: List[Document]) -> str:
        """Format documents into a readable string.

        Args:
            docs: The documents to format.

        Returns:
            A formatted string.
        """
        formatted_docs = []
        for i, doc in enumerate(docs, 1):
            formatted_docs.append(f"Document {i}:\n{doc.page_content}\n")
        
        return "\n".join(formatted_docs)


@tool
def nebius_search(query: str, retriever: NebiusRetriever, k: int = 3) -> str:
    """Search for information using Nebius embeddings.

    Args:
        query: The search query to find relevant documents.
        retriever: The Nebius retriever instance.
        k: Number of documents to return.

    Returns:
        Content of the most relevant documents.
    """
    docs = retriever.get_relevant_documents(query, k=k)
    if not docs:
        return "No relevant documents found."
    
    formatted_docs = []
    for i, doc in enumerate(docs, 1):
        formatted_docs.append(f"Document {i}:\n{doc.page_content}\n")
    
    return "\n".join(formatted_docs)
