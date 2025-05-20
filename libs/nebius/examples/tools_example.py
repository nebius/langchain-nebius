"""Example using Nebius Tools for document retrieval."""

import os
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_openai import ChatOpenAI

from langchain_nebius import (
    NebiusEmbeddings,
    NebiusRetriever,
    NebiusRetrievalTool,
    nebius_search,
)


def main():
    """Run the examples."""
    # Set API keys from environment
    nebius_api_key = os.environ.get("NEBIUS_API_KEY")
    if not nebius_api_key:
        print("Please set the NEBIUS_API_KEY environment variable")
        return

    # Sample documents
    docs = [
        Document(page_content="Paris is the capital of France and has the Eiffel Tower."),
        Document(page_content="Berlin is the capital of Germany and has the Brandenburg Gate."),
        Document(page_content="Rome is the capital of Italy and has the Colosseum."),
        Document(page_content="London is the capital of the United Kingdom and has Big Ben."),
        Document(page_content="Madrid is the capital of Spain and has the Prado Museum."),
    ]

    # Initialize embeddings
    embeddings = NebiusEmbeddings(api_key=nebius_api_key)

    # Initialize retriever
    retriever = NebiusRetriever(
        embeddings=embeddings,
        docs=docs,
        k=2,  # Default number of results to return
    )

    print("\n=== Example 1: Using NebiusRetrievalTool ===")
    # Example 1: Using the NebiusRetrievalTool
    tool = NebiusRetrievalTool(
        retriever=retriever,
        name="nebius_document_search",
        description="Search for information in documents about European capitals."
    )

    # Use the tool directly
    result = tool.invoke({"query": "What is in Paris?", "k": 1})
    print(f"Tool result: {result}")

    print("\n=== Example 2: Using nebius_search decorator-based tool ===")
    # Example 2: Using the nebius_search function
    result = nebius_search.invoke({
        "query": "Tell me about Rome",
        "retriever": retriever,
        "k": 1,
    })
    print(f"Search result: {result}")

    print("\n=== Example 3: Using tools with an agent ===")
    # Example 3: Using tools with an agent
    # Check if OpenAI API key is set
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        print("To run the agent example, please set the OPENAI_API_KEY environment variable")
        return

    # Create LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo")

    # Create prompt
    system_prompt = """You are a helpful assistant that answers questions about European capitals.
    Use the nebius_document_search tool to find relevant information before answering."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{input}")
    ])

    # Create agent
    agent = create_openai_functions_agent(llm, [tool], prompt)
    agent_executor = AgentExecutor(agent=agent, tools=[tool], verbose=True)

    # Run the agent
    response = agent_executor.invoke({"input": "What famous landmark is in Paris?"})
    print(f"Agent response: {response['output']}")


if __name__ == "__main__":
    main() 