"""Example of using the NebiusRetriever for document retrieval and Q&A."""

import os
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain_nebius import ChatNebius, NebiusEmbeddings, NebiusRetriever


def main():
    """Run the example."""
    # Check for API key
    api_key = os.environ.get("NEBIUS_API_KEY")
    if not api_key:
        print("Please set the NEBIUS_API_KEY environment variable.")
        return

    # Create sample documents
    documents = [
        Document(page_content="Paris is the capital and most populous city of France. It has an estimated population of 2.1 million."),
        Document(page_content="Berlin is the capital and largest city of Germany. It has a population of about 3.7 million."),
        Document(page_content="Rome is the capital city of Italy. It is located in the central-western portion of the Italian peninsula."),
        Document(page_content="Madrid is the capital and most populous city of Spain. The city has almost 3.4 million inhabitants."),
        Document(page_content="London is the capital and largest city of England and the United Kingdom. It had a population of over 9 million in 2021."),
    ]

    # Initialize the embeddings model
    print("Initializing embeddings model...")
    embeddings = NebiusEmbeddings()

    # Create the retriever
    print("Creating retriever...")
    retriever = NebiusRetriever(
        embeddings=embeddings,
        docs=documents,
        k=2  # Return top 2 most relevant documents
    )

    # Create a chat model
    print("Initializing chat model...")
    llm = ChatNebius(
        model="meta-llama/Llama-3.3-70B-Instruct-fast",
        temperature=0
    )

    # Create a simple RAG prompt
    template = """Answer the question based only on the following context:

Context:
{context}

Question: {question}

Answer:"""

    prompt = ChatPromptTemplate.from_template(template)

    # Helper function to format documents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Create a RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Test with a query
    query = "What is the capital of France and what is its population?"
    print("\nQuery:", query)

    # Retrieve relevant documents and show them
    print("\nRetrieving relevant documents...")
    docs = retriever.get_relevant_documents(query)
    print("\nRetrieved Documents:")
    for i, doc in enumerate(docs, 1):
        print(f"{i}. {doc.page_content}")

    # Generate answer using the RAG chain
    print("\nGenerating answer...")
    answer = rag_chain.invoke(query)
    print("\nAnswer:", answer)


if __name__ == "__main__":
    main() 