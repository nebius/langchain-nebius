# LangChain Nebius Integration

This package provides LangChain integration for Nebius AI Studio, enabling seamless use of Nebius AI Studio's chat and embedding models within LangChain.

## Installation

Install the package using pip:

```bash
pip install langchain-nebius
```

## Usage

### Chat Models

```python
from langchain_nebius import ChatNebius

chat = ChatNebius(api_key="your-api-key")
response = chat.invoke(
    [{"role": "user", "content": "What is 1 + 1?"}]
)
print(response.content)
```

### Embeddings

```python
from langchain_nebius import NebiusEmbeddings

embeddings = NebiusEmbeddings(api_key="your-api-key")
document_embeddings = embeddings.embed_documents(texts=["Hello, world!"])
query_embedding = embeddings.embed_query(text="Hello")
```

### Retrievers

```python
from langchain_core.documents import Document
from langchain_nebius import NebiusEmbeddings, NebiusRetriever

# Create embeddings
embeddings = NebiusEmbeddings(api_key="your-api-key")

# Create documents
docs = [
    Document(page_content="Paris is the capital of France"),
    Document(page_content="Berlin is the capital of Germany"),
    # Add more documents as needed
]

# Create retriever
retriever = NebiusRetriever(
    embeddings=embeddings,
    docs=docs,
    k=3  # Number of documents to return
)

# Retrieve relevant documents
query = "What is the capital of France?"
results = retriever.invoke(query)
for doc in results:
    print(doc.page_content)
```

### Tools

The package provides tools that can be used with LangChain agents:

#### Using NebiusRetrievalTool (Class-based Tool)

```python
from langchain_core.documents import Document
from langchain_nebius import NebiusEmbeddings, NebiusRetriever, NebiusRetrievalTool

# Prepare your documents
docs = [
    Document(page_content="Paris is the capital of France"),
    Document(page_content="Berlin is the capital of Germany"),
    Document(page_content="Rome is the capital of Italy"),
]

# Create embeddings and retriever
embeddings = NebiusEmbeddings(api_key="your-api-key")
retriever = NebiusRetriever(embeddings=embeddings, docs=docs)

# Create the tool
tool = NebiusRetrievalTool(
    retriever=retriever,
    name="nebius_search",
    description="Search for information in the document collection"
)

# Use the tool
result = tool.invoke({"query": "What is the capital of France?", "k": 1})
print(result)
```

#### Using nebius_search (Decorator-based Tool)

```python
from langchain_core.documents import Document
from langchain_nebius import NebiusEmbeddings, NebiusRetriever, nebius_search

# Prepare your documents
docs = [
    Document(page_content="Paris is the capital of France"),
    Document(page_content="Berlin is the capital of Germany"),
    Document(page_content="Rome is the capital of Italy"),
]

# Create embeddings and retriever
embeddings = NebiusEmbeddings(api_key="your-api-key")
retriever = NebiusRetriever(embeddings=embeddings, docs=docs)

# Use the tool
result = nebius_search.invoke({
    "query": "What is the capital of France?",
    "retriever": retriever,
    "k": 1
})
print(result)
```

### Building a RAG Application

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_nebius import ChatNebius, NebiusEmbeddings, NebiusRetriever

# Create components
embeddings = NebiusEmbeddings()
retriever = NebiusRetriever(embeddings=embeddings, docs=documents)
llm = ChatNebius(model="meta-llama/Llama-3.3-70B-Instruct-fast")

# Create prompt
prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:

Context:
{context}

Question: {question}
""")

# Format documents function
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Create RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Run the chain
answer = rag_chain.invoke("What is the capital of France?")
print(answer)
```

### Using Tools with an Agent

```python
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_nebius import NebiusEmbeddings, NebiusRetriever, NebiusRetrievalTool

# Create documents and retriever
docs = [
    Document(page_content="Paris is the capital of France"),
    Document(page_content="Berlin is the capital of Germany"),
    Document(page_content="Rome is the capital of Italy"),
]
embeddings = NebiusEmbeddings()
retriever = NebiusRetriever(embeddings=embeddings, docs=docs)

# Create the retrieval tool
retrieval_tool = NebiusRetrievalTool(
    retriever=retriever,
    name="document_search",
    description="Search for information in the document collection"
)

# Create an LLM (using OpenAI as an example)
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Create the system prompt
system_prompt = """You are an assistant that answers questions based on the available documents.
Use the document_search tool to find relevant information before answering."""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "{input}")
])

# Create the agent
tools = [retrieval_tool]
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Run the agent
response = agent_executor.invoke({"input": "What is the capital of France?"})
print(response["output"])
```

For more examples, see the [examples](examples/) directory.

## Documentation

For more details, refer to the [Nebius AI Studio API Documentation](https://studio.nebius.ai/docs/api-reference).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.