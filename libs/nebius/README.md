# LangChain Nebius

**LangChain integration for [Nebius Token Factory](https://tokenfactory.nebius.com/)** — use Nebius-hosted chat and embedding models through LangChain’s standard APIs (chat models, embeddings, retrievers, and tools).

- OpenAI-compatible HTTP API
- Chat completions with tool calling, streaming, and structured output (model-dependent)
- Text embeddings and a lightweight in-memory retriever for RAG-style workflows

## Installation

```bash
pip install langchain-nebius
```

Dependencies are declared in the package metadata; you typically only install `langchain-nebius` explicitly.

## Get an API key

Create a key in **Nebius Token Factory** and export it:

```bash
export NEBIUS_API_KEY="your-api-key"
```

Optional: set the API base URL. The library default is `https://api.studio.nebius.ai/v1/`. To call **Token Factory** instead, use:

```bash
export NEBIUS_API_BASE="https://api.tokenfactory.nebius.com/v1/"
```

You can also pass `api_key` / `base_url` when constructing clients (see below).

## Chat models

`ChatNebius` reads `NEBIUS_API_KEY` from the environment when `api_key` is omitted. The default chat model is `Qwen/Qwen3-32B` unless you pass `model=...`.

```python
from langchain_nebius import ChatNebius

chat = ChatNebius(
    model="Qwen/Qwen3-30B-A3B-Instruct-2507",
    temperature=0.2,
)
response = chat.invoke([{"role": "user", "content": "What is 1 + 1?"}])
print(response.content)
```

With an explicit key:

```python
chat = ChatNebius(api_key="your-api-key")
```

## Embeddings

```python
from langchain_nebius import NebiusEmbeddings

embeddings = NebiusEmbeddings(
    model="Qwen/Qwen3-Embedding-8B",
    # api_key="your-api-key",  # optional if NEBIUS_API_KEY is set
)

document_embeddings = embeddings.embed_documents(["Hello, world!"])
query_embedding = embeddings.embed_query("Hello")
```

## Retrievers

`NebiusRetriever` embeds your documents with `NebiusEmbeddings`, then scores them with cosine similarity. Use `invoke` for retrieval (LangChain’s standard entrypoint).

```python
from langchain_core.documents import Document
from langchain_nebius import NebiusEmbeddings, NebiusRetriever

embeddings = NebiusEmbeddings()

docs = [
    Document(page_content="Paris is the capital of France"),
    Document(page_content="Berlin is the capital of Germany"),
]

retriever = NebiusRetriever(
    embeddings=embeddings,
    docs=docs,
    k=3,
)

results = retriever.invoke("What is the capital of France?")
for doc in results:
    print(doc.page_content)
```

## Tools

Use these with LangChain agents, graphs, or any code that expects LangChain tools.

### `NebiusRetrievalTool` (class-based)

```python
from langchain_core.documents import Document
from langchain_nebius import NebiusEmbeddings, NebiusRetriever, NebiusRetrievalTool

docs = [
    Document(page_content="Paris is the capital of France"),
    Document(page_content="Berlin is the capital of Germany"),
    Document(page_content="Rome is the capital of Italy"),
]

embeddings = NebiusEmbeddings()
retriever = NebiusRetriever(embeddings=embeddings, docs=docs, k=2)

tool = NebiusRetrievalTool(
    retriever=retriever,
    name="nebius_search",
    description="Search for information in the document collection",
)

result = tool.invoke({"query": "What is the capital of France?"})
print(result)
```

### `nebius_search` (tool function)

```python
from langchain_core.documents import Document
from langchain_nebius import NebiusEmbeddings, NebiusRetriever, nebius_search

docs = [
    Document(page_content="Paris is the capital of France"),
    Document(page_content="Berlin is the capital of Germany"),
    Document(page_content="Rome is the capital of Italy"),
]

embeddings = NebiusEmbeddings()
retriever = NebiusRetriever(embeddings=embeddings, docs=docs)

result = nebius_search.invoke(
    {"query": "What is the capital of France?", "retriever": retriever}
)
print(result)
```

## Building a RAG chain

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_nebius import ChatNebius, NebiusEmbeddings, NebiusRetriever

docs = [
    Document(page_content="Paris is the capital of France"),
    Document(page_content="Berlin is the capital of Germany"),
]

embeddings = NebiusEmbeddings()
retriever = NebiusRetriever(embeddings=embeddings, docs=docs, k=3)
llm = ChatNebius(model="Qwen/Qwen3-32B")

prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:

Context:
{context}

Question: {question}"""
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("What is the capital of France?")
print(answer)
```

## Using a retrieval tool with an agent

Runnable agents in LangChain live in the **`langchain`** package (separate from `langchain-core`):

```bash
pip install langchain
```

This pattern matches the scripts under [`examples/`](examples/): a **tool-calling** chat model plus `AgentExecutor`. Here the chat model is **`ChatNebius`**, so the agent and retrieval both use Nebius:

```python
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_nebius import (
    ChatNebius,
    NebiusEmbeddings,
    NebiusRetriever,
    NebiusRetrievalTool,
)

docs = [
    Document(page_content="Paris is the capital of France"),
    Document(page_content="Berlin is the capital of Germany"),
    Document(page_content="Rome is the capital of Italy"),
]
embeddings = NebiusEmbeddings()
retriever = NebiusRetriever(embeddings=embeddings, docs=docs, k=3)

retrieval_tool = NebiusRetrievalTool(
    retriever=retriever,
    name="document_search",
    description="Search for information in the document collection",
)

llm = ChatNebius(model="Qwen/Qwen3-32B")

system_prompt = """You are an assistant that answers questions based on the available documents.
Use the document_search tool to find relevant information before answering."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("user", "{input}"),
    ]
)

tools = [retrieval_tool]
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

response = agent_executor.invoke({"input": "What is the capital of France?"})
print(response["output"])
```

> **Note:** Newer LangChain releases may expose `create_tool_calling_agent` instead of `create_openai_functions_agent`; upgrade `langchain` if imports fail, or follow the versions pinned in your project.

## Examples

See the [`examples`](examples/) folder in this repository for runnable scripts (retrieval, tools, and agents).

## Documentation

- **[Nebius Token Factory documentation](https://docs.tokenfactory.nebius.com/)** — API reference, quickstart, and auth.
- **[LangChain Nebius integration](https://python.langchain.com/docs/integrations/chat/nebius/)** — ecosystem docs and examples.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
