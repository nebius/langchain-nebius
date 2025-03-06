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
from langchain_nebius.chat_models import NebiusChat

chat = NebiusChat(api_key="your-api-key")
response = chat._generate(
    messages=[{"role": "user", "content": "What is 1 + 1?"}]
)
print(response.generations[0]["message"]["content"])
```

### Embeddings

```python
from langchain_nebius.embeddings import NebiusEmbeddings

embeddings = NebiusEmbeddings(api_key="your-api-key")
document_embeddings = embeddings.embed_documents(texts=["Hello, world!"])
query_embedding = embeddings.embed_query(text="Hello")
```

## Documentation

For more details, refer to the [Nebius AI Studio API Documentation](https://studio.nebius.ai/docs/api-reference).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.