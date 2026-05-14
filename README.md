# LangChain Nebius

This repository contains the **[langchain-nebius](https://pypi.org/project/langchain-nebius/)** package: LangChain integrations for **[Nebius Token Factory](https://tokenfactory.nebius.com/)** (chat models, embeddings, retrievers, and tools).

Package source and PyPI README live under [`libs/nebius`](libs/nebius/).

## Setup for testing

```bash
cd libs/nebius
poetry install --with lint,typing,test,test_integration
```

## Running the unit tests

```bash
cd libs/nebius
make tests
```

## Running the integration tests

```bash
cd libs/nebius
export NEBIUS_API_KEY=<your-api-key>
make integration_tests
```

© Nebius BV, 2025
