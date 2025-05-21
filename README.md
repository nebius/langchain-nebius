# 🦜️🔗 LangChain Nebius

This repository contains 1 package with Nebius AI Studio integrations with LangChain:

- [langchain-nebius](https://pypi.org/project/langchain-nebius/)

## Setup for Testing

```bash
cd libs/nebius
poetry install --with lint,typing,test,test_integration,
```

## Running the Unit Tests

```bash
cd libs/nebius
make tests
```

## Running the Integration Tests

```bash
cd libs/nebius
export NEBIUS_API_KEY=<your-api-key>
make integration_tests
```

© Nebius BV, 2025