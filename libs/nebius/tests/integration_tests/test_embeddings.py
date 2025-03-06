"""Test Nebius AI Studio embeddings."""

from langchain_nebius import NebiusEmbeddings


def test_langchain_nebius_embed_documents() -> None:
    """Test Nebius AI Studio embeddings."""
    documents = ["foo bar", "bar foo"]
    embedding = NebiusEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 2
    assert len(output[0]) > 0


def test_langchain_nebius_embed_query() -> None:
    """Test Nebius AI Studio embeddings."""
    query = "foo bar"
    embedding = NebiusEmbeddings()
    output = embedding.embed_query(query)
    assert len(output) > 0


async def test_langchain_nebius_aembed_documents() -> None:
    """Test Nebius AI Studio embeddings asynchronous."""
    documents = ["foo bar", "bar foo"]
    embedding = NebiusEmbeddings()
    output = await embedding.aembed_documents(documents)
    assert len(output) == 2
    assert len(output[0]) > 0


async def test_langchain_nebius_aembed_query() -> None:
    """Test Nebius AI Studio embeddings asynchronous."""
    query = "foo bar"
    embedding = NebiusEmbeddings()
    output = await embedding.aembed_query(query)
    assert len(output) > 0
