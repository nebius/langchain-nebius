from langchain_nebius import ChatNebius, NebiusEmbeddings


def test_chat_nebius_secrets() -> None:
    o = ChatNebius(nebius_api_key="foo")  # type: ignore[call-arg]
    s = str(o)
    assert "foo" not in s


def test_nebius_embeddings_secrets() -> None:
    o = NebiusEmbeddings(nebius_api_key="foo")  # type: ignore[call-arg]
    s = str(o)
    assert "foo" not in s
