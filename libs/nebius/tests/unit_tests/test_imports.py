from langchain_nebius import __all__

EXPECTED_ALL = ["ChatNebius", "NebiusEmbeddings"]


def test_all_imports() -> None:
    assert sorted(EXPECTED_ALL) == sorted(__all__)
