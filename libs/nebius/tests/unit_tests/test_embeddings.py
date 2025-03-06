"""Test embedding model integration."""

import os

import pytest  # type: ignore[import-not-found]

from langchain_nebius import NebiusEmbeddings


def test_initialization() -> None:
    """Test embedding model initialization."""
    NebiusEmbeddings()


def test_nebius_invalid_model_kwargs() -> None:
    with pytest.raises(ValueError):
        NebiusEmbeddings(model_kwargs={"model": "foo"})


def test_nebius_incorrect_field() -> None:
    with pytest.warns(match="not default parameter"):
        llm = NebiusEmbeddings(foo="bar")  # type: ignore[call-arg]
    assert llm.model_kwargs == {"foo": "bar"}
