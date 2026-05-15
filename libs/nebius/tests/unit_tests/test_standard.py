"""Standard LangChain interface unit tests for ChatNebius."""

from langchain_core.language_models import BaseChatModel
from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_nebius import ChatNebius


class TestNebiusStandard(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatNebius

    @property
    def chat_model_params(self) -> dict:
        return {
            "model": "meta-llama/Llama-3.3-70B-Instruct-fast",
            "api_key": "test_api_key",
        }

    @property
    def init_from_env_params(self) -> tuple[dict, dict, dict]:
        return (
            {
                "NEBIUS_API_KEY": "api_key",
                "NEBIUS_API_BASE": "https://base.com",
            },
            {"model": "meta-llama/Llama-3.3-70B-Instruct-fast"},
            {
                "nebius_api_key": "api_key",
                "nebius_api_base": "https://base.com",
            },
        )
