"""Standard LangChain interface integration tests for ChatNebius.

Requires NEBIUS_API_KEY in the environment. The default model is the
"fast" Llama 3.3 70B variant; per-test override available via
chat_model_params.
"""

from langchain_core.language_models import BaseChatModel
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_nebius import ChatNebius


class TestNebiusStandardIntegration(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatNebius

    @property
    def chat_model_params(self) -> dict:
        return {"model": "meta-llama/Llama-3.3-70B-Instruct-fast"}

    @property
    def has_tool_calling(self) -> bool:
        return True

    @property
    def has_structured_output(self) -> bool:
        return True

    @property
    def supports_json_mode(self) -> bool:
        return True

    @property
    def supports_image_inputs(self) -> bool:
        return False

    @property
    def supports_anthropic_inputs(self) -> bool:
        return False

    @property
    def returns_usage_metadata(self) -> bool:
        return True
