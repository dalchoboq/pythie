from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional
from vertexai.preview.language_models import TextGenerationModel
from vertexai.preview.language_models import ChatModel, InputOutputTextPair

class PaLM(LLM):
    """
    This is a class to make an existing, Google PaLM Model
    into an LLM that can interface with LangChain.

    Properties:
        model (private): The PaLM model.
    """

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")

        response = TextGenerationModel.from_pretrained("text-bison@001").predict(prompt, max_output_tokens=1024)

        return response.text

    #@property
    #def _identifying_params(self) -> Mapping[str, Any]:
    #    """Get the identifying parameters."""
    #    return {"model": self.model}