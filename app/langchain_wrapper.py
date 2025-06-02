from typing import List, Optional
from langchain.schema import LLMResult, Generation
from langchain.llms.base import BaseLLM
from pydantic import BaseModel, PrivateAttr

# Import your existing GeminiClient here
from app.llm_client import GeminiClient

class GeminiLangChainWrapper(BaseLLM, BaseModel):
    # Tell pydantic: "Hey, ignore this field in your schema validation"
    _client: GeminiClient = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = GeminiClient()  # initialize your client here

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
        generations = []
        for prompt in prompts:
            text = self._client.generate_user_story(prompt)
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)

    @property
    def _llm_type(self) -> str:
        return "gemini-custom"
