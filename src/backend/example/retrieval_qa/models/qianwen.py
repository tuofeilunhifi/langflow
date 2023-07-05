
from abc import ABC

from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional

from dashscope import Generation
from http import HTTPStatus


class QianWen(LLM, ABC):
    model = "qwen-plus-v1"

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "QianWen"

    @property
    def _history_len(self) -> int:
        return self.history_len

    def set_history_len(self, history_len: int = 10) -> None:
        self.history_len = history_len

    def _call(self, prompt: str, 
        stop: Optional[List[str]] = None) -> str:
        """_call
        """
        if isinstance(prompt, list):
            prompt = prompt[0]

        response = Generation.call(prompt=prompt, model=self.model)
        if response.status_code==HTTPStatus.OK:
            return response.output['text']
        else:
            return 'Code: %d, status: %s, message: %s' % (response.status_code, response.code, response.message)
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters.
        """
        _param_dict = {
            "model": self.model
        }
        return _param_dict
    