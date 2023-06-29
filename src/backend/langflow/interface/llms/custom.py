
from abc import ABC

from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional


from langchain.callbacks.manager import CallbackManagerForLLMRun

from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
model.eval()


class ChatGLM(LLM, ABC):
    max_token: int = 10000
    temperature: float = 0.01
    top_p = 0.9
    # history = []
    history_len: int = 3

    def __init__(self, *args, **kwargs):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "ChatGLM"

    @property
    def _history_len(self) -> int:
        return self.history_len

    def set_history_len(self, history_len: int = 10) -> None:
        self.history_len = history_len

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        response, _ = model.chat(tokenizer, prompt, history=[], max_length=self.max_token, top_p=self.top_p, temperature=self.temperature)
        return response
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"max_token": self.max_token,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "history_len": self.history_len}

CUSTOM_LLMS = {
    "ChatGLM": ChatGLM,
}