import json
import requests
from abc import ABC

from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional


from langchain.callbacks.manager import CallbackManagerForLLMRun

# from transformers import AutoTokenizer, AutoModel

# tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
# model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
# model.eval()


class ChatGLM(LLM, ABC):
    max_tokens: int = 10000
    temperature: float = 0.01
    top_p: float = 0.9
    history_len: int = 3
    hosts: Optional[str] = None
    authorization: Optional[str] = None

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
        # response, _ = model.chat(tokenizer, prompt, history=[], max_length=self.max_token, top_p=self.top_p, temperature=self.temperature)

        datas = json.dumps({
            "prompt": prompt,
            "max_length": self.max_tokens,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "history": []
        })
        head = {
        "Authorization": self.authorization
    }

        r = requests.post(self.hosts, data=datas, headers=head)
        data = r.content.decode('utf-8')
        data = json.loads(data)
        response = data['response']
        return response
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "history_len": self.history_len,
                "hosts": self.hosts,
                "authorization": self.authorization}

CUSTOM_LLMS = {
    "ChatGLM": ChatGLM,
}