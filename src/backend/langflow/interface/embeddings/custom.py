from typing import Any
from sentence_transformers import SentenceTransformer
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

class SentenceTransformerSpecb(SentenceTransformer):
    # Requires:
    # pip install git+https://github.com/Muennighoff/sentence-transformers.git@sgpt_poolings_specb
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tokens = ["[SOS]", "{SOS}"]
        self._first_module().tokenizer.add_tokens(tokens, special_tokens=True)
        self._first_module().auto_model.resize_token_embeddings(len(self._first_module().tokenizer))
        # Will be replaced with the rep tokens in the model ones
        # The problem is we don't know if a text is query or document when tokenizing in the Transformer.py module, 
        # so we use the SOS tokens as an identifier if we have a query or document at hand & then replace them
        # If we would directly use the brackets here, they may become part of another token
        self._first_module().bos_spec_token_q = self._first_module().tokenizer.encode("[SOS]", add_special_tokens=False)[0]
        self._first_module().bos_spec_token_d = self._first_module().tokenizer.encode("{SOS}", add_special_tokens=False)[0]
        self._first_module().bos_spec_token_q_rep = self._first_module().tokenizer.encode("[", add_special_tokens=False)[0]
        self._first_module().eos_spec_token_q = self._first_module().tokenizer.encode("]", add_special_tokens=False)[0]
        self._first_module().bos_spec_token_d_rep = self._first_module().tokenizer.encode("{", add_special_tokens=False)[0]
        self._first_module().eos_spec_token_d = self._first_module().tokenizer.encode("}", add_special_tokens=False)[0]
        self._first_module().replace_bos = True

    def encode(self, sentences, **kwargs):
        is_query = kwargs.pop("is_query", True)
        if is_query:
            sentences = "[SOS]" + sentences if isinstance(sentences, str) else ["[SOS]" + sent for sent in sentences]
        else:
            sentences = "{SOS}" + sentences if isinstance(sentences, str) else ["{SOS}" + sent for sent in sentences]    
        return super().encode(sentences, **kwargs)

class HuggingFaceEmbeddingsSpecb(HuggingFaceEmbeddings):
    def __init__(self, **kwargs: Any):
        """Initialize the sentence_transformer."""
        super().__init__(**kwargs)

        self.client = SentenceTransformerSpecb(
            self.model_name, cache_folder=self.cache_folder, **self.model_kwargs
        )

CUSTOM_EMBEDDINGS = {
    "HuggingFaceEmbeddingsSpecb": HuggingFaceEmbeddingsSpecb,
}