from typing import Any
# from docarray import Document, DocumentArray
import torch
from transformers import AutoModel, AutoTokenizer
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.vectorstores import DocArrayInMemorySearch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pandas as pd

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

# def tokenize_with_specb(texts, is_query):
#     # Tokenize without padding
#     batch_tokens = tokenizer(texts, padding=False, truncation=True)
#     # Add special brackets & pay attention to them
#     for seq, att in zip(batch_tokens["input_ids"], batch_tokens["attention_mask"]):
#         if is_query:
#             seq.insert(0, SPECB_QUE_BOS)
#             seq.append(SPECB_QUE_EOS)
#         else:
#             seq.insert(0, SPECB_DOC_BOS)
#             seq.append(SPECB_DOC_EOS)
#         att.insert(0, 1)
#         att.append(1)
#     # Add padding
#     batch_tokens = tokenizer.pad(batch_tokens, padding=True, return_tensors="pt")
#     return batch_tokens

# def get_weightedmean_embedding(batch_tokens, model):
#     # Get the embeddings
#     with torch.no_grad():
#         # Get hidden state of shape [bs, seq_len, hid_dim]
#         last_hidden_state = model(**batch_tokens, output_hidden_states=True, return_dict=True).last_hidden_state

#     # Get weights of shape [bs, seq_len, hid_dim]
#     weights = (
#         torch.arange(start=1, end=last_hidden_state.shape[1] + 1)
#         .unsqueeze(0)
#         .unsqueeze(-1)
#         .expand(last_hidden_state.size())
#         .float().to(last_hidden_state.device)
#     )

#     # Get attn mask of shape [bs, seq_len, hid_dim]
#     input_mask_expanded = (
#         batch_tokens["attention_mask"]
#         .unsqueeze(-1)
#         .expand(last_hidden_state.size())
#         .float()
#     )

#     # Perform weighted mean pooling across seq_len: bs, seq_len, hidden_dim -> bs, hidden_dim
#     sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded * weights, dim=1)
#     sum_mask = torch.sum(input_mask_expanded * weights, dim=1)

#     embeddings = sum_embeddings / sum_mask

#     return embeddings

if __name__ == '__main__':
    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    # tokenizer = AutoTokenizer.from_pretrained("Muennighoff/SGPT-125M-weightedmean-msmarco-specb-bitfit")
    # model = AutoModel.from_pretrained("Muennighoff/SGPT-125M-weightedmean-msmarco-specb-bitfit")
    # model.eval()

    embeddings = HuggingFaceEmbeddingsSpecb(model_name="Muennighoff/SGPT-125M-weightedmean-msmarco-specb-bitfit")

    # SPECB_QUE_BOS = tokenizer.encode("[", add_special_tokens=False)[0]
    # SPECB_QUE_EOS = tokenizer.encode("]", add_special_tokens=False)[0]

    # SPECB_DOC_BOS = tokenizer.encode("{", add_special_tokens=False)[0]
    # SPECB_DOC_EOS = tokenizer.encode("}", add_special_tokens=False)[0]

    data = pd.read_csv('./dataworks_QA_data.csv')
    data = data[['问题', '回答']]
    chunks = []
    # q = []
    for i in range(len(data)):
        question = str(data.loc[i, '问题'])
        answer = str(data.loc[i, '回答'])
        # q.append(question)
        text = question + ':' + answer
        doc = Document(page_content=text)
        chunks.append(doc)
    
    db = DocArrayInMemorySearch.from_documents(chunks, embeddings)
    print(db)
    exit()

    group_ls = []
    intra_ls = []
    for i in range(len(chunks)):
        group_ls.append('0')
        intra_ls.append(0)

    batch_size = 128
    index = 0
    num_batches = (len(chunks) + batch_size - 1) // batch_size

    print('set data begin:')
    for i in tqdm(range(num_batches), desc='Processing'):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(chunks))
        # batch = q[start:end]
        batch_text = chunks[start:end]
        # batch_embeddings = model.client.encode(batch, is_query=False) # get_weightedmean_embedding(tokenize_with_specb(batch, is_query=False), model)
        batch_embeddings = DocArrayInMemorySearch.from_documents(batch_text, embeddings)
        print(batch_embeddings)
        # batch_doc = [
        #     Document(
        #         text=batch_text[tidx] if batch_text[tidx] is not None else None,
        #         embedding=batch_embeddings[tidx].numpy() if batch_text[tidx] is not None else None,
        #         tags=dict(
        #             group_id=group_ls[tidx],
        #             intra_id=intra_ls[tidx],
        #         )
        #     ) for tidx in range(len(batch))
        # ]
        # batch_da = DocumentArray(batch_doc)

        if i == 0:
            support_da = batch_da
        else:
            for item in batch_da:
                support_da.append(item)

    save_name = 'DataWorks_qa.dabin'
    support_da.save(save_name)
