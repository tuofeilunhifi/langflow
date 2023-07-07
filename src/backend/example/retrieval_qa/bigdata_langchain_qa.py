import os
import torch
from models import ChatGLM, QianWen
import argparse
from typing import Any, Dict, List, Optional
import copy

from langchain.vectorstores import FAISS, DocArrayInMemorySearch

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from typing import List, Tuple
import numpy as np
from langchain.document_loaders import UnstructuredFileLoader, TextLoader, CSVLoader
from textsplitter import ChineseTextSplitter
from langchain.text_splitter import TextSplitter
from loader import UnstructuredPaddlePDFLoader, UnstructuredPaddleImageLoader
from langchain.docstore.document import Document
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders.base import BaseLoader
import pandas as pd

from sentence_transformers import SentenceTransformer

from easy_retrieval_docarrayindex_mini import EasyRetrievalDocArrayIndexMini

class IdentitySplitter(TextSplitter):
    """Interface for splitting text into chunks."""

    def split_text(self, text: str) -> List[str]:
        sent_list = [text]
        return sent_list


class QACSVLoader(BaseLoader):
    """Loads a CSV file into a list of documents.

    Each document represents one row of the CSV file. Every row is converted into a
    key/value pair and outputted to a new line in the document's page_content.

    The source for each document loaded from csv is set to the value of the
    `file_path` argument for all doucments by default.
    You can override this by setting the `source_column` argument to the
    name of a column in the CSV file.
    The source of each document will then be set to the value of the column
    with the name specified in `source_column`.

    Output Example:
        .. code-block:: txt

            column1: value1
            column2: value2
            column3: value3
    """

    def __init__(
        self,
        file_path: str,
        source_column: Optional[str] = None,
        csv_args: Optional[Dict] = None,
        encoding: Optional[str] = None,
    ):
        self.file_path = file_path
        self.source_column = source_column
        self.encoding = encoding
        self.csv_args = csv_args or {}

    def load(self) -> List[Document]:
        """Load data into document objects."""

        # data = pd.read_csv('./dataworks_QA_data.csv')
        data = pd.read_csv(self.file_path)
        data = data[['问题', '回答']]
        docs = []
        # q = []
        for i in range(len(data)):
            question = str(data.loc[i, '问题'])
            answer = str(data.loc[i, '回答'])
            # q.append(question)
            text = question + ':' + answer
            doc = Document(page_content=text)
            docs.append(doc)
        return docs

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

def load_file(filepath, sentence_size=100):
    if filepath.lower().endswith(".md"):
        loader = UnstructuredFileLoader(filepath, mode="elements")
        docs = loader.load()
    elif filepath.lower().endswith(".txt"):
        loader = TextLoader(filepath, autodetect_encoding=True)
        textsplitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
        docs = loader.load_and_split(textsplitter)
    elif filepath.lower().endswith(".pdf"):
        loader = UnstructuredPaddlePDFLoader(filepath)
        textsplitter = ChineseTextSplitter(pdf=True, sentence_size=sentence_size)
        docs = loader.load_and_split(textsplitter)
    elif filepath.lower().endswith(".jpg") or filepath.lower().endswith(".png"):
        loader = UnstructuredPaddleImageLoader(filepath, mode="elements")
        textsplitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
        docs = loader.load_and_split(text_splitter=textsplitter)
    else:
        loader = UnstructuredFileLoader(filepath, mode="elements")
        textsplitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
        docs = loader.load_and_split(text_splitter=textsplitter)
    write_check_file(filepath, docs)
    return docs

def write_check_file(filepath, docs):
    folder_path = os.path.join(os.path.dirname(filepath), "tmp_files")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    fp = os.path.join(folder_path, 'load_file.txt')
    with open(fp, 'a+', encoding='utf-8') as fout:
        fout.write("filepath=%s,len=%s" % (filepath, len(docs)))
        fout.write('\n')
        for i in docs:
            fout.write(str(i))
            fout.write('\n')
        fout.close()

def seperate_list(ls: List[int]) -> List[List[int]]:
    lists = []
    ls1 = [ls[0]]
    for i in range(1, len(ls)):
        if ls[i - 1] + 1 == ls[i]:
            ls1.append(ls[i])
        else:
            lists.append(ls1)
            ls1 = [ls[i]]
    lists.append(ls1)
    return lists


class FAISSWrapper(FAISS):
    chunk_size = 250
    chunk_conent = True
    score_threshold = 0
    
    def similarity_search_with_score_by_vector(
            self, embedding: List[float], k: int = 4
    ) -> List[Tuple[Document, float]]:
        scores, indices = self.index.search(np.array([embedding], dtype=np.float32), k)
        docs = []
        id_set = set()
        store_len = len(self.index_to_docstore_id)
        for j, i in enumerate(indices[0]):
            if i == -1 or 0 < self.score_threshold < scores[0][j]:
                # This happens when not enough docs are returned.
                continue
            _id = self.index_to_docstore_id[i]
            doc = self.docstore.search(_id)
            if not self.chunk_conent:
                if not isinstance(doc, Document):
                    raise ValueError(f"Could not find document for id {_id}, got {doc}")
                doc.metadata["score"] = int(scores[0][j])
                docs.append(doc)
                continue
            id_set.add(i)
            docs_len = len(doc.page_content)
            for k in range(1, max(i, store_len - i)):
                break_flag = False
                for l in [i + k, i - k]:
                    if 0 <= l < len(self.index_to_docstore_id):
                        _id0 = self.index_to_docstore_id[l]
                        doc0 = self.docstore.search(_id0)
                        if docs_len + len(doc0.page_content) > self.chunk_size:
                            break_flag = True
                            break
                        elif doc0.metadata["source"] == doc.metadata["source"]:
                            docs_len += len(doc0.page_content)
                            id_set.add(l)
                if break_flag:
                    break
        if not self.chunk_conent:
            return docs
        if len(id_set) == 0 and self.score_threshold > 0:
            return []
        id_list = sorted(list(id_set))
        id_lists = seperate_list(id_list)
        for id_seq in id_lists:
            for id in id_seq:
                if id == id_seq[0]:
                    _id = self.index_to_docstore_id[id]
                    doc = self.docstore.search(_id)
                else:
                    _id0 = self.index_to_docstore_id[id]
                    doc0 = self.docstore.search(_id0)
                    doc.page_content += " " + doc0.page_content
            if not isinstance(doc, Document):
                raise ValueError(f"Could not find document for id {_id}, got {doc}")
            doc_score = min([scores[0][id] for id in [indices[0].tolist().index(i) for i in id_seq if i in indices[0]]])
            doc.metadata["score"] = int(doc_score)
            docs.append((doc, doc_score))
        return docs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', required=True, type=str)
    parser.add_argument('--llm', type=str, default='chatglm')
    parser.add_argument('--embedding', type=str, default='sgpt')
    parser.add_argument('--oss_path', required=True, type=str, default='oss://xxx/')
    parser.add_argument('--oss_config', required=True, type=str, default='xxx/.ossutilconfig')
    args = parser.parse_args()

    # load docs
    filepath = args.file_path
    oss_path = args.oss_path
    oss_config = args.oss_config

    # LLM name
    LLM_TYPE = args.llm

    # Embedding model name
    EMBEDDING_MODEL = args.embedding

    # 基于上下文的prompt模版，请务必保留"{question}"和"{context_str}"
    PROMPT_TEMPLATE = """已知信息：
    {context_str} 

    根据上述已知信息，简洁和专业的来回答用户的问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题” 或 “没有提供足够的相关信息”，不允许在答案中添加编造成分，答案请使用中文。 问题是：{question}"""

    REFINE_PROMPT_TEMPLATE = """已知信息：
    {context_str} 
    已有回答: {existing_answer}

    根据上述已知信息，简洁和专业的来回答用户的问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题” 或 “没有提供足够的相关信息”，不允许在答案中添加编造成分，答案请使用中文。 问题是：{question}"""

    # Embedding running device
    EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # return top-k text chunk from vector store
    VECTOR_SEARCH_TOP_K = 5

    # 文本分句长度
    SENTENCE_SIZE = 100

    CHAIN_TYPE = 'stuff'

    llm_model_dict = {
        "qianwen": QianWen,
        "chatglm": ChatGLM
    }

    embedding_model_dict = {
        "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
        "ernie-base": "nghuyong/ernie-3.0-base-zh",
        "text2vec-base": "shibing624/text2vec-base-chinese",
        "text2vec": "/tmp/retrieval_qa/GanymedeNil_text2vec-large-chinese",
        "sgpt": "Muennighoff/SGPT-125M-weightedmean-msmarco-specb-bitfit",
    }


    print("loading model start")
    llm = llm_model_dict[LLM_TYPE]()

    if EMBEDDING_MODEL == 'sgpt':
        embeddings = HuggingFaceEmbeddingsSpecb(model_name=embedding_model_dict[EMBEDDING_MODEL], model_kwargs={'device': EMBEDDING_DEVICE})
    else:
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict[EMBEDDING_MODEL],
                                                model_kwargs={'device': EMBEDDING_DEVICE})
    print("loading model done")

    print("loading documents start")
    # docs = load_file(filepath, sentence_size=SENTENCE_SIZE)
    loader = QACSVLoader(filepath)
    text_splitter = IdentitySplitter()
    docs = text_splitter.split_documents(loader.load())
    print("loading documents done")

    print("embedding start")
    # docsearch = DocArrayInMemorySearch.from_documents(docs, embeddings)

    docsearch = EasyRetrievalDocArrayIndexMini.from_documents(
        database_name='dataworks',
        docs=docs,
        embedding=embeddings,
        root_path=oss_path,
        oss_config_file=oss_config,
    )

    print("embedding done")

    print("loading qa start")
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE, input_variables=["context_str", "question"]
    )
    refine_prompt = PromptTemplate(
        input_variables=["context_str", "existing_answer", "question"],
        template=REFINE_PROMPT_TEMPLATE,
    )

    if CHAIN_TYPE == 'stuff':
        chain_type_kwargs = {"prompt": prompt, "document_variable_name": "context_str"}
    elif CHAIN_TYPE == 'refine':
        chain_type_kwargs = {"question_prompt": prompt, "refine_prompt": refine_prompt}

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=CHAIN_TYPE, 
        retriever=docsearch.as_retriever(search_kwargs={"k": VECTOR_SEARCH_TOP_K}), 
        chain_type_kwargs=chain_type_kwargs)
    print("loading qa done")

    print("qa")
    while True:
        print("请输入:")
        query = input("")
        if len(query.strip())==0:
            break
        print(qa.run(query))