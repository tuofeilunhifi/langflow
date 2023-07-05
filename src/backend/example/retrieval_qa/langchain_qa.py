import os
import torch
from models import ChatGLM, QianWen
import argparse

from langchain.vectorstores import FAISS, DocArrayInMemorySearch

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from typing import List, Tuple
import numpy as np
from langchain.document_loaders import UnstructuredFileLoader, TextLoader
from textsplitter import ChineseTextSplitter
from loader import UnstructuredPaddlePDFLoader, UnstructuredPaddleImageLoader
from langchain.docstore.document import Document
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import RetrievalQA

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
    parser.add_argument('--embedding', type=str, default='text2vec')
    args = parser.parse_args()

    # load docs
    filepath = args.file_path

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

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict[EMBEDDING_MODEL],
                                            model_kwargs={'device': EMBEDDING_DEVICE})
    print("loading model done")

    print("loading documents start")
    docs = load_file(filepath, sentence_size=SENTENCE_SIZE)
    print("loading documents done")

    print("embedding start")
    docsearch = FAISSWrapper.from_documents(docs, embeddings)
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