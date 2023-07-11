from __future__ import annotations
import os
from typing import Dict, Optional, Any, List, Literal
from langflow.interface.vector_store.ossio import ENVOSSIO, get_oss_config
from langchain.embeddings.base import Embeddings
from langchain.docstore.document import Document
from langchain.vectorstores.docarray.base import (
    DocArrayIndex,
    _check_docarray_import,
)

OSSPREFIX = 'oss://'
DAPOSTFIX = '.bin'

class DocArrayInMemorySearchOSS(DocArrayIndex):

    def __init__(
        self,
        embedding: Embeddings,
        doc_index: "BaseDocIndex",
        root_path: Optional[str] = None,
        oss_config: Optional[Dict[str, str]] = None,
        oss_config_file: Optional[str] = None,
    ):
        """Create an DocArrayInMemorySearchOSS with the given embedding and doc_index."""
        self.root_path = root_path
        self.embedding = embedding
        self.doc_index = doc_index

        if root_path is not None:
            if root_path.startswith(OSSPREFIX) and \
                    (oss_config is None and oss_config_file is None):
                raise Exception('EasyRetrievalDocArray with oss root_path\
                        should init with oss_config')

        if oss_config is not None:
            self.oss_config = oss_config
            ENVOSSIO.access_oss(**oss_config)
        if oss_config_file is not None:
            self.oss_config = get_oss_config(oss_config_file)
            ENVOSSIO.access_oss_byconfig(oss_config_file)
    
    @classmethod
    def set_oss_config(
        cls, 
        oss_config: Optional[Dict[str, str]] = None,
        oss_config_file: Optional[str] = None,
    ):
        """Set oss_config for DocArrayInMemorySearchOSS"""
        if oss_config is not None:
            ENVOSSIO.access_oss(**oss_config)
        if oss_config_file is not None:
            oss_config = get_oss_config(oss_config_file)
            ENVOSSIO.access_oss_byconfig(oss_config_file)
        return

    @classmethod
    def load(
        cls,
        path: Optional[str] = None,
    ):  
        """Load an DocArrayInMemorySearchOSS from path"""
        load_path = None
        if type(path) is str:
            if path.startswith(OSSPREFIX):
                if ENVOSSIO.exists(path):
                    local_name = path.split('/')[-1]
                    save_path = f'./{local_name}'
                    ENVOSSIO.copy(path, save_path)
                    load_path = save_path
        return load_path

    def save(
        self,
        doc_index,
        path: Optional[str] = None,
    ):
        """Save an DocArrayInMemorySearchOSS to path and push to oss if path starts with oss://"""
        local_name = path.split('/')[-1]
        self.doc_index.persist(local_name)
        if path.startswith(OSSPREFIX):
            ENVOSSIO.copy(local_name, path)
        return {'save': 'Done'}

    #-------------------------------------------------------------------------------#
    
    @classmethod
    def from_params(
        cls,
        embedding: Embeddings,
        data_path: Optional[str] = None,
        root_path: Optional[str] = None,
        oss_config: Optional[Dict[str, str]] = None,
        oss_config_file: Optional[str] = None,
        metric: Literal[
            "cosine_sim", "euclidian_dist", "sgeuclidean_dist"
        ] = "cosine_sim",
        **kwargs: Any,
    ) -> DocArrayInMemorySearchOSS:

        """Initialize DocArrayInMemorySearchOSS store."""
        _check_docarray_import()
        from docarray.index import InMemoryExactNNIndex

        doc_cls = cls._get_doc_cls(space=metric, **kwargs)
        if data_path is not None:
            doc_index = InMemoryExactNNIndex[doc_cls](index_file_path=data_path)
        else:
            doc_index = InMemoryExactNNIndex[doc_cls]()  # type: ignore
        return cls(embedding, doc_index, root_path, oss_config, oss_config_file)

    @classmethod
    def from_texts(
        cls,
        embedding: Embeddings,
        database_name: str,
        texts: List[str] = None,
        metadatas: Optional[List[Dict[Any, Any]]] = None,
        root_path: Optional[str] = None,
        oss_config: Optional[Dict[str, str]] = None,
        oss_config_file: Optional[str] = None,
        **kwargs: Any,
    ):
        """Create an EasyRetrievalDocArrayIndexMini store and insert data."""

        # construct the file path to determine whether the database exists
        file_path = os.path.join(root_path, database_name + DAPOSTFIX)
        cls.set_oss_config(oss_config=oss_config, oss_config_file=oss_config_file)
        load_path = cls.load(path=file_path)
        
        # if the database exists, load it
        if load_path is not None:
            return cls.from_params(embedding=embedding, 
                                    data_path=load_path,**kwargs)
        # if the database does not exist, create it
        else:
            store = cls.from_params(embedding, 
                                    root_path=root_path,
                                    oss_config=oss_config,
                                    oss_config_file=oss_config_file,**kwargs)
            store.add_texts(texts=texts, metadatas=metadatas)

            local_name = database_name + DAPOSTFIX
            file_path = os.path.join(root_path, local_name)
            store.save(path=file_path, doc_index=store.doc_index)

            return store

    @classmethod
    def from_documents(
        cls,
        database_name: str,
        embedding: Embeddings,
        documents: List[Document] = None,
        root_path: Optional[str] = None,
        oss_config: Optional[Dict[str, str]] = None,
        oss_config_file: Optional[str] = None,
        **kwargs: Any,
    ):
        # construct the file path to determine whether the database exists
        file_path = os.path.join(root_path, database_name + DAPOSTFIX)
        cls.set_oss_config(oss_config=oss_config, oss_config_file=oss_config_file)
        load_path = cls.load(path=file_path)
        
        # if the database exists, load it
        if load_path is not None:
            return cls.from_params(embedding=embedding, 
                                    data_path=load_path,**kwargs)
        
        # if the database does not exist, create it
        else:
            texts = [d.page_content for d in documents]
            metadatas = [d.metadata for d in documents]

            return cls.from_texts(texts=texts, embedding=embedding, 
                                metadatas=metadatas,
                                database_name=database_name,
                                root_path=root_path,
                                oss_config=oss_config,
                                oss_config_file=oss_config_file,**kwargs)

    
CUSTOM_VECTOR_STORES = {
    "DocArrayInMemorySearchOSS": DocArrayInMemorySearchOSS,
}