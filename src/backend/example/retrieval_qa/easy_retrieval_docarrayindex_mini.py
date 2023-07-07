import os
from typing import Any, Dict, Optional, overload, Iterable, Literal
from langchain.vectorstores import DocArrayInMemorySearch
from ossio import ENVOSSIO, get_oss_config
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.docarray.base import (
    _check_docarray_import,
)

class WrapperDocArrayInMemorySearch(DocArrayInMemorySearch):
    @classmethod
    def load(
        cls,
        embedding: Embeddings,
        data_path: str,
        metric: Literal[
            "cosine_sim", "euclidian_dist", "sgeuclidean_dist"
        ] = "cosine_sim",
        **kwargs: Any,
    ) -> DocArrayInMemorySearch:
        """Initialize DocArrayInMemorySearch store.

        Args:
            embedding (Embeddings): Embedding function.
            metric (str): metric for exact nearest-neighbor search.
                Can be one of: "cosine_sim", "euclidean_dist" and "sqeuclidean_dist".
                Defaults to "cosine_sim".
            **kwargs: Other keyword arguments to be passed to the get_doc_cls method.
        """
        _check_docarray_import()
        from docarray.index import InMemoryExactNNIndex

        doc_cls = cls._get_doc_cls(space=metric, **kwargs)
        doc_index = InMemoryExactNNIndex[doc_cls](index_file_path=data_path)  # type: ignore
        return cls(doc_index, embedding)

OSSPREFIX = 'oss://'
DAPOSTFIX = '.bin'

class EasyRetrievalDocArrayIndexMini(object):


    @overload
    def __init__(
        self,
        root_path: Optional[str],
    ):
        """Create an EasyRetrievalDocArray wit root_path."""
        ...

    @overload
    def __init__(
        self,
        root_path: Optional[str],
        oss_config: Optional[Dict[str, str]] = None,
    ):
        """Create an EasyRetrievalDocArray wit oss_config."""
        ...

    @overload
    def __init__(
        self,
        root_path: Optional[str],
        oss_config_file: Optional[str] = None,
    ):
        """Create an EasyRetrievalDocArray wit oss_config_file."""
        ...

    def __init__(
        self,
        root_path: Optional[str] = None,
        oss_config: Optional[Dict[str, str]] = None,
        oss_config_file: Optional[str] = None,
    ):
        """Create an EasyRetrievalDocArray wit oss_config and root_path"""
        self.root_path = root_path

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

    @overload
    def load(self):
        """Load an EasyRetrievalDocArray from root_path"""
        ...


    def load(
        self,
        path: Optional[str] = None,
    ):
        if path is None:
            path = self.root_path
        
        load_path = None
        if type(path) is str:
            if path.startswith(OSSPREFIX):
                if ENVOSSIO.exists(path):
                    local_name = path.split('/')[-1]
                    save_path = f'./{local_name}'
                    ENVOSSIO.copy(path, save_path)
                    load_path = save_path
        return load_path

    @overload
    def save(self):
        """Load an EasyRetrievalDocArray to root_path"""
        ...

    def save(
        self,
        path: Optional[str] = None,
        db: Optional[DocArrayInMemorySearch] = None,
    ):
        if path is None:
            path = self.root_path
        local_name = path.split('/')[-1]
        db.doc_index.persist(local_name)
        if path.startswith(OSSPREFIX):
            ENVOSSIO.copy(local_name, path)
        return {'save': 'Done'}

    # -------------------------------------------------------------------------------#
    
    @staticmethod
    def from_documents(
        # cls,
        database_name: str,
        docs : Iterable[str],
        embedding: Embeddings,
        root_path: Optional[str] = None,
        oss_config: Optional[Dict[str, str]] = None,
        oss_config_file: Optional[str] = None,
    ):
        """Create an EasyRetrievalDocArray from documents"""
        erd = EasyRetrievalDocArrayIndexMini(root_path, oss_config, oss_config_file)

        file_path = os.path.join(root_path, database_name + DAPOSTFIX)
        load_path = erd.load(file_path)

        if load_path is not None:
            db = WrapperDocArrayInMemorySearch.load(embedding, load_path)
        else:
            db = WrapperDocArrayInMemorySearch.from_documents(docs, embedding)
            local_name = file_path.split('/')[-1]
            db.doc_index.persist(local_name)
            erd.save(file_path, db)
        return db