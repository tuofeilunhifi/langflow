import logging
import os
from functools import wraps
from typing import Any, Dict, List, Optional, overload, Iterable, Literal
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.docarray.base import (
    _check_docarray_import,
)
from ossio import ENVOSSIO, get_oss_config

OSSPREFIX = 'oss://'
DAPOSTFIX = '.bin'

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


def assert_database_exists(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        self = args[0]
        if len(args) > 1:
            database_name = args[1]
        else:
            database_name = kwargs.get('database_name')

        if database_name is None:
            return f(*args, **kwargs)

        if self.docarray_dict.get(database_name, None) is not None:
            return f(*args, **kwargs)
        else:
            fname = str(f.__name__)
            error_msg = f"Database {database_name} is not exits in function {fname}."
            raise Exception(error_msg)

    return decorated


def assert_embedding_exists(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        self = args[0]
        if getattr(self, 'embedding', None) is not None:
            return f(*args, **kwargs)
        else:
            raise Exception('EasyRetrievalDocArray embedding is not exits, should call set_embedding() before')

    return decorated


class EasyRetrievalDocArrayIndex(object):
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
        self.docarray_dict = {}
        self.embedding = None
        self.load()

    def set_root_path(
        self,
        root_path: str,
    ):
        if root_path.startswith(OSSPREFIX):
            if ENVOSSIO.is_accessed('ENVOSSIO'):
                target_path = os.path.join(root_path, '*' + DAPOSTFIX)
                list_dir = ENVOSSIO.glob(target_path)
                logging.info('root_path contains \
                    database {} '.format(list_dir))
        self.root_path = root_path
        return


    def set_embedding(
        self,
        embedding
    ):
        # every database in EasyRetrievalDocArray will use this embedding
        if embedding is not None:
            self.embedding = embedding
        for da in self.docarray_dict.values():
            da.embedding = self.embedding
        return


    def set_oss_config(self, oss_config: Optional[Dict[str, str]]):
        ENVOSSIO.access_oss(**oss_config)
        return


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

        if type(path) is str:
            target_path = os.path.join(path, '*' + DAPOSTFIX)
            list_dir = ENVOSSIO.glob(target_path)
            # get all docarray object binaryfile in root_path, load them
            da_bin = {}
            for filepath in list_dir:
                if filepath.endswith(DAPOSTFIX):
                    da_name = filepath.split('/')[-1][:-4]
                    da_bin[da_name] = da_bin
                    if filepath.startswith(OSSPREFIX):
                        tname = f'./{da_name}{DAPOSTFIX}'
                        ENVOSSIO.copy(filepath, tname)
                        self.docarray_dict[da_name] = WrapperDocArrayInMemorySearch.load(
                            self.embedding,tname)
                    else:
                        self.docarray_dict[da_name] = WrapperDocArrayInMemorySearch.load(
                            self.embedding,filepath)
        return {'load': self.get(get_all=True)}

    @overload
    def save(self):
        """Load an EasyRetrievalDocArray to root_path"""
        ...


    def save(
        self,
        path: Optional[str] = None,
    ):
        if path is None:
            path = self.root_path
        for da_name, da in self.docarray_dict.items():
            if type(path) is str:
                da_output_path = os.path.join(path, da_name + DAPOSTFIX)
                if path.startswith(OSSPREFIX):
                    local_name = da_output_path.split('/')[-1]
                    da.doc_index.persist(local_name)
                    ENVOSSIO.copy(local_name, da_output_path)
                else:
                    da.doc_index.persist(da_output_path)
        return {'save': 'Done'}

    # -------------------------------------------------------------------------------#
    @assert_database_exists
    def get(
        self,
        database_name: str = None,
        get_all: bool = False,
    ):
        """get EasyRetrievalDocArray all database info"""

        # get all database name
        if get_all:
            da_name_ls = []
            for da_name, _ in self.docarray_dict.items():
                da_name_ls.append({'database_name': da_name, 'length': self.db_get(da_name)['db_get']})
            return {'get': da_name_ls}
        
        # get database for retrieval
        return self.docarray_dict[database_name]

    @assert_embedding_exists
    def add(
        self,
        database_name: str,
        docs = None,
    ):   
        if database_name in self.docarray_dict.keys():
            raise Exception('database_name {} already exists'.format(database_name))
        if docs is None:
            self.docarray_dict[database_name] = WrapperDocArrayInMemorySearch.from_params(self.embedding)
            lenght = 0
        else:
            self.docarray_dict[database_name] = WrapperDocArrayInMemorySearch.from_documents(docs,self.embedding)
            lenght = self.db_get(database_name)['db_get']
        self.save()
        return {'add': "{database_name}".format(database_name=database_name),'length':lenght}

    @assert_database_exists
    def delete(
        self,
        database_name: str,
    ):
        del self.docarray_dict[database_name]
        if type(self.root_path) is str:
            output_path = os.path.join(self.root_path,
                                       database_name + DAPOSTFIX)
            if ENVOSSIO.exists(output_path):
                try:
                    ENVOSSIO.remove(output_path)
                except BaseException:
                    raise Exception('ENVOSSIO remove {output_path} \
                        failed'.format(output_path=output_path))
        self.save()
        return {'delete': "{database_name} delete successfully".format(database_name=database_name)}

    # ------------------------------------------------------------------------------------#
    @assert_database_exists
    @assert_embedding_exists
    def db_set(
        self,
        database_name: str,
        texts: Iterable[str],
    ):
        if texts is None:
            raise Exception('texts is None')
        self.docarray_dict[database_name].add_texts(texts)
        return {'db_set': self.db_get(database_name)['db_get']}
    
    @assert_database_exists
    def db_get(
        self,
        database_name: str,
    ):
        db = self.docarray_dict[database_name].doc_index
        return {'db_get': db.num_docs()}

    @assert_database_exists
    @assert_embedding_exists
    def db_delete(
        self,
        database_name: str,
        querys: Iterable[str],
    ):
        if querys is None and embeddings is None:
            raise Exception('querys is None')

        if querys is not None:
            delete_info = []
            for query in querys:
                query_embedding = self.embedding.embed_query(query)
                query_doc = self.docarray_dict[database_name].doc_cls(embedding=query_embedding)  # type: ignore
                docs, scores = self.docarray_dict[database_name].doc_index.find(query_doc, search_field="embedding", limit=1)
                if(docs[0].text == query):
                    del self.docarray_dict[database_name].doc_index[docs[0].id]
                    delete_info.append({'text': query, 'delete': 'success'})
                else:
                    delete_info.append({'text': query, 'delete': 'fail'})
            return {'db_delete': delete_info}
    
    @assert_database_exists
    @assert_embedding_exists
    def db_search(
        self,
        database_name: str,
        querys: Iterable[str],
        embedding: List[float] = None,
        top_k: int = 10,
    ):
        if querys is None and embeddings is None:
            raise Exception('query and embedding is None')
        
        result = []
        if querys is not None:
            for query in querys:
                docs = self.docarray_dict[database_name].similarity_search(query=query,k=top_k)
                result.append({'text': query, 'result': docs})
        
        if embedding is not None:
            for embedding in embeddings:
                docs = self.docarray_dict[database_name].similarity_search_by_vector(embedding=embedding,k=top_k)
                result.append({'result': docs})
            
        return {'db_search': result}