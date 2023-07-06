from typing import Dict, List, Optional
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
import pandas as pd

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

CUSTOM_DOCUMENT_LOADERS = {
    "QACSVLoader": QACSVLoader,
}