from typing import List
from langchain.text_splitter import TextSplitter

class IdentitySplitter(TextSplitter):
    """Interface for splitting text into chunks."""

    def split_text(self, text: str) -> List[str]:
        sent_list = [text]
        return sent_list

CUSTOM_TEXT_SPLITTERS = {
    "IdentitySplitter": IdentitySplitter,
}