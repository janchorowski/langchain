"""
Pathway vector store server and client.


The PathwayVectorServer builds a pipeline which indexes all files in a given folder,
embeds them and builds a vector index. The pipeline reacts to changes in source files,
automatically updating appropriate index entries.

The PathwayVectorClient implements the LangChain VectorStore interface and queries the
PathwayVectorServer to retrieve up-to-date documents.

"""

from typing import Callable, List, Optional, Tuple

import json
import requests

from langchain_core.documents import BaseDocumentTransformer, Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


# Copied from https://github.com/pathwaycom/pathway/blob/main/python/pathway/xpacks/llm/vector_store.py
# to remove dependency on Pathway library.
class _VectorStoreClient:
    """
    A client you can use to query :py:class:`VectorStoreServer`.

    Args:
        - host: host on which `:py:class:`VectorStoreServer` listens
        - port: port on which `:py:class:`VectorStoreServer` listens
    """

    def __init__(self, host, port):
        self.host = host
        self.port = port

    def query(
        self, query: str, k: int = 3, metadata_filter: Optional[str] = None
    ) -> list[dict]:
        """
        Perform a query to the vector store and fetch results.

        Args:
            - query:
            - k: number of documents to be returned
            - metadata_filter: optional string representing the metadata filtering query
                in the JMESPath format. The search will happen only for documents
                satisfying this filtering.
        """

        data = {"query": query, "k": k}
        if metadata_filter is not None:
            data["metadata_filter"] = metadata_filter
        url = f"http://{self.host}:{self.port}/v1/retrieve"
        response = requests.post(
            url,
            data=json.dumps(data),
            headers={"Content-Type": "application/json"},
            timeout=3,
        )
        responses = response.json()
        return sorted(responses, key=lambda x: x["dist"])

    # Make an alias
    __call__ = query

    def get_vectorstore_statistics(self):
        """Fetch basic statistics about the vector store."""
        url = f"http://{self.host}:{self.port}/v1/statistics"
        response = requests.post(
            url,
            json={},
            headers={"Content-Type": "application/json"},
        )
        responses = response.json()
        return responses

    def get_input_files(
        self,
        metadata_filter: Optional[str] = None,
        filepath_globpattern: Optional[str] = None,
    ):
        """
        Fetch information on documents in the the vector store.

        Args:
            metadata_filter: optional string representing the metadata filtering query
                in the JMESPath format. The search will happen only for documents
                satisfying this filtering.
            filepath_globpattern: optional glob pattern specifying which documents
                will be searched for this query.
        """
        url = f"http://{self.host}:{self.port}/v1/inputs"
        response = requests.post(
            url,
            json={
                "metadata_filter": metadata_filter,
                "filepath_globpattern": filepath_globpattern,
            },
            headers={"Content-Type": "application/json"},
        )
        responses = response.json()
        return responses


class PathwayVectorClient(VectorStore):
    """
    VectorStore connecting to PathwayVectorServer realtime data pipeline.
    """

    def __init__(
        self,
        host,
        port,
    ) -> None:
        self.client = _VectorStoreClient(host, port)

    def add_texts(
        self,
        *args,
        **kwargs,
    ) -> List[str]:
        """Pathway is not suitable for this method."""
        raise NotImplementedError(
            "Pathway vector store does not support adding or removing texts"
            " from client."
        )

    @classmethod
    def from_texts(cls, *args, **kwargs):
        raise NotImplementedError(
            "Pathway vector store does not support initializing from_texts."
        )

    def similarity_search(
        self, query: str, k: int = 4, metadata_filter: Optional[str] = None
    ) -> List[Document]:
        rets = self.client(query=query, k=k, metadata_filter=metadata_filter)

        return [
            Document(page_content=ret["text"], metadata=ret["metadata"]) for ret in rets
        ]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        metadata_filter: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        """Run similarity search with Pathway with distance.

        Args:
            - query (str): Query text to search for.
            - k (int): Number of results to return. Defaults to 4.
            - metadata_filter (Optional[str]): Filter by metadata.
                Filtering query should be in JMESPath format. Defaults to None.

        Returns:
            List[Tuple[Document, float]]: List of documents most similar to
            the query text and cosine distance in float for each.
            Lower score represents more similarity.
        """
        rets = self.client(query=query, k=k, metadata_filter=metadata_filter)

        return [
            (Document(page_content=ret["text"], metadata=ret["metadata"]), ret["dist"])
            for ret in rets
        ]

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        return self._cosine_relevance_score_fn

    def get_vectorstore_statistics(self):
        """Fetch basic statistics about the Vector Store."""
        return self.client.get_vectorstore_statistics()
