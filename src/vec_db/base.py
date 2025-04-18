"""
This module defines the base class for a vector database,
 which is used to store, retrieve, and compute similarity between vectors.
"""

from __future__ import annotations
from typing import Dict, List, Union, Any


class VectorDatabase:
    """
    A base class to implement vector database methods.
    """

    def __init__(self, logger: object, cfg: Dict[str, Any]) -> None:
        """
        Initialize the vector database.

        Args:
            logger (object): Logger instance for tracking operations.
            cfg (Dict[str, Any]): Configuration dictionary for database settings.
        """
        self.logger = logger
        self.cfg = cfg
        self.logger.info("VectorDatabase initialized with provided configuration.")
        self.client = None

    def __enter__(self):
        """
        Context manager entry : Establish a connection to the vector database.
        """
        self.logger.info("Establishing database connection.")
        self.client = self.connect_db()
        return self

    def __exit__(self, *_):
        """
        Context manager exit : Close the connection to the vector database.
        """
        self.logger.info("Closing database connection.")
        if self.client:
            self.client.close()
            self.client = None

    def connect_db(self):
        """
        Abstract method for establishing a database connection.
        """
        raise NotImplementedError("connect_db must be implemented in a subclass.")

    def init_db(self) -> None:
        """
        Initialize the vector database structure.
        """
        raise NotImplementedError("Method not implemented yet.")

    def is_collection_empty(self) -> bool:
        """
        Check if the collection is empty.

        Returns:
            bool: True if the collection is empty, False otherwise.
        """
        raise NotImplementedError("Method not implemented yet.")

    def ingest_from_file(self, file_path: str) -> None:
        """
        Ingest data from a file into the vector database.

        Args:
            file_path (str): The path to the file to ingest.
        """
        raise NotImplementedError("Method not implemented yet.")

    def add_vector(self) -> None:
        """
        Add a vector to the vector database.

        Args:
            vector (List[float]): The vector to be added.
            metadata (Dict[str, Any]): The metadata associated with the vector.
        """
        raise NotImplementedError("Method not implemented yet.")

    def query_loop(self, top_k: int) -> None:
        """
        Start a query loop for the vector database.

        Args:
            top_k (int): Number of top similar vectors to retrieve.
        """
        raise NotImplementedError("Method not implemented yet.")

    def retrieve_similar_facts(
        self, query: str, top_k: int = 2
    ) -> Union[List[str], None]:
        """
        Retrieve the most similar facts based on the query.

        Args:
            query (str): The input query for finding similar facts.
            top_k (int): Number of top similar facts to return.

        Returns:
            Union[List[str], None]: The top_k most similar facts.
        """
        raise NotImplementedError("Method not implemented yet.")

    def insert_data(self) -> None:
        """
        Insert new data into the vector database.
        """
        raise NotImplementedError("Method not implemented yet.")

    def delete_data(self) -> None:
        """
        Delete data from the vector database.
        """
        raise NotImplementedError("Method not implemented yet.")
